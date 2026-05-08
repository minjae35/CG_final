"""Run Grounding DINO + SAM2 to produce a binary mask (native image resolution)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from segmentation.debug_overlay_style import apply_debug_green_mask_overlay


def _ensure_torch_shared_libs_visible() -> None:
    """Best-effort fix for ImportError: libc10.so when loading CUDA extensions.

    Some environments don't include torch's bundled shared libs on the dynamic linker path.
    We avoid requiring users to export LD_LIBRARY_PATH manually by preloading key libs
    from ``site-packages/torch/lib`` with RTLD_GLOBAL before importing ``groundingdino._C``.
    """
    try:
        import ctypes
        import os
        import torch

        torch_lib = (Path(torch.__file__).resolve().parent / "lib").resolve()
        if not torch_lib.is_dir():
            return

        # Also prepend to LD_LIBRARY_PATH for subprocesses / downstream dlopen.
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        torch_lib_s = torch_lib.as_posix()
        if not cur.startswith(torch_lib_s):
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_s}:{cur}" if cur else torch_lib_s

        # Preload commonly required libs. Missing ones are OK.
        for name in (
            "libc10.so",
            "libtorch_cpu.so",
            "libtorch.so",
            "libc10_cuda.so",
            "libtorch_cuda.so",
        ):
            p = torch_lib / name
            if p.is_file():
                try:
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # If it still fails, the caller will raise a detailed exception on import.
                    pass
    except Exception:
        # Never block segmentation on this best-effort helper.
        return


def sam2_hydra_config_name(sam2_config_file: Path) -> str:
    """SAM2 ``build_sam2`` uses Hydra ``compose(config_name=...)`` — needs names like
    ``configs/sam2.1/sam2.1_hiera_l.yaml``, not an absolute filesystem path.
    """
    p = Path(sam2_config_file).resolve().as_posix()
    key = "sam2/configs/"
    if key in p:
        return "configs/" + p.split(key, 1)[1]
    return "configs/sam2.1/sam2.1_hiera_l.yaml"


def _normalize_grounding_caption(text_prompt: str) -> str:
    cap = text_prompt.strip().lower()
    if not cap.endswith("."):
        cap = cap + "."
    return cap


def grounded_dino_best_box_xyxy_confidence(
    image_path: Path | str,
    text_prompt: str,
    *,
    gs2_root: Path,
    gdino_config: Path,
    gdino_checkpoint: Path,
    box_threshold: float,
    text_threshold: float,
    device: str = "cuda",
) -> tuple[np.ndarray, float] | None:
    """Run Grounding DINO only (no SAM2). Return best box in **pixel** ``xyxy`` and score, or ``None``."""
    import cv2
    import torch
    from torchvision.ops import box_convert

    image_path = Path(image_path).resolve()
    gs2_root = Path(gs2_root).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h0, w0 = img_bgr.shape[:2]
    cap = _normalize_grounding_caption(text_prompt)

    import sys

    if str(gs2_root) not in sys.path:
        sys.path.insert(0, str(gs2_root))

    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

    use_cuda = torch.cuda.is_available() and str(device).lower().startswith("cuda")
    dev = torch.device("cuda" if use_cuda else "cpu")

    gdino = load_model(
        model_config_path=str(gdino_config.resolve()),
        model_checkpoint_path=str(gdino_checkpoint.resolve()),
        device=str(dev),
    )
    image_source, image_t = load_image(str(image_path))
    boxes, confidences, _labels = predict(
        model=gdino,
        image=image_t,
        caption=cap,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        device=str(dev),
    )
    boxes = boxes.detach().cpu()
    confidences = confidences.detach().cpu()
    if boxes.numel() == 0 or boxes.shape[0] == 0:
        return None
    best = int(torch.argmax(confidences).item())
    boxes = boxes[best : best + 1]
    H, W, _ = image_source.shape
    boxes_px = boxes * torch.tensor([W, H, W, H], dtype=boxes.dtype)
    xyxy = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy().reshape(4).astype(np.float64)
    conf = float(confidences[best].item())
    # clip to image (predict can be slightly out of range)
    xyxy[0] = float(np.clip(xyxy[0], 0, w0 - 1))
    xyxy[1] = float(np.clip(xyxy[1], 0, h0 - 1))
    xyxy[2] = float(np.clip(xyxy[2], 0, w0 - 1))
    xyxy[3] = float(np.clip(xyxy[3], 0, h0 - 1))
    return xyxy, conf


def grounded_sam2_binary_mask(
    image_path: Path | str,
    text_prompt: str,
    *,
    gs2_root: Path,
    gdino_config: Path,
    gdino_checkpoint: Path,
    sam2_config: Path,
    sam2_checkpoint: Path,
    box_threshold: float,
    text_threshold: float,
    device: str = "cuda",
    allow_cpu_fallback: bool = False,
    debug_dir: Path | None = None,
    debug_stem: str = "view",
    log_progress: bool = False,
) -> np.ndarray:
    """Return ``(H, W)`` bool mask. Uses top-1 DINO box by score to limit spill onto other objects."""
    import cv2
    import torch
    from torchvision.ops import box_convert

    image_path = Path(image_path).resolve()
    gs2_root = Path(gs2_root).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h0, w0 = img_bgr.shape[:2]

    cap = _normalize_grounding_caption(text_prompt)

    import sys

    if str(gs2_root) not in sys.path:
        sys.path.insert(0, str(gs2_root))

    _ensure_torch_shared_libs_visible()

    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    def _log(msg: str) -> None:
        if log_progress:
            print(f"[grounded_sam2] {msg}", flush=True)

    _log(f"start  image={image_path.name}  prompt={cap!r}  device={device}")

    def _run(*, dev: torch.device, allow_cuda_autocast: bool) -> tuple[np.ndarray, float | None]:
        _log(
            "loading Grounding DINO (weights 디스크에서 읽는 동안 멈춘 것처럼 보일 수 있음) …"
            + f" dev={dev}"
        )
        gdino = load_model(
            model_config_path=str(gdino_config.resolve()),
            model_checkpoint_path=str(gdino_checkpoint.resolve()),
            device=str(dev),
        )
        _log("Grounding DINO ready; loading SAM2 …")
        sam2_model = build_sam2(
            sam2_hydra_config_name(sam2_config),
            str(sam2_checkpoint.resolve()),
            device=str(dev),
        )
        predictor = SAM2ImagePredictor(sam2_model)
        _log("SAM2 ready; DINO detect …")

        image_source, image_t = load_image(str(image_path))
        predictor.set_image(image_source)

        boxes, confidences, _labels = predict(
            model=gdino,
            image=image_t,
            caption=cap,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            device=str(dev),
        )
        boxes = boxes.detach().cpu()
        confidences = confidences.detach().cpu()
        _log(f"DINO done  n_boxes={0 if boxes.numel() == 0 else boxes.shape[0]}")

        dino_best_conf = None
        if boxes.numel() == 0 or boxes.shape[0] == 0:
            mask = np.zeros((h0, w0), dtype=bool)
            _log("DINO found no boxes → empty mask")
            return mask, dino_best_conf

        best = int(torch.argmax(confidences).item())
        dino_best_conf = float(confidences[best].item())
        boxes = boxes[best : best + 1]
        H, W, _ = image_source.shape
        boxes_px = boxes * torch.tensor([W, H, W, H], dtype=boxes.dtype)
        xyxy = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy().reshape(-1)

        if allow_cuda_autocast and dev.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        _log("SAM2 segment in box …")
        if allow_cuda_autocast and dev.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks_np, _scores, _logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=xyxy,
                    multimask_output=False,
                )
        else:
            masks_np, _scores, _logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=xyxy,
                multimask_output=False,
            )
        m = masks_np[0]
        if m.dtype != np.bool_:
            m = m > 0.5
        if m.shape[0] != h0 or m.shape[1] != w0:
            m = cv2.resize(m.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
        _log("SAM2 mask done")
        return m.astype(bool), dino_best_conf

    requested_cuda = str(device).lower().startswith("cuda")
    use_cuda = torch.cuda.is_available() and requested_cuda
    if requested_cuda and not use_cuda:
        raise RuntimeError(
            "Grounded-SAM2 segmentation was requested on CUDA (device='cuda'), but CUDA is not available. "
            "Install a CUDA-enabled PyTorch / driver stack, or rerun with an explicit CPU device "
            "(and optionally set allow_cpu_fallback=True only if you want automatic retry)."
        )

    dev0 = torch.device("cuda" if use_cuda else "cpu")
    try:
        mask, dino_best_conf = _run(dev=dev0, allow_cuda_autocast=True)
    except Exception as exc:
        if dev0.type == "cuda":
            # Default policy: GPU-only. No silent fallback.
            if bool(allow_cpu_fallback):
                _log(f"CUDA path failed ({exc!r}) → retry on CPU (allow_cpu_fallback=True)")
                mask, dino_best_conf = _run(dev=torch.device("cpu"), allow_cuda_autocast=False)
            else:
                raise RuntimeError(
                    "Grounded-SAM2 failed on CUDA. This project defaults to GPU-only segmentation and will not "
                    "silently retry on CPU. If you want a slower CPU retry, rerun with allow_cpu_fallback=True. "
                    f"Original error: {exc!r}"
                ) from exc
        raise

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        mask_u8 = (mask.astype(np.uint8) * 255).reshape(h0, w0)
        cv2.imwrite(str(debug_dir / f"{debug_stem}_mask_binary.png"), mask_u8)
        overlay = apply_debug_green_mask_overlay(img_bgr, mask_u8)
        cv2.imwrite(str(debug_dir / f"{debug_stem}_mask_overlay_rgb.png"), overlay)
        if dino_best_conf is not None:
            (debug_dir / f"{debug_stem}_dino_best_conf.txt").write_text(
                f"{dino_best_conf:.6f}\n", encoding="utf-8"
            )

    return mask.astype(bool)
