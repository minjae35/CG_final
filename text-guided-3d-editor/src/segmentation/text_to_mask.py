"""Grounding DINO + SAM2 -> binary mask (PRD 2.1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import PROJECT_ROOT


def text_to_mask(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    submodule_root: Path | None = None,
    *,
    gdino_config: str | Path | None = None,
    gdino_checkpoint: str | Path | None = None,
    sam2_config: str | Path | None = None,
    sam2_checkpoint: str | Path | None = None,
    device: str = "cuda",
    allow_cpu_fallback: bool = False,
    debug_mask_dir: Path | str | None = None,
    debug_stem: str = "view",
    log_progress: bool = False,
    debug_info: dict | None = None,
) -> np.ndarray:
    """
    Returns bool mask (H, W). Uses Grounded-SAM-2 when models import; otherwise
    a large centre crop for offline CI only.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    gs2 = submodule_root or (PROJECT_ROOT / "submodules" / "Grounded-SAM-2")
    gs2 = Path(gs2).resolve()

    def _resolve(p: str | Path | None, default_rel: str) -> Path:
        if p is None:
            return (gs2 / default_rel).resolve()
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()

    gd_cfg = _resolve(gdino_config, "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    gd_ckpt = _resolve(gdino_checkpoint, "gdino_checkpoints/groundingdino_swinb_cogcoor.pth")
    s2_cfg = _resolve(sam2_config, "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    s2_ckpt = _resolve(sam2_checkpoint, "checkpoints/sam2.1_hiera_large.pt")

    dbg = Path(debug_mask_dir).resolve() if debug_mask_dir else None

    missing: list[Path] = [p for p in (gd_cfg, gd_ckpt, s2_cfg, s2_ckpt) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Grounded-SAM-2 assets: "
            + ", ".join(str(p) for p in missing)
            + " (check submodules + checkpoint paths in configs/pipeline_config.yaml)"
        )

    if debug_info is not None:
        debug_info.clear()
        debug_info["prompt"] = str(text_prompt)
        debug_info["image_path"] = str(image_path)
        debug_info["is_fallback_rect"] = False
        debug_info["dino_confidence"] = None
        debug_info["mask_bbox_xyxy"] = None
        debug_info["mask_true_pixels"] = None
        debug_info["mask_area_ratio"] = None
        debug_info["mask_hw"] = None

    try:
        from segmentation.grounded_sam2_mask import grounded_sam2_binary_mask

        mask = grounded_sam2_binary_mask(
            image_path,
            text_prompt,
            gs2_root=gs2,
            gdino_config=gd_cfg,
            gdino_checkpoint=gd_ckpt,
            sam2_config=s2_cfg,
            sam2_checkpoint=s2_ckpt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            allow_cpu_fallback=bool(allow_cpu_fallback),
            debug_dir=dbg,
            debug_stem=debug_stem,
            log_progress=log_progress,
        )
        if debug_info is not None:
            h, w = mask.shape
            debug_info["mask_hw"] = [int(h), int(w)]
            tp = int(mask.sum())
            debug_info["mask_true_pixels"] = tp
            debug_info["mask_area_ratio"] = float(tp) / float(h * w) if h * w > 0 else None
            ys, xs = np.where(mask)
            if ys.size:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                debug_info["mask_bbox_xyxy"] = [x0, y0, x1, y1]
        return mask
    except Exception as exc:
        # Policy: never return a fake/center-box mask. Fail loudly by default.
        # If allow_cpu_fallback=True, the underlying Grounded-SAM2 wrapper may retry on CPU,
        # but if we still land here we must raise a clear error.
        if debug_info is not None:
            debug_info["exception_repr"] = repr(exc)
        raise RuntimeError(
            "Grounded-SAM2 segmentation failed. This project does not generate a fake fallback mask. "
            "Fix the Grounded-SAM-2/CUDA extension build (e.g. GroundingDINO _C) and checkpoint paths. "
            "If you want a slow CPU retry, pass --allow-cpu-fallback (it will only retry; it will not "
            "replace the result with a synthetic mask). "
            f"Original exception: {exc!r}"
        ) from exc
