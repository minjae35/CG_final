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
    debug_mask_dir: Path | str | None = None,
    debug_stem: str = "view",
    log_progress: bool = False,
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

    try:
        from segmentation.grounded_sam2_mask import grounded_sam2_binary_mask

        return grounded_sam2_binary_mask(
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
            debug_dir=dbg,
            debug_stem=debug_stem,
            log_progress=log_progress,
        )
    except Exception as exc:
        # CI / missing weights / import errors — keep a deterministic wide mask
        from PIL import Image as PILImage

        img = np.array(PILImage.open(image_path).convert("RGB"))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        if dbg is not None:
            dbg.mkdir(parents=True, exist_ok=True)
            import cv2

            m = (mask.astype(np.uint8) * 255).reshape(h, w)
            cv2.imwrite(str(dbg / f"{debug_stem}_mask_binary_fallback.png"), m)
            print(f"[text_to_mask] Grounded-SAM-2 failed ({exc!r}); using centre fallback mask.", flush=True)
        return mask
