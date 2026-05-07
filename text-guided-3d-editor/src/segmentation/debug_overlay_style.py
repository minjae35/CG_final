"""Shared debug visualization colors (BGR) and mask overlay math.

``grounding_dino_bbox_*.png`` uses ``DEBUG_GREEN_BGR`` for the rectangle stroke;
``02_mask_overlay.png`` uses the same BGR as the blend target so hues match.
"""
from __future__ import annotations

import numpy as np

# OpenCV BGR — same saturated green as ``scripts/draw_grounding_dino_bbox_slide.py`` stroke
DEBUG_GREEN_BGR: tuple[int, int, int] = (0, 255, 0)

# Outside mask: original 100%. Inside: (1-beta)*orig + beta*DEBUG_GREEN_BGR
# Higher beta → closer to saturated ``DEBUG_GREEN_BGR`` (bbox stroke is 100% solid).
DEBUG_MASK_OVERLAY_BETA: float = 0.55

# SAM2 edges are often soft (0<gray<255); soft weights dilute hue vs a crisp bbox line.
DEBUG_MASK_OVERLAY_HARD_THRESHOLD: int = 128


def apply_debug_green_mask_overlay(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    beta: float | None = None,
    hard_threshold: int | None = None,
) -> np.ndarray:
    """Return uint8 BGR image. ``mask_u8`` is H×W, 0–255.

    Uses a **hard** binary mask at ``hard_threshold`` so anti-aliased fringe does not
    smear a different green than ``DEBUG_GREEN_BGR``.
    """
    b = float(DEBUG_MASK_OVERLAY_BETA if beta is None else beta)
    thr = int(DEBUG_MASK_OVERLAY_HARD_THRESHOLD if hard_threshold is None else hard_threshold)
    if img_bgr.shape[:2] != mask_u8.shape[:2]:
        raise ValueError(
            f"shape mismatch img {img_bgr.shape[:2]} vs mask {mask_u8.shape[:2]}"
        )
    img_f = img_bgr.astype(np.float32)
    mask_bin = (mask_u8.astype(np.int16) >= thr).astype(np.float32)[..., None]
    mask_f = mask_bin
    tint = np.zeros_like(img_f)
    tint[:, :, 0] = float(DEBUG_GREEN_BGR[0])
    tint[:, :, 1] = float(DEBUG_GREEN_BGR[1])
    tint[:, :, 2] = float(DEBUG_GREEN_BGR[2])
    out = img_f * (1.0 - b * mask_f) + tint * (b * mask_f)
    return np.clip(out, 0, 255).astype(np.uint8)
