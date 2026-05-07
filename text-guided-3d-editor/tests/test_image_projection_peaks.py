from __future__ import annotations

import numpy as np

from generation.image_generator import _count_projection_peaks


def _column_blob(height: int, width: int, x_centers: list[int], radius: int) -> np.ndarray:
    """Make a binary mask with vertical-stripe blobs centered at the given x positions."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for cx in x_centers:
        x0 = max(0, cx - radius)
        x1 = min(width, cx + radius)
        y0 = height // 4
        y1 = 3 * height // 4
        mask[y0:y1, x0:x1] = 1
    return mask


def test_single_blob_has_one_peak() -> None:
    mask = _column_blob(256, 512, [256], radius=40)
    assert _count_projection_peaks(mask, axis=0) == 1


def test_four_well_separated_blobs_have_four_peaks() -> None:
    mask = _column_blob(256, 512, [80, 200, 320, 440], radius=20)
    assert _count_projection_peaks(mask, axis=0) == 4


def test_empty_mask_returns_zero() -> None:
    assert _count_projection_peaks(np.zeros((128, 128), dtype=np.uint8), axis=0) == 0
