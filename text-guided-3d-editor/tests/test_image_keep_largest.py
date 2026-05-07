from __future__ import annotations

import numpy as np
from PIL import Image


def _two_disks() -> Image.Image:
    """Build a 256x256 RGBA image with two disks (large yellow + small red) on white."""
    h = w = 256
    rgba = np.full((h, w, 4), (255, 255, 255, 0), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    big = (yy - 110) ** 2 + (xx - 80) ** 2 < 50**2
    rgba[big] = (250, 220, 30, 255)
    small = (yy - 200) ** 2 + (xx - 200) ** 2 < 22**2
    rgba[small] = (220, 30, 30, 255)
    return Image.fromarray(rgba, mode="RGBA")


def test_keep_largest_subject_drops_smaller_blob() -> None:
    from generation.image_generator import _keep_largest_subject

    img = _two_disks()
    out, info = _keep_largest_subject(img)
    arr = np.asarray(out)
    assert arr.shape == (256, 256, 3)
    bg = np.array((245, 245, 245), dtype=np.uint8)
    assert info["n_components"] >= 1
    assert info["kept_area"] >= info["removed_area"]
    far_from_yellow_center = (np.s_[195:215, 190:215])
    sub = arr[far_from_yellow_center]
    diff = np.abs(sub.astype(np.int16) - bg.astype(np.int16)).sum(axis=-1)
    assert (diff <= 24).mean() > 0.9, "small (red) subject region should be repainted to background"
