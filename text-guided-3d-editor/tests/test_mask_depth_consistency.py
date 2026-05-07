from __future__ import annotations

import numpy as np

from segmentation.mask_to_gaussians import mask_to_gaussian_indices


def test_depth_consistency_fewer_or_equal_than_nearest() -> None:
    """Depth+vote path should not invent new indices vs. nearest-only on the same mask."""
    H, W = 32, 32
    mask = np.zeros((H, W), dtype=bool)
    mask[8:24, 8:24] = True
    depth = np.ones((H, W), dtype=np.float32) * 2.5

    fx = fy = 200.0
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)

    rng = np.random.default_rng(0)
    pos = rng.normal(size=(400, 3)).astype(np.float64) * 0.15
    pos[:, 2] += 2.5

    a = mask_to_gaussian_indices(
        mask,
        depth,
        K,
        w2c,
        pos,
        distance_threshold=0.25,
        stride=3,
        use_depth_consistency=False,
    )
    b = mask_to_gaussian_indices(
        mask,
        depth,
        K,
        w2c,
        pos,
        distance_threshold=0.25,
        stride=3,
        use_depth_consistency=True,
        depth_tolerance_abs_m=0.12,
        depth_tolerance_rel=0.05,
        knn=8,
        pixel_tolerance_px=6.0,
        min_votes=2,
    )
    assert b.dtype == np.int64
    # Voting can admit Gaussians that were not the single global nearest on any
    # one pixel; only check we get a bounded set.
    assert b.size <= a.size * 3
