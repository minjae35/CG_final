from __future__ import annotations

import numpy as np

from physics.mode_b_selection import (
    filter_expanded_to_shell_around_surface,
    filter_indices_by_opacity_logit,
    largest_component_touching_seeds,
    selection_xyz_bounds,
)


def test_shell_filter_removes_far_bbox_points() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],  # surface seed
            [0.1, 0.0, 0.0],  # surface seed
            [0.05, 0.0, 0.05],  # inside bbox, near surface
            [5.0, 5.0, 5.0],  # same loose bbox but far in space
        ],
        dtype=np.float64,
    )
    surface = np.array([0, 1], dtype=np.int64)
    expanded = np.array([0, 1, 2, 3], dtype=np.int64)
    filt = filter_expanded_to_shell_around_surface(pos, surface, expanded, max_dist_m=0.5)
    assert 3 not in filt.tolist()
    assert set(filt.tolist()) <= {0, 1, 2}


def test_opacity_filter() -> None:
    logits = np.array([-2.0, 0.0, 0.5], dtype=np.float64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    out = filter_indices_by_opacity_logit(logits, idx, min_logit=-0.1)
    assert set(out.tolist()) == {1, 2}


def test_selection_xyz_bounds_empty() -> None:
    pos = np.zeros((2, 3), dtype=np.float64)
    lo, hi = selection_xyz_bounds(pos, np.array([], dtype=np.int64))
    assert lo.shape == (3,) and hi.shape == (3,)


def test_largest_component_prefers_seed_touching_blob() -> None:
    # Two blobs: left has seeds, right is noise in candidate set
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.05, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cand = np.array([0, 1, 2, 3], dtype=np.int64)
    seeds = np.array([0], dtype=np.int64)
    out = largest_component_touching_seeds(pos, cand, seeds, link_radius_m=0.1)
    assert set(out.tolist()) == {0, 1}
