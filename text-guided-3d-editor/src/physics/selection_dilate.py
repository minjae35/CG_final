"""Expand 3DGS index selection in world space (mode-b physics)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def dilate_gaussian_indices(
    positions: np.ndarray,
    indices: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """
    Include every Gaussian whose world position lies within `radius_m` of any
    seed index. Fills thin mask+depth shells so the simulated blob stays
    connected to nearby desk geometry instead of sliding off as a separate sheet.
    """
    if radius_m <= 0.0 or indices.size == 0 or positions.size == 0:
        return np.asarray(indices, dtype=np.int64)

    pos = np.asarray(positions, dtype=np.float64)
    idx = np.unique(np.asarray(indices, dtype=np.int64).ravel())
    idx = idx[(idx >= 0) & (idx < pos.shape[0])]
    if idx.size == 0:
        return idx

    tree = cKDTree(pos)
    expanded: set[int] = set(int(i) for i in idx.tolist())
    r = float(radius_m)
    for i in idx.tolist():
        neigh = tree.query_ball_point(pos[i], r=r)
        expanded.update(int(j) for j in neigh)
    return np.array(sorted(expanded), dtype=np.int64)


def dilate_gaussian_indices_multi(
    positions: np.ndarray,
    indices: np.ndarray,
    radius_m: float,
    passes: int = 1,
) -> np.ndarray:
    """Repeat ball dilation `passes` times (bridges sparse gaps between front/back desk splats)."""
    out = np.asarray(indices, dtype=np.int64)
    n = max(1, int(passes))
    for _ in range(n):
        out = dilate_gaussian_indices(positions, out, radius_m)
    return out
