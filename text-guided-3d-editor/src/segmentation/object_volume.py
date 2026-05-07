"""Expand sparse segmentation hits into a coherent object volume."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def expand_indices_to_bbox_volume(
    positions: np.ndarray,
    seed_indices: np.ndarray,
    margin_ratio: float = 0.12,
    z_margin_ratio: float = 0.20,
    max_indices: int | None = 35000,
) -> np.ndarray:
    """Return all Gaussian indices inside an expanded seed bounding box.

    Multi-view mask projection tends to select visible surface Gaussians.  For
    PhysGaussian this behaves like a loose shell.  Expanding the selected hits to
    the object's 3D extent keeps hidden tabletop/leg Gaussians in the same
    dynamic body, which makes jelly-like motion coherent.
    """
    if len(seed_indices) == 0:
        return np.asarray(seed_indices, dtype=np.int64)

    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    pts = positions[seed_indices]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    ext = np.maximum(hi - lo, 1e-6)
    margin = ext * float(margin_ratio)
    margin[2] = ext[2] * float(z_margin_ratio)
    lo -= margin
    hi += margin

    inside = np.all((positions >= lo) & (positions <= hi), axis=1)
    expanded = np.nonzero(inside)[0].astype(np.int64)
    if max_indices is not None and len(expanded) > max_indices:
        tree = cKDTree(pts)
        d, _ = tree.query(positions[expanded], k=1)
        expanded = expanded[np.argsort(d)[:max_indices]]
    return np.sort(expanded.astype(np.int64))
