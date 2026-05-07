"""Mode-b selection: 2D mask support + rendered-depth gate (precision over recall)."""
from __future__ import annotations

import numpy as np

from segmentation.mask_to_gaussians import cam_pinhole_xyz_to_world_row, world_view_transform_apply


def mask_centroid_world_xyz(
    mask: np.ndarray,
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_view_transform: np.ndarray,
    stride: int = 8,
) -> np.ndarray | None:
    """Mean world XYZ of unprojected SAM pixels (foreground mask centroid)."""
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    W = np.asarray(world_view_transform, dtype=np.float64)
    xs = []
    ys, xs_i = np.where(mask)
    for v, u in zip(ys[::stride], xs_i[::stride]):
        z = float(depth_map[v, u])
        if z <= 0.0:
            continue
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        Xc = np.array([[x, y, z]], dtype=np.float64)
        Xw = cam_pinhole_xyz_to_world_row(W, Xc)[0]
        xs.append(Xw)
    if not xs:
        return None
    return np.stack(xs, axis=0).mean(axis=0)


def filter_indices_mask_and_rendered_depth(
    positions: np.ndarray,
    indices: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_view_transform: np.ndarray,
    mask: np.ndarray,
    depth_map: np.ndarray,
    *,
    depth_tolerance_abs_m: float,
    depth_tolerance_rel: float,
) -> np.ndarray:
    """Keep Gaussians that project inside ``mask`` and match ``depth_map`` at that pixel."""
    idx = np.asarray(indices, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < positions.shape[0])]
    if idx.size == 0:
        return idx.astype(np.int64)

    H, Wm = mask.shape
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    W = np.asarray(world_view_transform, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)

    keep: list[int] = []
    for g in idx.tolist():
        Xw = pos[int(g)].reshape(1, 3)
        Xc = world_view_transform_apply(W, Xw)[0]
        z = float(Xc[2])
        if z <= 1e-6:
            continue
        u = int(round(fx * float(Xc[0]) / z + cx))
        v = int(round(fy * float(Xc[1]) / z + cy))
        if u < 0 or u >= Wm or v < 0 or v >= H:
            continue
        if not bool(mask[v, u]):
            continue
        zr = float(depth_map[v, u])
        if zr <= 0.0:
            continue
        tol = max(float(depth_tolerance_abs_m), float(depth_tolerance_rel) * zr)
        if abs(z - zr) > tol:
            continue
        keep.append(int(g))
    return np.sort(np.unique(np.asarray(keep, dtype=np.int64)))


def filter_indices_multiview_mask_depth_intersection(
    positions: np.ndarray,
    indices: np.ndarray,
    views: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    depth_tolerance_abs_m: float,
    depth_tolerance_rel: float,
) -> np.ndarray:
    """Intersect per-view mask+depth filters (precision). ``views`` = (K, w2c, mask, depth)."""
    out = np.asarray(indices, dtype=np.int64).ravel()
    for K, w2c, mask, depth in views:
        out = filter_indices_mask_and_rendered_depth(
            positions,
            out,
            K,
            w2c,
            mask,
            depth,
            depth_tolerance_abs_m=depth_tolerance_abs_m,
            depth_tolerance_rel=depth_tolerance_rel,
        )
        if out.size == 0:
            return out.astype(np.int64)
    return out.astype(np.int64)
