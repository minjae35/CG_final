"""Project 2D mask + depth to 3DGS Gaussian indices (PRD 2.2)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def world_view_transform_apply(W: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """
    Apply 3DGS ``world_view_transform`` (same tensor as ``GaussianRasterizer``).

    Row-homogeneous convention matching CUDA ``transformPoint4x3``:
    ``Xc_h = Xw_h @ W`` with ``Xw_h = [xw, yw, zw, 1]`` (1×4).
    """
    W = np.asarray(W, dtype=np.float64)
    Xw = np.atleast_2d(np.asarray(Xw, dtype=np.float64))
    hom = np.concatenate([Xw, np.ones((Xw.shape[0], 1), dtype=np.float64)], axis=1)
    Xc_h = hom @ W
    return Xc_h[:, :3]


def cam_pinhole_xyz_to_world_row(W: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Camera-frame points (N,3) from (u,v)+depth unprojection → world (N,3)."""
    W = np.asarray(W, dtype=np.float64)
    Xc = np.atleast_2d(np.asarray(Xc, dtype=np.float64))
    hom = np.concatenate([Xc, np.ones((Xc.shape[0], 1), dtype=np.float64)], axis=1)
    Winv = np.linalg.inv(W)
    Xw_h = hom @ Winv
    return Xw_h[:, :3]


def project_world_to_pixels(
    K: np.ndarray,
    W: np.ndarray,
    Xw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World (N,3) → pixel float u, v and camera z (for valid z>0)."""
    Xc = world_view_transform_apply(W, Xw)
    z = Xc[:, 2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.empty_like(z)
    v = np.empty_like(z)
    valid = z > 1e-6
    u[:] = np.nan
    v[:] = np.nan
    u[valid] = fx * Xc[valid, 0] / z[valid] + cx
    v[valid] = fy * Xc[valid, 1] / z[valid] + cy
    return u, v, z


def indices_project_inside_mask(
    mask: np.ndarray,
    gaussian_positions: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_view_transform: np.ndarray,
    *,
    subsample: int = 1,
) -> np.ndarray:
    """
    Gaussians whose centre projects inside ``mask`` (no depth / kNN filter).

    ``subsample``: only test every k-th Gaussian index (1 = all) for speed on huge clouds.
    """
    pos = np.asarray(gaussian_positions, dtype=np.float64)
    W = np.asarray(world_view_transform, dtype=np.float64)
    if pos.size == 0:
        return np.array([], dtype=np.int64)
    ss = max(1, int(subsample))
    idx_all = np.arange(0, pos.shape[0], ss, dtype=np.int64)
    sub = pos[idx_all]
    u, v, z = project_world_to_pixels(camera_intrinsics, W, sub)
    H, Wm = mask.shape
    valid = z > 1e-6
    ui = np.floor(u + 0.5).astype(np.int32)
    vi = np.floor(v + 0.5).astype(np.int32)
    inb = valid & (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < H)
    inside = np.zeros_like(inb, dtype=bool)
    inside[inb] = mask[vi[inb], ui[inb]]
    return np.sort(idx_all[inside])


def mask_to_gaussian_indices(
    mask: np.ndarray,
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_view_transform: np.ndarray,
    gaussian_positions: np.ndarray,
    distance_threshold: float = 0.05,
    stride: int = 4,
    *,
    use_depth_consistency: bool = False,
    depth_tolerance_abs_m: float = 0.04,
    depth_tolerance_rel: float = 0.03,
    knn: int = 12,
    pixel_tolerance_px: float = 5.0,
    min_votes: int = 2,
    stats: dict[str, int | float] | None = None,
) -> np.ndarray:
    """
    mask: (H, W) bool
    depth_map: (H, W) float32, camera-space z (meters) consistent with 3DGS render depth
    camera_intrinsics: (3, 3)
    world_view_transform: (4, 4) from ``cam_meta_*.npz`` (same as ``view.world_view_transform``)
    gaussian_positions: (N, 3)
    """
    if gaussian_positions.size == 0:
        if stats is not None:
            stats["mask_true_pixels"] = int(mask.sum())
        return np.array([], dtype=np.int64)

    W = np.asarray(world_view_transform, dtype=np.float64)
    pos = np.asarray(gaussian_positions, dtype=np.float64)
    H, Wm = mask.shape
    fx, fy = float(camera_intrinsics[0, 0]), float(camera_intrinsics[1, 1])
    cx, cy = float(camera_intrinsics[0, 2]), float(camera_intrinsics[1, 2])

    if stats is not None:
        stats["mask_true_pixels"] = int(mask.sum())
        stats["mask_height"] = int(H)
        stats["mask_width"] = int(Wm)
        stats["depth_map_shape_h"] = int(depth_map.shape[0])
        stats["depth_map_shape_w"] = int(depth_map.shape[1])

    if not use_depth_consistency:
        return _mask_to_gaussian_indices_nearest(
            mask,
            depth_map,
            fx,
            fy,
            cx,
            cy,
            W,
            pos,
            distance_threshold,
            stride,
            stats,
        )

    tree = cKDTree(pos)
    votes: defaultdict[int, int] = defaultdict(int)
    ys, xs = np.where(mask)
    kq = int(min(max(1, knn), pos.shape[0]))

    n_samples = 0
    n_valid_z = 0
    n_knn_in_radius = 0
    n_after_pixel_tol = 0
    n_after_depth_tol = 0

    for v, u in zip(ys[::stride], xs[::stride]):
        n_samples += 1
        z_ref = float(depth_map[v, u])
        if z_ref <= 0.0:
            continue
        n_valid_z += 1
        x = (u - cx) / fx * z_ref
        y = (v - cy) / fy * z_ref
        Xc_ref = np.array([[x, y, z_ref]], dtype=np.float64)
        Xw_ref = cam_pinhole_xyz_to_world_row(W, Xc_ref)[0]

        d_nn, idx_nn = tree.query(Xw_ref, k=kq, distance_upper_bound=float(distance_threshold))
        if kq == 1:
            idx_nn = np.array([idx_nn], dtype=np.int64)
            d_nn = np.array([d_nn], dtype=np.float64)
        if not np.isfinite(d_nn).any():
            continue
        tol_z = max(float(depth_tolerance_abs_m), float(depth_tolerance_rel) * z_ref)
        for d_i, gi in zip(np.atleast_1d(d_nn), np.atleast_1d(idx_nn)):
            if not np.isfinite(d_i) or d_i > distance_threshold:
                continue
            gi = int(gi)
            if gi < 0 or gi >= pos.shape[0]:
                continue
            n_knn_in_radius += 1
            Xw_g = pos[gi]
            Xc_g = world_view_transform_apply(W, Xw_g.reshape(1, 3))[0]
            zg = float(Xc_g[2])
            if zg <= 1e-6:
                continue
            ug = fx * float(Xc_g[0]) / zg + cx
            vg = fy * float(Xc_g[1]) / zg + cy
            if abs(ug - u) > pixel_tolerance_px or abs(vg - v) > pixel_tolerance_px:
                continue
            n_after_pixel_tol += 1
            if abs(zg - z_ref) > tol_z:
                continue
            n_after_depth_tol += 1
            votes[gi] += 1

    out = [g for g, c in votes.items() if c >= int(min_votes)]
    if stats is not None:
        stats["depth_stride_samples"] = int(n_samples)
        stats["depth_samples_valid_z"] = int(n_valid_z)
        stats["depth_hits_knn_in_radius"] = int(n_knn_in_radius)
        stats["depth_hits_after_pixel_tolerance"] = int(n_after_pixel_tol)
        stats["depth_hits_after_depth_tolerance"] = int(n_after_depth_tol)
        stats["depth_unique_before_vote"] = int(len(votes))
        stats["depth_unique_after_vote_min_votes"] = int(len(out))

    return np.array(sorted(out), dtype=np.int64)


def _mask_to_gaussian_indices_nearest(
    mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    W: np.ndarray,
    pos: np.ndarray,
    distance_threshold: float,
    stride: int,
    stats: dict[str, int | float] | None,
) -> np.ndarray:
    pts = []
    ys, xs = np.where(mask)
    n_pix = 0
    n_valid_z = 0
    for v, u in zip(ys[::stride], xs[::stride]):
        n_pix += 1
        z = float(depth_map[v, u])
        if z <= 0:
            continue
        n_valid_z += 1
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        Xc = np.array([[x, y, z]], dtype=np.float64)
        Xw = cam_pinhole_xyz_to_world_row(W, Xc)[0]
        pts.append(Xw)
    if stats is not None:
        stats["nearest_stride_samples"] = int(n_pix)
        stats["nearest_samples_valid_z"] = int(n_valid_z)
    if not pts:
        if stats is not None:
            stats["nearest_after_knn_unique"] = 0
        return np.array([], dtype=np.int64)

    cloud = np.stack(pts, axis=0)
    tree = cKDTree(pos)
    d, idx = tree.query(cloud, k=1, distance_upper_bound=distance_threshold)
    valid = np.isfinite(d) & (d <= distance_threshold)
    idx = idx[valid]
    out = np.unique(idx.astype(np.int64))
    if stats is not None:
        stats["nearest_after_knn_unique"] = int(out.size)
    return out


def save_gaussian_projection_debug_image(
    rgb_path: str | Path,
    out_path: str | Path,
    gaussian_positions: np.ndarray,
    indices: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_view_transform: np.ndarray,
    *,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    radius: int = 2,
) -> None:
    """Draw projected centres of ``indices`` onto a copy of ``rgb_path`` (BGR file on disk)."""
    import cv2

    rgb_path = Path(rgb_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = cv2.imread(str(rgb_path))
    if base is None:
        raise FileNotFoundError(rgb_path)
    H, W = base.shape[:2]
    K = np.asarray(camera_intrinsics, dtype=np.float64)
    Wm = np.asarray(world_view_transform, dtype=np.float64)
    pos = np.asarray(gaussian_positions, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < pos.shape[0])]
    sub = pos[idx]
    u, v, z = project_world_to_pixels(K, Wm, sub)
    valid = z > 1e-4
    ui = np.floor(u[valid] + 0.5).astype(np.int32)
    vi = np.floor(v[valid] + 0.5).astype(np.int32)
    for uu, vv in zip(ui.tolist(), vi.tolist()):
        if 0 <= uu < W and 0 <= vv < H:
            cv2.circle(base, (uu, vv), int(radius), color_bgr, thickness=-1)
    cv2.imwrite(str(out_path), base)


def save_subsampled_all_gaussian_projections(
    rgb_path: str | Path,
    out_path: str | Path,
    gaussian_positions: np.ndarray,
    camera_intrinsics: np.ndarray,
    world_view_transform: np.ndarray,
    *,
    max_points: int = 100_000,
    color_bgr: tuple[int, int, int] = (140, 140, 140),
    radius: int = 1,
) -> int:
    """Draw projected centres for a stride over *all* Gaussians; returns number drawn."""
    import cv2

    rgb_path = Path(rgb_path)
    out_path = Path(out_path)
    pos = np.asarray(gaussian_positions, dtype=np.float64)
    n = pos.shape[0]
    stride = max(1, int(np.ceil(n / max(1, max_points))))
    sub = pos[::stride]
    base = cv2.imread(str(rgb_path))
    if base is None:
        raise FileNotFoundError(rgb_path)
    H, W = base.shape[:2]
    K = np.asarray(camera_intrinsics, dtype=np.float64)
    Wm = np.asarray(world_view_transform, dtype=np.float64)
    u, v, z = project_world_to_pixels(K, Wm, sub)
    valid = z > 1e-4
    ui = np.floor(u[valid] + 0.5).astype(np.int32)
    vi = np.floor(v[valid] + 0.5).astype(np.int32)
    drawn = 0
    for uu, vv in zip(ui.tolist(), vi.tolist()):
        if 0 <= uu < W and 0 <= vv < H:
            cv2.circle(base, (uu, vv), int(radius), color_bgr, thickness=-1)
            drawn += 1
    cv2.imwrite(str(out_path), base)
    return drawn
