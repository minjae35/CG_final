from __future__ import annotations

import numpy as np

from segmentation.mask_to_gaussians import cam_pinhole_xyz_to_world_row, project_world_to_pixels


def complete_selection_by_mask_projection(
    *,
    positions: np.ndarray,
    selected_indices: np.ndarray,
    mask: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    world_view_transform: np.ndarray,
    # 3D gate:
    # Prefer a world-space AABB derived from the mask+depth itself.
    # Fall back to the current selected AABB if mask depth is missing.
    selected_aabb_lo: np.ndarray,
    selected_aabb_hi: np.ndarray,
    aabb_expand_ratio: float = 0.08,
    mask_unproject_stride: int = 8,
    mask_dilate_px: int = 0,
    neighbor_radius_m: float | None = None,
    # Depth gate: only accept points whose camera-z matches rendered depth at that pixel.
    depth_tolerance_abs_m: float = 0.035,
    depth_tolerance_rel: float = 0.03,
    # Safety: cap additions.
    max_add: int | None = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand `selected_indices` by adding gaussians whose projected centers fall inside `mask`,
    and whose projected z matches `depth_map` at that pixel (within tolerance).

    Returns:
      (expanded_selected_indices, added_indices)

    Notes:
    - Vectorized; designed for ~1e6 gaussians.
    - Uses a 3D AABB gate first to avoid pulling in distant wall/curtain gaussians.
    - Uses only center projection (not full splat footprint); that's a deliberate first pass
      that remains general and fast.
    """
    pos = np.asarray(positions, dtype=np.float64)
    sel = np.unique(np.asarray(selected_indices, dtype=np.int64).ravel())
    sel = sel[(sel >= 0) & (sel < pos.shape[0])]
    if sel.size == 0:
        return sel.astype(np.int64), sel.astype(np.int64)

    # Build a gating AABB from unprojected mask pixels (more robust when current selection
    # is missing visible parts like the backrest).
    lo_m = None
    hi_m = None
    try:
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if ys.size:
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            pts = []
            step = int(max(1, mask_unproject_stride))
            for v, u in zip(ys[::step], xs[::step]):
                z = float(depth_map[int(v), int(u)])
                if z <= 0.0:
                    continue
                x = (float(u) - cx) / fx * z
                y = (float(v) - cy) / fy * z
                Xc = np.array([[x, y, z]], dtype=np.float64)
                Xw = cam_pinhole_xyz_to_world_row(np.asarray(world_view_transform, dtype=np.float64), Xc)[0]
                pts.append(Xw)
            if pts:
                P = np.stack(pts, axis=0)
                lo_m = P.min(axis=0)
                hi_m = P.max(axis=0)
    except Exception:
        lo_m = None
        hi_m = None

    if lo_m is None or hi_m is None:
        lo = np.asarray(selected_aabb_lo, dtype=np.float64).reshape(3)
        hi = np.asarray(selected_aabb_hi, dtype=np.float64).reshape(3)
    else:
        lo = np.asarray(lo_m, dtype=np.float64).reshape(3)
        hi = np.asarray(hi_m, dtype=np.float64).reshape(3)

    ext = (hi - lo) * float(aabb_expand_ratio)
    lo2 = lo - ext
    hi2 = hi + ext

    # 3D AABB gate first.
    in_aabb = np.all((pos >= lo2[None, :]) & (pos <= hi2[None, :]), axis=1)
    idx_cand = np.nonzero(in_aabb)[0].astype(np.int64)
    if idx_cand.size == 0:
        return sel.astype(np.int64), np.array([], dtype=np.int64)

    # 3D neighborhood gate (conservative): keep only points near the original selection.
    try:
        r3 = float(neighbor_radius_m) if neighbor_radius_m is not None else None
    except Exception:
        r3 = None
    if r3 is not None and r3 > 0:
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(pos[sel])
            d, _ = tree.query(pos[idx_cand], k=1, workers=-1)
            idx_cand = idx_cand[np.asarray(d <= r3)]
        except Exception:
            # If KDTree fails, continue without this gate (but caller should keep params conservative).
            pass
        if idx_cand.size == 0:
            return sel.astype(np.int64), np.array([], dtype=np.int64)

    # Optionally dilate mask to account for projected footprint overlap.
    mask_gate = np.asarray(mask, dtype=bool)
    try:
        r = int(mask_dilate_px)
    except Exception:
        r = 0
    if r > 0:
        try:
            from scipy.ndimage import binary_dilation

            yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
            st = (xx * xx + yy * yy) <= (r * r)
            mask_gate = binary_dilation(mask_gate, structure=st, iterations=1)
        except Exception:
            # Best-effort; if scipy not present, proceed without dilation.
            mask_gate = np.asarray(mask, dtype=bool)

    # IMPORTANT: depth must be anchored to the *original* object mask, not the dilated region.
    # Otherwise dilation can pick wall pixels whose depth matches the wall, incorrectly adding background.
    depth_obj = np.asarray(depth_map, dtype=np.float64).copy()
    depth_obj[~np.asarray(mask, dtype=bool)] = np.inf
    depth_near = depth_obj
    if r > 0:
        try:
            from scipy.ndimage import minimum_filter

            # local min of object-mask depth within dilation radius
            depth_near = minimum_filter(depth_obj, size=(2 * r + 1, 2 * r + 1), mode="nearest")
        except Exception:
            depth_near = depth_obj

    # Project candidate centers and test mask overlap.
    Xw = pos[idx_cand]
    u, v, z = project_world_to_pixels(np.asarray(K, dtype=np.float64), np.asarray(world_view_transform, dtype=np.float64), Xw)
    H, Wm = mask.shape
    valid = z > 1e-6
    ui = np.floor(u + 0.5).astype(np.int32)
    vi = np.floor(v + 0.5).astype(np.int32)
    inb = valid & (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < H)
    if not np.any(inb):
        return sel.astype(np.int64), np.array([], dtype=np.int64)

    inside = np.zeros_like(inb, dtype=bool)
    inside[inb] = mask_gate[vi[inb], ui[inb]]
    if not np.any(inside):
        return sel.astype(np.int64), np.array([], dtype=np.int64)

    idx_inside = idx_cand[inside]
    ui2 = ui[inside]
    vi2 = vi[inside]
    z2 = z[inside]

    # Depth-consistency gate.
    zr = depth_near[vi2, ui2].astype(np.float64)
    okz = np.isfinite(zr) & (zr > 0.0)
    if not np.any(okz):
        return sel.astype(np.int64), np.array([], dtype=np.int64)

    zr = zr[okz]
    z2 = z2[okz]
    idx_inside = idx_inside[okz]
    tol = np.maximum(float(depth_tolerance_abs_m), float(depth_tolerance_rel) * zr)
    # Conservative: reject points BEHIND the visible object surface.
    # Allow slightly in front (z <= zr + tol).
    depth_ok = z2 <= (zr + tol)
    idx_depth = idx_inside[depth_ok]
    if idx_depth.size == 0:
        return sel.astype(np.int64), np.array([], dtype=np.int64)

    # Add only new indices.
    sel_mask = np.zeros((pos.shape[0],), dtype=bool)
    sel_mask[sel] = True
    added = idx_depth[~sel_mask[idx_depth]]
    if added.size == 0:
        return sel.astype(np.int64), np.array([], dtype=np.int64)

    if max_add is not None and int(max_add) > 0 and added.size > int(max_add):
        # Prefer closer-to-camera points (smaller z) under the same mask.
        # (We still keep it deterministic.)
        # Re-project just the added ones to get z.
        Xwa = pos[added]
        _, _, za = project_world_to_pixels(np.asarray(K, dtype=np.float64), np.asarray(world_view_transform, dtype=np.float64), Xwa)
        order = np.argsort(za)
        added = added[order[: int(max_add)]]

    expanded = np.sort(np.unique(np.concatenate([sel, added]).astype(np.int64)))
    return expanded, np.sort(np.unique(added.astype(np.int64)))

