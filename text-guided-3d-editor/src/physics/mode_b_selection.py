"""Mode-b (scene edit) Gaussian index filtering beyond raw mask consensus + bbox fill."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def read_ply_opacity_logits(ply_path: Path | str) -> np.ndarray:
    """Return per-vertex opacity in logit space as stored in 3DGS PLY (shape (N,) or (N,1))."""
    v = PlyData.read(str(ply_path))["vertex"]
    op = np.asarray(v["opacity"], dtype=np.float64).reshape(-1)
    return op


def filter_indices_by_opacity_logit(
    opacity_logits: np.ndarray,
    indices: np.ndarray,
    min_logit: float,
) -> np.ndarray:
    """Keep indices whose PLY opacity logit is >= ``min_logit`` (higher = more opaque after sigmoid)."""
    if indices.size == 0:
        return np.asarray(indices, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < opacity_logits.shape[0])]
    if idx.size == 0:
        return idx.astype(np.int64)
    keep = opacity_logits[idx] >= float(min_logit)
    return np.sort(idx[keep].astype(np.int64))


def filter_expanded_to_shell_around_surface(
    positions: np.ndarray,
    surface_indices: np.ndarray,
    expanded_indices: np.ndarray,
    max_dist_m: float,
) -> np.ndarray:
    """Drop expanded Gaussians farther than ``max_dist_m`` from any *surface* seed point.

    ``expand_indices_to_bbox_volume`` fills a loose axis-aligned box, which often
    includes walls, floor slabs, and neighbouring objects.  Keeping only points
    near the multi-view surface hits preserves a tight object shell.
    """
    if max_dist_m <= 0.0 or expanded_indices.size == 0:
        return np.asarray(expanded_indices, dtype=np.int64)
    surf = np.asarray(surface_indices, dtype=np.int64).ravel()
    surf = surf[(surf >= 0) & (surf < positions.shape[0])]
    exp = np.asarray(expanded_indices, dtype=np.int64).ravel()
    exp = exp[(exp >= 0) & (exp < positions.shape[0])]
    if surf.size == 0:
        return np.sort(exp.astype(np.int64))
    tree = cKDTree(positions[surf])
    d, _ = tree.query(positions[exp], k=1)
    keep = d <= float(max_dist_m)
    return np.sort(exp[keep].astype(np.int64))


def largest_component_touching_seeds(
    positions: np.ndarray,
    candidate_indices: np.ndarray,
    seed_indices: np.ndarray,
    link_radius_m: float,
) -> np.ndarray:
    """Keep one 3D connected component (``query_pairs`` graph) that touches surface seeds.

    Seeds are restricted to ``np.intersect1d(seed_indices, candidate_indices)``.
    If none intersect, the **largest** component in the candidate graph is kept
    (fallback when shell already dropped all pure seeds).
    """
    cand = np.unique(np.asarray(candidate_indices, dtype=np.int64).ravel())
    cand = cand[(cand >= 0) & (cand < positions.shape[0])]
    if cand.size == 0:
        return cand.astype(np.int64)
    if cand.size == 1:
        return cand.astype(np.int64)

    P = positions[cand]
    tree = cKDTree(P)
    pairs = tree.query_pairs(r=float(link_radius_m))
    nloc = int(cand.shape[0])
    parent = np.arange(nloc, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(int(i), int(j))

    loc_by_global = {int(c): k for k, c in enumerate(cand.tolist())}
    seed_set = np.intersect1d(
        np.asarray(seed_indices, dtype=np.int64).ravel(),
        cand,
        assume_unique=False,
    )
    touched_roots: set[int] = set()
    for s in seed_set.tolist():
        li = loc_by_global.get(int(s))
        if li is not None:
            touched_roots.add(find(li))

    roots = np.array([find(i) for i in range(nloc)], dtype=np.int64)
    candidate_roots: set[int] = touched_roots if touched_roots else set(np.unique(roots).tolist())

    best_root: int | None = None
    best_size = -1
    for r in candidate_roots:
        sz = int(np.sum(roots == r))
        if sz > best_size:
            best_size = sz
            best_root = int(r)
    assert best_root is not None
    mask = roots == best_root
    return np.sort(cand[mask].astype(np.int64))


def filter_components_by_mask_projection(
    *,
    positions: np.ndarray,
    candidate_indices: np.ndarray,
    mask: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    world_view_transform: np.ndarray,
    link_radius_m: float,
    depth_tolerance_abs_m: float,
    depth_tolerance_rel: float,
    max_components: int,
    score_min: float,
    min_good_points: int = 80,
    min_component_size: int = 250,
    neighbor_radius_m: float | None = None,
    seed_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Keep multiple 3D components whose projected points overlap the SAM mask (with depth gate).

    This is a conservative alternative to `largest_component_touching_seeds` for objects that
    are fragmented into multiple components (e.g., seat and backrest split by sparse sampling).
    """
    cand = np.unique(np.asarray(candidate_indices, dtype=np.int64).ravel())
    cand = cand[(cand >= 0) & (cand < positions.shape[0])]
    if cand.size == 0:
        return cand.astype(np.int64), {"kept_components": 0, "total_components": 0}

    # Optional neighborhood gate: only evaluate candidates near seeds (helps reject walls).
    if neighbor_radius_m is not None and seed_indices is not None and float(neighbor_radius_m) > 0.0:
        seeds = np.unique(np.asarray(seed_indices, dtype=np.int64).ravel())
        seeds = seeds[(seeds >= 0) & (seeds < positions.shape[0])]
        if seeds.size > 0:
            tree = cKDTree(positions[seeds])
            d, _ = tree.query(positions[cand], k=1)
            cand = cand[d <= float(neighbor_radius_m)]
            if cand.size == 0:
                return cand.astype(np.int64), {"kept_components": 0, "total_components": 0}

    # Build connected components using the same link radius as mode-b CC.
    P = positions[cand]
    tree = cKDTree(P)
    r = float(link_radius_m)
    pairs = tree.query_pairs(r=r)
    nloc = int(cand.shape[0])
    parent = np.arange(nloc, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(int(i), int(j))

    roots = np.array([find(i) for i in range(nloc)], dtype=np.int64)
    uniq_roots, inv = np.unique(roots, return_inverse=True)

    H, W = mask.shape[:2]
    m = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_map, dtype=np.float64)
    # Anchor depth to object surface only: mask-only depth then min-filter in a small window.
    # (Avoid dilated-mask pixels comparing against wall depth.)
    try:
        from scipy.ndimage import minimum_filter
    except Exception:
        minimum_filter = None
    depth_obj = np.where(m & np.isfinite(depth) & (depth > 1e-6), depth, np.inf)
    if minimum_filter is not None:
        depth_ref = minimum_filter(depth_obj, size=5, mode="nearest")
    else:
        depth_ref = depth_obj

    # IMPORTANT: match 3DGS CUDA convention (row-vector homogeneous): Xc_h = Xw_h @ W
    from segmentation.mask_to_gaussians import project_world_to_pixels

    u_all, v_all, z_all = project_world_to_pixels(K, world_view_transform, P)
    in_front = z_all > 1e-6
    ui = np.rint(u_all).astype(np.int64)
    vi = np.rint(v_all).astype(np.int64)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & in_front & np.isfinite(z_all)
    inside_mask = np.zeros_like(in_bounds, dtype=bool)
    z_ok = np.zeros_like(in_bounds, dtype=bool)
    idxb = np.where(in_bounds)[0]
    if idxb.size > 0:
        inside_mask[idxb] = m[vi[idxb], ui[idxb]]
        zr = depth_ref[vi[idxb], ui[idxb]]
        tol = float(depth_tolerance_abs_m) + float(depth_tolerance_rel) * np.where(np.isfinite(zr), zr, 0.0)
        # Directional: reject points behind the visible surface.
        z_ok[idxb] = np.isfinite(zr) & (z_all[idxb] <= (zr + tol))

    pass_gate = inside_mask & z_ok
    # Score components by fraction of *all* points that pass the gate (robust when many points
    # project out-of-frame). Also keep raw counts for debugging.
    scores: list[tuple[int, float, int, int, int]] = []
    for ci, root in enumerate(uniq_roots.tolist()):
        loc = np.where(inv == ci)[0]
        if loc.size == 0:
            continue
        if int(loc.size) < int(min_component_size):
            continue
        inb = int(np.sum(in_bounds[loc]))
        good = int(np.sum(pass_gate[loc]))
        score = float(good / max(1, int(loc.size)))
        scores.append((int(root), score, good, inb, int(loc.size)))

    scores.sort(key=lambda t: t[1], reverse=True)
    kept_roots = [
        r0
        for (r0, sc, good, _inb, _sz) in scores
        if (sc >= float(score_min) and int(good) >= int(min_good_points))
    ][: int(max_components)]
    if not kept_roots:
        # Fallback: keep the single best-scoring component.
        if scores:
            kept_roots = [scores[0][0]]
    keep_mask = np.isin(roots, np.array(kept_roots, dtype=np.int64))
    kept = np.sort(cand[keep_mask].astype(np.int64))
    dbg = {
        "component_split_radius_m": r,
        "total_components": int(len(uniq_roots)),
        "kept_components": int(len(set(kept_roots))),
        "kept_roots": kept_roots,
        "top_components": [
            {"root": r0, "score": float(sc), "good": int(g), "in_bounds": int(inb), "size": int(sz)}
            for (r0, sc, g, inb, sz) in scores[: min(12, len(scores))]
        ],
    }
    return kept, dbg


def selection_xyz_bounds(positions: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < positions.shape[0])]
    if idx.size == 0:
        z = np.zeros(3, dtype=np.float64)
        return z.copy(), z.copy()
    pts = positions[idx]
    return pts.min(axis=0).astype(np.float64), pts.max(axis=0).astype(np.float64)


def save_subset_points_ply(
    positions: np.ndarray,
    indices: np.ndarray,
    out_path: Path,
    *,
    rgb: tuple[float, float, float] = (1.0, 0.2, 0.05),
) -> None:
    """Minimal ASCII PLY point cloud for quick inspection (MeshLab / CloudCompare)."""
    idx = np.asarray(indices, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < positions.shape[0])]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts = positions[idx]
    n = pts.shape[0]
    r, g, b = rgb
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    body = "\n".join(
        f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(r * 255)} {int(g * 255)} {int(b * 255)}"
        for x, y, z in pts
    )
    out_path.write_text("\n".join(lines) + "\n" + body + ("\n" if body else ""), encoding="utf-8")
