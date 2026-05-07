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
