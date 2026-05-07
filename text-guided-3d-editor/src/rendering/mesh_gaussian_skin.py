"""KNN soft skinning: transfer Gaussian world-space motion to mesh vertices."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def skinning_indices_weights(
    mesh_vertices: np.ndarray,
    gauss_xyz0: np.ndarray,
    *,
    k: int = 12,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(idx, w)`` with shapes ``(Nv, k)`` int64 and ``(Nv, k)`` float64, rows of ``w`` sum to 1."""
    from scipy.spatial import cKDTree

    mv = np.asarray(mesh_vertices, dtype=np.float64)
    g0 = np.asarray(gauss_xyz0, dtype=np.float64)
    if mv.ndim != 2 or mv.shape[1] != 3:
        raise ValueError("mesh_vertices must be (Nv, 3)")
    if g0.ndim != 2 or g0.shape[1] != 3:
        raise ValueError("gauss_xyz0 must be (Ng, 3)")
    kk = min(int(k), int(g0.shape[0]))
    if kk < 1:
        raise ValueError("empty gaussians")
    tree = cKDTree(g0)
    dists, idx = tree.query(mv, k=kk, workers=-1)
    if kk == 1:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 1)
        dists = np.asarray(dists, dtype=np.float64).reshape(-1, 1)
    else:
        idx = np.asarray(idx, dtype=np.int64)
        dists = np.asarray(dists, dtype=np.float64)
    d2 = np.maximum(dists**2, eps)
    w = 1.0 / d2
    w /= w.sum(axis=1, keepdims=True)
    return idx, w


def skinning_cache_valid(
    idx: np.ndarray,
    weights: np.ndarray,
    *,
    n_gauss: int,
    n_vertices: int,
    k_request: int,
) -> bool:
    """True if ``mesh_skin.npz`` matches current bind-pose Gaussian count and vertex/K layout."""
    idx = np.asarray(idx)
    w = np.asarray(weights)
    if idx.ndim != 2 or w.ndim != 2 or idx.shape != w.shape:
        return False
    if int(n_vertices) != int(idx.shape[0]) or idx.size == 0:
        return False
    kk = min(int(k_request), int(n_gauss))
    if kk < 1 or int(idx.shape[1]) != kk:
        return False
    if int(idx.min()) < 0 or int(idx.max()) >= int(n_gauss):
        return False
    return True


def deform_mesh_vertices(
    mesh_rest: np.ndarray,
    gauss0: np.ndarray,
    gausst: np.ndarray,
    idx: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """``mesh_rest + weighted sum of (gausst - gauss0)`` at KNN Gaussian indices."""
    mr = np.asarray(mesh_rest, dtype=np.float64)
    g0 = np.asarray(gauss0, dtype=np.float64)
    gt = np.asarray(gausst, dtype=np.float64)
    if g0.shape != gt.shape:
        raise ValueError(f"gauss0 {g0.shape} vs gausst {gt.shape}")
    if int(idx.max()) >= g0.shape[0] or int(idx.min()) < 0:
        raise ValueError(
            "gauss index out of range for gauss0/gausst "
            "(stale mesh_skin.npz vs mesh_gauss_xyz counts? delete mesh_skin.npz and retry)"
        )
    delta = gt - g0
    nei = delta[idx]
    disp = (weights[..., None] * nei).sum(axis=1)
    return (mr + disp).astype(np.float64)


def save_skinning_npz(path: Path | str, idx: np.ndarray, weights: np.ndarray, k: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, idx=idx, w=weights, k=np.array([k], dtype=np.int32))


def load_skinning_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(str(path))
    return z["idx"], z["w"]
