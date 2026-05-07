from __future__ import annotations

from pathlib import Path

import numpy as np

from rendering.gs_mesh_composite import sorted_centroid_npy_paths, sorted_gauss_npy_paths
from rendering.mesh_gaussian_skin import skinning_cache_valid


def test_sorted_centroid_npy_paths_numeric_order(tmp_path: Path) -> None:
    d = tmp_path / "mesh_centroids"
    d.mkdir()
    (d / "centroid_0010.npy").write_bytes(b"x")
    (d / "centroid_0002.npy").write_bytes(b"x")
    (d / "centroid_0000.npy").write_bytes(b"x")
    paths = sorted_centroid_npy_paths(d)
    assert [p.name for p in paths] == ["centroid_0000.npy", "centroid_0002.npy", "centroid_0010.npy"]


def test_skinning_cache_valid_rejects_stale_indices() -> None:
    """Old mesh_skin.npz can index more Gaussians than current gauss_0000.npy rows."""
    nv, kk, ng = 4, 3, 50
    idx = np.zeros((nv, kk), dtype=np.int64)
    idx[:, :] = np.array([0, 1, 2], dtype=np.int64)
    w = np.ones((nv, kk), dtype=np.float64) / kk
    assert skinning_cache_valid(idx, w, n_gauss=ng, n_vertices=nv, k_request=3)
    assert not skinning_cache_valid(idx, w, n_gauss=2, n_vertices=nv, k_request=3)


def test_sorted_gauss_npy_paths_numeric_order(tmp_path: Path) -> None:
    d = tmp_path / "mesh_gauss_xyz"
    d.mkdir()
    (d / "gauss_0009.npy").write_bytes(b"x")
    (d / "gauss_0001.npy").write_bytes(b"x")
    paths = sorted_gauss_npy_paths(d)
    assert [p.name for p in paths] == ["gauss_0001.npy", "gauss_0009.npy"]
