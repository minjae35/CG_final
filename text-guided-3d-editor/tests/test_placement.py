from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from placement.ply_merger import merge_ply_files
from placement.surface_aware import find_surface_height


def _tiny_ply(path: Path, zoff: float = 0.0) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    v = np.zeros(2, dtype=dtype)
    v["z"] = [1.0 + zoff, 0.5 + zoff]
    v["opacity"] = 1.0
    v["rot_0"] = 1.0
    v["scale_0"] = v["scale_1"] = v["scale_2"] = -2.0
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def test_surface_height(tmp_path: Path) -> None:
    p = tmp_path / "s.ply"
    _tiny_ply(p)
    from placement.surface_aware import load_gaussian_xyz_opacity

    pos, op = load_gaussian_xyz_opacity(str(p))
    g = np.concatenate([pos, op[:, None]], axis=1)
    h = find_surface_height(g, 0.0, 0.0, search_radius=10.0, opacity_threshold=0.1)
    assert h >= 0.5


def test_merge_ply(tmp_path: Path) -> None:
    b = tmp_path / "b.ply"
    o = tmp_path / "o.ply"
    _tiny_ply(b, 0.0)
    _tiny_ply(o, 10.0)
    out = tmp_path / "m.ply"
    merge_ply_files(str(b), str(o), np.array([0.0, 0.0, 1.0]), str(out))
    m = PlyData.read(str(out))
    assert len(m["vertex"]) == 4
