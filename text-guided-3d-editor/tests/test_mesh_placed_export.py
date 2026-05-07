from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

from generation.mesh_placed_export import export_placed_dreamgaussian_mesh


def _minimal_gaussian_ply(path: Path, center: tuple[float, float, float], spread: float = 0.1) -> None:
    dtype = [
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
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
    cx, cy, cz = center
    v = np.zeros(8, dtype=dtype)
    for i, (dx, dy, dz) in enumerate(
        [
            (-1, -1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (1, 1, 0),
            (-1, -1, 2),
            (1, -1, 2),
            (-1, 1, 2),
            (1, 1, 2),
        ]
    ):
        v["x"][i] = cx + dx * spread
        v["y"][i] = cy + dy * spread
        v["z"][i] = cz + dz * spread
        v["opacity"][i] = 1.0
        v["rot_0"][i] = 1.0
        v["scale_0"][i] = v["scale_1"][i] = v["scale_2"][i] = -2.0
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def test_export_placed_mesh_writes_obj(tmp_path: Path) -> None:
    obj_ply = tmp_path / "object.ply"
    base_ply = tmp_path / "base.ply"
    _minimal_gaussian_ply(obj_ply, (0.0, 0.0, 0.0), spread=0.05)
    _minimal_gaussian_ply(base_ply, (10.0, 10.0, 10.0), spread=1.0)

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh_path = tmp_path / "cube.obj"
    mesh.export(mesh_path)

    out = tmp_path / "placed.obj"
    Vw, F = export_placed_dreamgaussian_mesh(
        mesh_path,
        out,
        object_ply_path=obj_ply,
        base_ply_path=base_ply,
        scale_factor=0.5,
        canonical_frame="text",
        post_rotation_deg=(0.0, 0.0, 0.0),
        align_y_axis_to=None,
        merge_translation=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )
    assert out.is_file()
    assert Vw.shape[1] == 3
    assert F.ndim == 2 and F.shape[1] == 3
    assert np.isfinite(Vw).all()
