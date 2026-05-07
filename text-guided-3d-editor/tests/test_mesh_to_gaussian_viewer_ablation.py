from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from generation.mesh_to_gaussian import (
    mesh_to_gaussian_ply,
    write_axis_rotation_variant_plys,
    write_superplat_viewer_test_ply,
)


def _minimal_obj(tmp_path: Path) -> Path:
    obj = tmp_path / "tri.obj"
    obj.write_text(
        "o tri\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )
    return obj


def test_write_superplat_viewer_test_ply_forces_identity_and_isotropic_scales(tmp_path: Path) -> None:
    obj = _minimal_obj(tmp_path)
    raw = tmp_path / "g.ply"
    mesh_to_gaussian_ply(obj, raw, num_points=50, scale_logit=-2.0, opacity_logit=1.0)
    ply = PlyData.read(str(raw))
    v0 = np.array(ply["vertex"].data, copy=True)
    v0["rot_0"] = 0.7
    v0["rot_1"] = 0.1
    v0["rot_2"] = 0.2
    v0["rot_3"] = 0.3
    v0["scale_0"] = -1.0
    v0["scale_1"] = -3.0
    v0["scale_2"] = -2.0
    PlyData([PlyElement.describe(v0, "vertex")], text=False).write(str(raw))

    out = tmp_path / "g_test.ply"
    write_superplat_viewer_test_ply(raw, out, scale_logit=-2.5)
    v = PlyData.read(str(out))["vertex"].data
    assert np.allclose(v["rot_0"], 1.0)
    assert np.allclose(v["rot_1"], 0.0)
    assert np.allclose(v["rot_2"], 0.0)
    assert np.allclose(v["rot_3"], 0.0)
    assert np.allclose(v["scale_0"], -2.5)
    assert np.allclose(v["scale_1"], -2.5)
    assert np.allclose(v["scale_2"], -2.5)


def test_write_axis_rotation_variant_plys_writes_five_files(tmp_path: Path) -> None:
    obj = _minimal_obj(tmp_path)
    raw = tmp_path / "g.ply"
    mesh_to_gaussian_ply(obj, raw, num_points=30, scale_logit=-2.0, opacity_logit=1.0)
    outd = tmp_path / "variants"
    paths = write_axis_rotation_variant_plys(raw, outd)
    assert len(paths) == 5
    for fname in (
        "object_mesh_gaussians_rx180.ply",
        "object_mesh_gaussians_ry180.ply",
        "object_mesh_gaussians_rz180.ply",
        "object_mesh_gaussians_rx90.ply",
        "object_mesh_gaussians_rxm90.ply",
    ):
        p = outd / fname
        assert p.is_file()
        v = PlyData.read(str(p))["vertex"].data
        assert np.allclose(v["rot_0"], 1.0)
        assert np.allclose(v["scale_0"], v["scale_1"]) and np.allclose(v["scale_1"], v["scale_2"])
