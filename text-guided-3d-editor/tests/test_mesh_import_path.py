from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import SimpleMaterial

from generation.mesh_to_gaussian import (
    load_insertion_mesh,
    mesh_to_gaussian_ply,
    resolve_insertion_mesh_path,
    stage_external_mesh_into_gen_dir,
)
from generation.mesh_to_gaussian import _sample_surface_point_rgb


def test_resolve_insertion_mesh_path_prefers_glb(tmp_path: Path) -> None:
    (tmp_path / "object_mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (tmp_path / "object_mesh.glb").write_bytes(b"glb")
    # invalid glb bytes — resolve still picks glb path; load would fail
    assert resolve_insertion_mesh_path(tmp_path).name == "object_mesh.glb"


def test_resolve_insertion_mesh_path_falls_back_to_obj(tmp_path: Path) -> None:
    obj = tmp_path / "object_mesh.obj"
    obj.write_text(
        "o t\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    assert resolve_insertion_mesh_path(tmp_path) == obj


def test_resolve_insertion_mesh_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_insertion_mesh_path(tmp_path)


def test_stage_external_mesh_into_gen_dir_obj(tmp_path: Path) -> None:
    src = tmp_path / "duck.obj"
    trimesh.creation.box(extents=[0.2, 0.3, 0.4]).export(str(src))
    gen = tmp_path / "sds_run"
    out = stage_external_mesh_into_gen_dir(gen, src)
    assert out == gen / "object_mesh.obj"
    assert out.is_file()
    assert resolve_insertion_mesh_path(gen) == out


def test_load_insertion_mesh_scene_picks_one_submesh_no_concat(tmp_path: Path) -> None:
    a = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    b = trimesh.creation.box(
        extents=[0.2, 0.2, 0.2],
        transform=trimesh.transformations.translation_matrix([2.0, 0.0, 0.0]),
    )
    scene = trimesh.Scene([a, b])
    path = tmp_path / "two.glb"
    scene.export(str(path))
    m = load_insertion_mesh(path)
    assert isinstance(m, trimesh.Trimesh)
    # Concatenate is avoided so GLB materials survive; one submesh only.
    assert len(m.faces) == len(a.faces)


def test_corner_uv_texture_sampling() -> None:
    """glTF-style UV rows = 3 * n_faces (per-corner), not per-vertex."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    im = Image.new("RGB", (16, 16), (40, 220, 50))
    vis = TextureVisuals(uv=uv, material=SimpleMaterial(image=im))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=vis, process=False)
    pts = np.array([[0.25, 0.25, 0.0]], dtype=np.float64)
    fi = np.array([0], dtype=np.int64)
    rgb, src = _sample_surface_point_rgb(mesh, pts, fi)
    assert src == "uv_texture"
    assert int(rgb[0, 1]) > 100


def test_vertex_colors_used_in_sampling(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    vc = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=ColorVisuals(vertex_colors=vc), process=False)
    pts = np.array([[0.25, 0.25, 0.0]], dtype=np.float64)
    fi = np.array([0], dtype=np.int64)
    rgb, src = _sample_surface_point_rgb(mesh, pts, fi)
    assert src == "vertex_colors"
    assert rgb.shape == (1, 3)
    assert rgb[0].max() > 10


def test_mesh_to_gaussian_ply_writes_dc_from_colors(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    vc = np.array([[200, 30, 30, 255], [200, 30, 30, 255], [200, 30, 30, 255]], dtype=np.uint8)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=ColorVisuals(vertex_colors=vc), process=False)
    obj = tmp_path / "c.obj"
    mesh.export(str(obj))
    out = tmp_path / "out.ply"
    mesh_to_gaussian_ply(obj, out, num_points=100, apply_export_axis_rotation=False)
    from plyfile import PlyData

    v = PlyData.read(str(out))["vertex"].data
    assert np.mean(v["f_dc_0"]) != 0.0 or np.mean(v["f_dc_1"]) != 0.0
