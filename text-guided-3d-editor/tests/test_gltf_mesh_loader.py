from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from generation.gltf_mesh_loader import load_glb_colored_primitive_meshes, load_trimesh_from_glb_pygltflib
from generation.mesh_to_gaussian import _sample_surface_point_rgb, load_insertion_mesh


def test_pygltflib_recovers_basecolor_texture(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    im = Image.new("RGB", (32, 32), (40, 200, 80))
    vis = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.SimpleMaterial(image=im))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=vis, process=False)
    glb = tmp_path / "x.glb"
    mesh.export(str(glb))

    m2 = load_trimesh_from_glb_pygltflib(glb, log=None)
    assert m2 is not None
    pts = np.array([[0.25, 0.25, 0.0]], dtype=np.float64)
    fi = np.array([0], dtype=np.int64)
    rgb, src = _sample_surface_point_rgb(m2, pts, fi)
    assert src == "uv_texture"
    assert int(rgb[0, 1]) > 80


def test_load_insertion_mesh_glb_prefers_pygltflib(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    im = Image.new("RGB", (16, 16), (200, 50, 50))
    vis = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.SimpleMaterial(image=im))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=vis, process=False)
    glb = tmp_path / "duckish.glb"
    mesh.export(str(glb))
    parts = load_glb_colored_primitive_meshes(glb, log=None)
    assert parts is not None and len(parts) == 1
    rgb, src = _sample_surface_point_rgb(parts[0], np.array([[0.25, 0.25, 0.0]]), np.array([0]))
    assert src == "uv_texture"
    assert int(rgb[0, 0]) > 80
    # Placement path merges geometry only (no texture) — mesh exists for export.
    m_merge = load_insertion_mesh(glb)
    assert len(m_merge.faces) == 1
