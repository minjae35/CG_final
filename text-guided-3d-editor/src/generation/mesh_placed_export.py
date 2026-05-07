"""Export DreamGaussian ``object_mesh.obj`` into scene coordinates (same as merged PLY)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from generation.mesh_to_gaussian import load_insertion_mesh
from generation.object_rescaler import transform_mesh_vertices_like_rescaled_ply


def export_placed_dreamgaussian_mesh(
    dg_mesh_obj: Path | str,
    out_path: Path | str,
    *,
    object_ply_path: Path | str,
    base_ply_path: Path | str,
    scale_factor: float,
    canonical_frame: str,
    post_rotation_deg: tuple[float, float, float],
    align_y_axis_to: np.ndarray | None,
    merge_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load DG mesh, transform vertices like ``object_scaled.ply``, write ``out_path`` (OBJ)."""
    dg_mesh_obj, out_path = Path(dg_mesh_obj), Path(out_path)
    mesh = load_insertion_mesh(dg_mesh_obj)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    Vw = transform_mesh_vertices_like_rescaled_ply(
        V,
        object_ply_path,
        base_ply_path,
        scale_factor,
        canonical_frame,
        post_rotation_deg,
        align_y_axis_to,
        merge_translation=np.asarray(merge_translation, dtype=np.float64).reshape(3),
    )
    out = trimesh.Trimesh(vertices=Vw, faces=F, process=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(out_path))
    return Vw, F
