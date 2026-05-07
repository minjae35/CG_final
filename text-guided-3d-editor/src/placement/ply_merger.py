"""Merge two 3DGS PLY files (PRD 3.3 + 4.2 schema alignment)."""
from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement


def align_gaussian_ply_properties(
    base_ply: PlyData,
    other_ply: PlyData,
) -> np.ndarray:
    """Return vertex array of other with dtype matching base (pad missing SH etc.)."""
    vb = base_ply["vertex"].data
    vo = other_ply["vertex"].data
    if vb.dtype == vo.dtype:
        return np.array(vo, copy=True)
    names = vb.dtype.names
    if names is None:
        raise ValueError("base ply has no vertex properties")
    out = np.zeros(len(vo), dtype=vb.dtype)
    for n in names:
        if n not in vo.dtype.names:
            if n.startswith("f_rest_") or n.startswith("f_dc_"):
                out[n] = 0.0
            elif n == "opacity":
                out[n] = 1.0
            elif n.startswith("scale_"):
                out[n] = -4.0
            elif n.startswith("rot_"):
                out[n] = 0.0 if n != "rot_0" else 1.0
            else:
                out[n] = 0.0
        else:
            out[n] = vo[n]
    return out


def merge_ply_files(
    base_ply: str,
    object_ply: str,
    translation: np.ndarray,
    output_ply: str,
) -> str:
    base = PlyData.read(base_ply)
    obj = PlyData.read(object_ply)
    vb = np.array(base["vertex"].data, copy=True)
    vo = align_gaussian_ply_properties(base, obj)
    vo = np.array(vo, copy=True)
    t = translation.astype(np.float64)
    vo["x"] = vo["x"] + t[0]
    vo["y"] = vo["y"] + t[1]
    vo["z"] = vo["z"] + t[2]
    merged = np.concatenate([vb, vo])
    el = PlyElement.describe(merged, "vertex")
    PlyData([el], text=False).write(output_ply)
    return output_ply
