"""PhysGaussian / Warp MPM sanity checks before launching gs_simulation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


def _vertex_dtype_names(vertex: Any) -> set[str]:
    data = vertex.data
    if data.dtype.names:
        return set(data.dtype.names)
    return {str(d[0]) for d in data.dtype.descr}


def approximate_mpm_normalized_positions(world_xyz: np.ndarray) -> np.ndarray:
    """Match PhysGaussian ``transform2origin`` + ``shift2center111`` (no extra rotation)."""
    if world_xyz.size == 0:
        return world_xyz
    lo = world_xyz.min(axis=0).astype(np.float64)
    hi = world_xyz.max(axis=0).astype(np.float64)
    max_diff = float(np.max(hi - lo))
    if max_diff < 1e-12:
        max_diff = 1e-12
    scale = 1.0 / max_diff
    mean = (lo + hi) * 0.5
    out = (world_xyz.astype(np.float64) - mean) * scale
    out += np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return out


def validate_merged_object_for_mpm(
    merged_ply: Path | str,
    object_indices: np.ndarray,
    *,
    grid_lim: float = 2.0,
    edge_eps: float = 0.04,
    max_scale_logit_abs: float = 12.0,
    max_opacity_logit_abs: float = 12.0,
) -> dict[str, Any]:
    """Load object rows from merged PLY; assert finite fields and MPM bbox inside ``[0, grid_lim]``.

    Returns diagnostics dict (min/max world, min/max MPM, counts).
    """
    merged_ply = Path(merged_ply)
    ply = PlyData.read(str(merged_ply))
    v = ply["vertex"]
    n = len(v)
    idx = np.asarray(object_indices, dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"object_indices out of range [0,{n}): min={idx.min()} max={idx.max()}")

    names = _vertex_dtype_names(v)
    need = ("x", "y", "z")
    for k in need:
        if k not in names:
            raise ValueError(f"merged PLY vertex missing {k!r}")

    x = np.asarray(v["x"], dtype=np.float64)[idx]
    y = np.asarray(v["y"], dtype=np.float64)[idx]
    z = np.asarray(v["z"], dtype=np.float64)[idx]
    pos = np.stack([x, y, z], axis=1)

    if not np.isfinite(pos).all():
        bad = int(np.sum(~np.isfinite(pos)))
        raise ValueError(f"object xyz has {bad} non-finite values")

    for field, lim in (
        ("scale_0", max_scale_logit_abs),
        ("scale_1", max_scale_logit_abs),
        ("scale_2", max_scale_logit_abs),
    ):
        if field in names:
            s = np.asarray(v[field], dtype=np.float64)[idx]
            if not np.isfinite(s).all():
                raise ValueError(f"object {field} has non-finite values")
            if float(np.max(np.abs(s))) > lim:
                raise ValueError(
                    f"object {field} magnitude too large (>{lim}); check PLY / rescale"
                )

    if "opacity" in names:
        o = np.asarray(v["opacity"], dtype=np.float64)[idx]
        if not np.isfinite(o).all():
            raise ValueError("object opacity has non-finite values")
        if float(np.max(np.abs(o))) > max_opacity_logit_abs:
            raise ValueError("object opacity logit extreme; may destabilize filling / MPM")

    mpm = approximate_mpm_normalized_positions(pos)
    lo_m = mpm.min(axis=0)
    hi_m = mpm.max(axis=0)
    if np.any(lo_m < -edge_eps) or np.any(hi_m > grid_lim + edge_eps):
        raise ValueError(
            "object Gaussians extend outside approximate MPM domain "
            f"[0, {grid_lim}] after origin transform (min={lo_m.tolist()} max={hi_m.tolist()}). "
            "Try smaller object / different placement."
        )

    return {
        "n_object": int(len(idx)),
        "world_xyz_min": pos.min(axis=0).tolist(),
        "world_xyz_max": pos.max(axis=0).tolist(),
        "mpm_norm_min": lo_m.tolist(),
        "mpm_norm_max": hi_m.tolist(),
    }


def log_phys_preflight(
    console: Any,
    diag: dict[str, Any],
    sim_area: list[float] | None,
    *,
    n_grid: int,
    substep_dt: float,
    frame_dt: float,
    material: str,
    floor_friction: float,
    gravity_y_cfg: float,
) -> None:
    console.print("[bold cyan]MPM preflight[/] (object branch, merged PLY)")
    console.print(f"  object Gaussians: {diag['n_object']}")
    console.print(f"  world xyz min: {diag['world_xyz_min']}")
    console.print(f"  world xyz max: {diag['world_xyz_max']}")
    console.print(f"  approx MPM norm min: {diag['mpm_norm_min']}")
    console.print(f"  approx MPM norm max: {diag['mpm_norm_max']}")
    if sim_area is not None:
        console.print(f"  sim_area (world, xmin..ymax..): {sim_area}")
    console.print(
        f"  n_grid={n_grid}  substep_dt={substep_dt}  frame_dt={frame_dt}  "
        f"material={material}  floor_friction={floor_friction}  gravity(cfg)={gravity_y_cfg}"
    )
