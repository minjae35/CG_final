"""Build PhysGaussian JSON config (PRD 4.2)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import PROJECT_ROOT
from physics.material_presets import get_preset


def _bbox_from_indices(positions: np.ndarray, indices: np.ndarray, margin: float = 0.2) -> list[float]:
    """Return sim_area in PhysGaussian order: xmin, xmax, ymin, ymax, zmin, zmax (NOT xmin..zmin then xmax..)."""
    if len(indices) == 0:
        lo = positions.min(0)
        hi = positions.max(0)
    else:
        pts = positions[indices]
        lo, hi = pts.min(0), pts.max(0)
    ext = (hi - lo) * margin
    lo2, hi2 = lo - ext, hi + ext
    return [
        float(lo2[0]),
        float(hi2[0]),
        float(lo2[1]),
        float(hi2[1]),
        float(lo2[2]),
        float(hi2[2]),
    ]


def generate_phys_config(
    gaussian_indices: np.ndarray,
    gaussian_positions: np.ndarray,
    material_name: str,
    output_path: Path | str,
    template_path: Path | str | None = None,
    n_grid: int = 100,
    frame_num: int = 120,
    frame_dt: float = 0.01,
    substep_dt: float = 4e-4,
    gravity: float = -9.8,
    camera_index: int = 0,
    simulate_indices_npy: Path | str | None = None,
    particle_filling: dict | None = None,
    enable_internal_particle_fill: bool = True,
    floor_y: float | None = None,
    subtract_rigid_drift: bool = False,
    world_up: tuple[float, float, float] | None = None,
    floor_collider: bool = False,
    floor_friction: float = 0.5,
    in_place_wobble: bool = False,
    wobble_velocity: float = 0.16,
    wobble_end_time: float = 0.04,
    sim_area_margin: float = 0.2,
    material_overrides: dict[str, Any] | None = None,
    pin_initial_com_mpm: bool = False,
    pin_com_vertical_only: bool = False,
    pin_zero_mean_velocity_gs: bool = False,
    mode_b_mpm_displacement_retention: float | None = None,
    mode_b_render_freeze_gaussian_cov: bool = False,
    mode_b_mpm_kabsch_rigid_strip: bool = False,
    mode_b_mpm_kabsch_elastic_amp: float = 1.0,
    mode_b_mpm_anchor_feet_y_percentile: float | None = None,
) -> tuple[str, list[float]]:
    output_path = Path(output_path)
    preset = get_preset(material_name)
    sim_area = _bbox_from_indices(
        gaussian_positions, gaussian_indices, margin=float(sim_area_margin)
    )
    # In this scene +y is DOWN (camera down ≈ world +y). Extend sim_area ymax to
    # the actual visible floor so the duck falls to the floor instead of stopping
    # at the object bbox boundary.
    if floor_y is not None:
        sim_area[3] = float(floor_y) + 0.30

    base_particle_filling = {
        "n_grid": min(50, n_grid),
        "density_threshold": 5.0,
        "search_threshold": 0.5,
        "search_exclude_direction": 0,
        "ray_cast_direction": 1,
        "max_particles_num": 2000000,
        "max_partciels_per_cell": 1,
        # boundary must be null here: PhysGaussian applies transform2origin before
        # calling fill_particles, so positions are no longer in world space.
        # The sim_area filter is already applied upstream in gs_simulation.py.
        "boundary": None,
        "smooth": False,
        "visualize": False,
    }
    if particle_filling is not None:
        base_particle_filling.update(particle_filling)

    sim_pts = gaussian_positions[gaussian_indices]
    mins = np.min(sim_pts, axis=0)
    maxs = np.max(sim_pts, axis=0)
    mean_pos = (mins + maxs) / 2.0
    max_diff = float((maxs - mins).max())
    scale = 1.0 / max(max_diff, 1e-8)

    boundary_conditions: list[dict] = []
    if floor_collider and floor_y is not None:
        # PhysGaussian's ``transform2origin`` shifts the simulated particles to
        # the centre of the [0, grid_lim] cube and scales them by 1 / max_diff.
        # Reproduce the same transform here so the floor in WORLD coords can be
        # written as a plane equation in MPM space.
        floor_y_mpm = (float(floor_y) - float(mean_pos[1])) * scale + 1.0
        boundary_conditions.append(
            {
                "type": "surface_collider",
                "point": [1.0, float(floor_y_mpm), 1.0],
                # World up = -y → MPM up = -y; collider normal must point INTO
                # the side that contains the duck (i.e. -y in MPM).
                "normal": [0.0, -1.0, 0.0],
                "surface": "slip",
                "friction": float(floor_friction),
                "start_time": 0.0,
                "end_time": 999.0,
            }
        )
    if in_place_wobble:
        # Shear-style wobble in MPM space (``transform2origin`` + ``shift2center111``).
        # +Y is world-down: ``y_lo_mpm`` ≈ head, ``y_hi_mpm`` ≈ feet.  Two non-overlapping
        # horizontal slabs get opposite lateral ``vx`` so the impulse excites internal
        # deformation instead of a uniform body translation; the lowest ~28% of the
        # height (feet / floor contact band) is left without ``enforce`` so the base
        # stays more stable on the collider.
        y_lo_mpm = (float(sim_pts[:, 1].min()) - float(mean_pos[1])) * scale + 1.0
        y_hi_mpm = (float(sim_pts[:, 1].max()) - float(mean_pos[1])) * scale + 1.0
        y_span = max(1e-6, float(y_hi_mpm - y_lo_mpm))
        v_top = float(wobble_velocity)
        v_mid = -0.58 * float(wobble_velocity)
        xz_half = 1.18
        t0, t1 = 0.0, float(wobble_end_time)

        # Upper / head–chest band (~top 36% of height): +vx
        u0 = float(y_lo_mpm + 0.02 * y_span)
        u1 = float(y_lo_mpm + 0.38 * y_span)
        u_c = 0.5 * (u0 + u1)
        u_h = max(0.5 * (u1 - u0), 0.04 * y_span)

        # Mid torso band (~next 32%): opposite vx (shear); gap before feet band
        m0 = float(y_lo_mpm + 0.41 * y_span)
        m1 = float(y_lo_mpm + 0.72 * y_span)
        m_c = 0.5 * (m0 + m1)
        m_h = max(0.5 * (m1 - m0), 0.04 * y_span)

        boundary_conditions.append(
            {
                "type": "enforce_particle_translation",
                "point": [1.0, u_c, 1.0],
                "size": [xz_half, float(u_h), xz_half],
                "velocity": [v_top, 0.0, 0.0],
                "start_time": t0,
                "end_time": t1,
            }
        )
        boundary_conditions.append(
            {
                "type": "enforce_particle_translation",
                "point": [1.0, m_c, 1.0],
                "size": [xz_half, float(m_h), xz_half],
                "velocity": [v_mid, 0.0, 0.0],
                "start_time": t0,
                "end_time": t1,
            }
        )
    boundary_conditions.append({"type": "bounding_box"})

    base: dict = {
        "opacity_threshold": 0.02,
        "rotation_degree": [0.0],
        "rotation_axis": [0],
        "substep_dt": substep_dt,
        "frame_dt": frame_dt,
        "frame_num": frame_num,
        "g": [0.0, -float(gravity), 0.0],
        "n_grid": n_grid,
        "sim_area": sim_area,
        "boundary_conditions": boundary_conditions,
        "default_camera_index": camera_index,
    }
    if enable_internal_particle_fill:
        base["particle_filling"] = base_particle_filling
    if simulate_indices_npy is not None:
        base["simulate_indices_npy"] = str(Path(simulate_indices_npy).resolve())
    if subtract_rigid_drift:
        base["subtract_rigid_drift"] = True
    if pin_initial_com_mpm:
        base["pin_initial_com_mpm"] = True
    if pin_com_vertical_only:
        base["pin_com_vertical_only"] = True
    if pin_zero_mean_velocity_gs:
        base["pin_zero_mean_velocity_gs"] = True
    if (
        mode_b_mpm_displacement_retention is not None
        and 0.0 <= float(mode_b_mpm_displacement_retention) < 1.0
    ):
        base["mode_b_mpm_displacement_retention"] = float(mode_b_mpm_displacement_retention)
    if mode_b_render_freeze_gaussian_cov:
        base["mode_b_render_freeze_gaussian_cov"] = True
    if mode_b_mpm_kabsch_rigid_strip:
        base["mode_b_mpm_kabsch_rigid_strip"] = True
        base["mode_b_mpm_kabsch_elastic_amp"] = float(mode_b_mpm_kabsch_elastic_amp)
    if mode_b_mpm_anchor_feet_y_percentile is not None:
        ap = float(mode_b_mpm_anchor_feet_y_percentile)
        if 0.0 < ap < 100.0:
            base["mode_b_mpm_anchor_feet_y_percentile"] = ap
    if world_up is not None:
        base["mpm_space_vertical_upward_axis"] = [float(c) for c in world_up]
    base.update(preset)
    if material_overrides:
        base.update(material_overrides)

    if template_path and Path(template_path).is_file():
        with open(template_path, encoding="utf-8") as f:
            tpl = json.load(f)
        tpl.update(base)
        base = tpl

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
    return str(output_path), sim_area


def default_template_dir() -> Path:
    return PROJECT_ROOT / "configs" / "phys_templates"
