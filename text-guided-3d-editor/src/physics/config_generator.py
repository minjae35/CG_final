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


def _triple_floats(
    v: tuple[float, float, float] | list[float] | None,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if v is None or len(v) < 3:
        return default
    return (float(v[0]), float(v[1]), float(v[2]))


def _quad_floats(
    v: tuple[float, float, float, float] | list[float] | None,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if v is None or len(v) < 4:
        return default
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


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
    # If True (recommended for mode-b): L/R mirror velocity boxes → near-zero net +X impulse vs single-axis shear.
    shear_symmetric_lr_split: bool = True,
    wobble_velocity: float = 0.16,
    wobble_end_time: float = 0.04,
    sim_area_margin: float = 0.2,
    material_overrides: dict[str, Any] | None = None,
    pin_initial_com_mpm: bool = False,
    pin_com_vertical_only: bool = False,
    pin_zero_mean_velocity_gs: bool = False,
    mode_b_mpm_displacement_retention: float | None = None,
    mode_b_render_freeze_gaussian_cov: bool = False,
    mode_b_render_cov_override: str = "none",
    mode_b_render_cov_override_scope: str = "selected",
    mode_b_render_tiny_splats_var: float = 1.0e-6,
    mode_b_render_cov_diag_min: float = 1.0e-6,
    mode_b_render_cov_diag_max: float = 5.0e-3,
    mode_b_disp_propagation_enabled: bool = False,
    mode_b_disp_propagation_k: int = 8,
    mode_b_disp_low_percentile: float = 10.0,
    mode_b_disp_min_neighbor_moved_m: float = 0.05,
    mode_b_disp_propagation_alpha: float = 1.0,
    mode_b_save_selected_means3d_per_frame: bool = False,
    mode_b_debug_dump_selected_only_renders: bool = False,
    mode_b_extra_render_dump_variants: str = "all",
    mode_b_sand_render_override: bool = False,
    mode_b_sand_opacity_scale: float = 1.0,
    mode_b_sand_opacity_min: float = 0.05,
    mode_b_sand_opacity_max: float = 0.85,
    mode_b_sand_cov_scale: float = 1.0,
    mode_b_sand_color_override_rgb: list[float] | None = None,
    mode_b_sand_motion_correction: bool = False,
    mode_b_sand_motion_alpha: float = 0.65,
    mode_b_sand_global_dy_percentile: float = 60.0,
    mode_b_sand_min_dy_as_global_frac: float | None = 0.15,
    mode_b_sand_extreme_all_selected_down: bool = False,
    mode_b_sand_extreme_total_down_m: float = 0.60,
    mode_b_render_gaussian_subset: str = "all",
    mode_b_mpm_kabsch_rigid_strip: bool = False,
    mode_b_mpm_kabsch_elastic_amp: float = 1.0,
    mode_b_mpm_anchor_feet_y_percentile: float | None = None,
    mode_b_mpm_tilt_diagnostics: bool = False,
    mode_b_mpm_tilt_top_y_percentile: float = 32.0,
    # Mode-b only: use tall L/R slabs covering tabletop+mid+legs (see pipeline mode_b).
    shear_wobble_full_volume: bool = False,
    # PhysGaussian MPM: sinusoidal shear velocity Dirichlet for full sim horizon (short impulse omitted).
    phys_sustained_wobble: bool = False,
    phys_wobble_frequency_hz: float = 1.25,
    phys_wobble_velocity_peak: float = 0.085,
    phys_wobble_force_scale: float = 1.0,
    phys_wobble_decay_lambda_per_s: float = 0.0,
    phys_wobble_duration_s: float | None = None,
    phys_wobble_ramp_time_s: float = 0.4,
    phys_wobble_velocity_peak_x: float | None = None,
    phys_wobble_velocity_peak_y: float = 0.0,
    phys_wobble_phase_y_rad: float = 1.5707963267948966,
    phys_wobble_band_amp_x: list[float] | None = None,
    phys_wobble_band_amp_y: list[float] | None = None,
    phys_wobble_band_vy_polarity: list[float] | None = None,
    phys_wobble_band_phase_y_offset_rad: list[float] | None = None,
    phys_wobble_velocity_peak_z: float = 0.0,
    phys_wobble_velocity_peak_diag1: float = 0.0,
    phys_wobble_velocity_peak_diag2: float = 0.0,
    phys_wobble_velocity_peak_twist: float = 0.0,
    phys_wobble_phase_z_rad: float = 0.0,
    phys_wobble_phase_diag1_rad: float = 0.7853981633974483,
    phys_wobble_phase_diag2_rad: float = 2.356194490192345,
    phys_wobble_phase_twist_rad: float = 1.0471975511965976,
    phys_wobble_bundle_quad_phase_rad: list[float] | None = None,
    phys_wobble_bundle_band_phase_rad: list[float] | None = None,
    phys_wobble_band_amp_z: list[float] | None = None,
    phys_wobble_band_amp_diag1: list[float] | None = None,
    phys_wobble_band_amp_diag2: list[float] | None = None,
    phys_wobble_band_amp_twist: list[float] | None = None,
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
        # +Y is world-down: ``y_lo_mpm`` ≈ head, ``y_hi_mpm`` ≈ feet.
        y_lo_mpm = (float(sim_pts[:, 1].min()) - float(mean_pos[1])) * scale + 1.0
        y_hi_mpm = (float(sim_pts[:, 1].max()) - float(mean_pos[1])) * scale + 1.0
        y_span = max(1e-6, float(y_hi_mpm - y_lo_mpm))
        v_top = float(wobble_velocity)
        v_mid = -0.58 * float(wobble_velocity)
        xz_half = 1.18
        phys_horizon = (
            float(phys_wobble_duration_s)
            if phys_wobble_duration_s is not None
            else float(frame_num) * float(frame_dt) + 0.5
        )
        peak_x_scaled = float(
            phys_wobble_velocity_peak
            if phys_wobble_velocity_peak_x is None
            else phys_wobble_velocity_peak_x
        ) * float(phys_wobble_force_scale)
        peak_y_scaled = float(phys_wobble_velocity_peak_y) * float(phys_wobble_force_scale)
        sine_common = (
            ("sinusoidal", float(phys_wobble_frequency_hz), float(phys_wobble_decay_lambda_per_s))
            if phys_sustained_wobble
            else ("constant", 0.0, 0.0)
        )

        if phys_sustained_wobble and shear_symmetric_lr_split:
            prof, freq_hz, decay_lam = sine_common
            if prof != "sinusoidal":
                raise ValueError("phys_sustained_wobble expects sinusoidal time profile")
            bax = _triple_floats(phys_wobble_band_amp_x, (1.0, 1.0, 1.0))
            bay = _triple_floats(phys_wobble_band_amp_y, (1.0, 0.55, 0.45))
            vpol = _triple_floats(phys_wobble_band_vy_polarity, (1.0, 0.38, -0.82))
            phb = _triple_floats(phys_wobble_band_phase_y_offset_rad, (0.0, 0.1, -0.12))
            hr = (1.0, -0.58, 0.52)
            bz = _triple_floats(phys_wobble_band_amp_z, (1.0, 0.92, 0.85))
            bd1 = _triple_floats(phys_wobble_band_amp_diag1, (1.0, 0.9, 0.84))
            bd2 = _triple_floats(phys_wobble_band_amp_diag2, (1.0, 0.9, 0.84))
            btw = _triple_floats(phys_wobble_band_amp_twist, (1.0, 0.95, 0.9))
            qb = _quad_floats(
                phys_wobble_bundle_quad_phase_rad,
                (0.0, 0.52, 1.05, 1.58),
            )
            bb = _triple_floats(phys_wobble_bundle_band_phase_rad, (0.0, 0.2, -0.14))
            peak_z_scaled = float(phys_wobble_velocity_peak_z) * float(
                phys_wobble_force_scale
            )
            pk_d1 = float(phys_wobble_velocity_peak_diag1) * float(phys_wobble_force_scale)
            pk_d2 = float(phys_wobble_velocity_peak_diag2) * float(phys_wobble_force_scale)
            pk_tw = float(phys_wobble_velocity_peak_twist) * float(phys_wobble_force_scale)
            xq = 0.32
            zq = 0.32
            x_hw_q = 0.22
            z_hw_q = 0.22
            quad_xc = (1.0 - xq, 1.0 - xq, 1.0 + xq, 1.0 + xq)
            quad_zc = (1.0 - zq, 1.0 + zq, 1.0 - zq, 1.0 + zq)
            bands_frac = ((0.03, 0.38), (0.40, 0.68), (0.70, 0.97))
            for bi in range(3):
                y0f, y1f = bands_frac[bi]
                y0 = float(y_lo_mpm + y0f * y_span)
                y1 = float(y_lo_mpm + y1f * y_span)
                y_c = 0.5 * (y0 + y1)
                y_h = max(0.5 * (y1 - y0), 0.035 * y_span)
                phase_y_e = float(phys_wobble_phase_y_rad) + phb[bi]
                phase_z_e = float(phys_wobble_phase_z_rad) + phb[bi] * 0.42
                bundle_band = bb[bi]
                for qi in range(4):
                    lr = 1.0 if qi in (0, 1) else -1.0
                    fz = 1.0 if qi in (0, 2) else -1.0
                    d1s = 1.0 if qi in (0, 3) else -1.0
                    d2s = 1.0 if qi in (1, 2) else -1.0
                    tws = 1.0 if qi in (0, 2) else -1.0
                    vx_axis = peak_x_scaled * bax[bi] * hr[bi] * lr
                    vy_b = peak_y_scaled * bay[bi] * vpol[bi]
                    vz_axis = peak_z_scaled * bz[bi] * fz
                    diag1_b = pk_d1 * bd1[bi] * d1s
                    diag2_b = pk_d2 * bd2[bi] * d2s
                    twist_b = pk_tw * btw[bi] * tws
                    bundle_ph = qb[qi] + bundle_band
                    boundary_conditions.append(
                        {
                            "type": "enforce_particle_translation",
                            "point": [float(quad_xc[qi]), y_c, float(quad_zc[qi])],
                            "size": [float(x_hw_q), float(y_h), float(z_hw_q)],
                            "velocity": [
                                float(vx_axis),
                                float(vy_b),
                                float(vz_axis),
                            ],
                            "start_time": 0.0,
                            "end_time": float(phys_horizon),
                            "velocity_profile": "sinusoidal_jelly_multi",
                            "frequency_hz": freq_hz,
                            "phase_rad": 0.0,
                            "phase_y_rad": float(phase_y_e),
                            "phase_z_rad": float(phase_z_e),
                            "phase_diag1_rad": float(phys_wobble_phase_diag1_rad)
                            + phb[bi] * 0.18,
                            "phase_diag2_rad": float(phys_wobble_phase_diag2_rad)
                            - phb[bi] * 0.12,
                            "phase_twist_rad": float(phys_wobble_phase_twist_rad)
                            + phb[bi] * 0.25,
                            "diag1_amp": float(diag1_b),
                            "diag2_amp": float(diag2_b),
                            "twist_amp": float(twist_b),
                            "bundle_spatial_phase_rad": float(bundle_ph),
                            "decay_lambda_per_s": decay_lam,
                            "ramp_duration_s": float(phys_wobble_ramp_time_s),
                            "velocity_driver_kind": 3,
                        }
                    )

        elif phys_sustained_wobble:
            raise ValueError(
                "phys_sustained_wobble requires shear_symmetric_lr_split for zero-net L/R shear"
            )

        elif shear_symmetric_lr_split:
            t0, t1 = 0.0, float(wobble_end_time)

            u0 = float(y_lo_mpm + 0.02 * y_span)
            u1 = float(y_lo_mpm + 0.38 * y_span)
            u_c = 0.5 * (u0 + u1)
            u_h = max(0.5 * (u1 - u0), 0.04 * y_span)

            m0 = float(y_lo_mpm + 0.41 * y_span)
            m1 = float(y_lo_mpm + 0.72 * y_span)
            m_c = 0.5 * (m0 + m1)
            m_h = max(0.5 * (m1 - m0), 0.04 * y_span)

            # Separate L/R boxes at the same Y so net +X momentum cancels (reduces one-way torque).
            x_off = 0.48
            x_hw = 0.42
            if shear_wobble_full_volume:
                bands: list[tuple[float, float, float]] = [
                    (0.03, 0.38, float(v_top)),
                    (0.40, 0.68, float(-0.58 * v_top)),
                    (0.70, 0.97, float(0.52 * v_top)),
                ]
                for y0f, y1f, v_x in bands:
                    y0 = float(y_lo_mpm + y0f * y_span)
                    y1 = float(y_lo_mpm + y1f * y_span)
                    y_c = 0.5 * (y0 + y1)
                    y_h = max(0.5 * (y1 - y0), 0.035 * y_span)
                    boundary_conditions.append(
                        {
                            "type": "enforce_particle_translation",
                            "point": [float(1.0 - x_off), y_c, 1.0],
                            "size": [float(x_hw), float(y_h), xz_half],
                            "velocity": [v_x, 0.0, 0.0],
                            "start_time": t0,
                            "end_time": t1,
                        }
                    )
                    boundary_conditions.append(
                        {
                            "type": "enforce_particle_translation",
                            "point": [float(1.0 + x_off), y_c, 1.0],
                            "size": [float(x_hw), float(y_h), xz_half],
                            "velocity": [-v_x, 0.0, 0.0],
                            "start_time": t0,
                            "end_time": t1,
                        }
                    )
            else:
                boundary_conditions.append(
                    {
                        "type": "enforce_particle_translation",
                        "point": [float(1.0 - x_off), u_c, 1.0],
                        "size": [float(x_hw), float(u_h), xz_half],
                        "velocity": [v_top, 0.0, 0.0],
                        "start_time": t0,
                        "end_time": t1,
                    }
                )
                boundary_conditions.append(
                    {
                        "type": "enforce_particle_translation",
                        "point": [float(1.0 + x_off), u_c, 1.0],
                        "size": [float(x_hw), float(u_h), xz_half],
                        "velocity": [-v_top, 0.0, 0.0],
                        "start_time": t0,
                        "end_time": t1,
                    }
                )
                boundary_conditions.append(
                    {
                        "type": "enforce_particle_translation",
                        "point": [float(1.0 - x_off), m_c, 1.0],
                        "size": [float(x_hw), float(m_h), xz_half],
                        "velocity": [v_mid, 0.0, 0.0],
                        "start_time": t0,
                        "end_time": t1,
                    }
                )
                boundary_conditions.append(
                    {
                        "type": "enforce_particle_translation",
                        "point": [float(1.0 + x_off), m_c, 1.0],
                        "size": [float(x_hw), float(m_h), xz_half],
                        "velocity": [-v_mid, 0.0, 0.0],
                        "start_time": t0,
                        "end_time": t1,
                    }
                )
        else:
            t0, t1 = 0.0, float(wobble_end_time)
            u0 = float(y_lo_mpm + 0.02 * y_span)
            u1 = float(y_lo_mpm + 0.38 * y_span)
            u_c = 0.5 * (u0 + u1)
            u_h = max(0.5 * (u1 - u0), 0.04 * y_span)

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
    # Render-space artifact checks (consumed by main-repo shim).
    if isinstance(mode_b_render_cov_override, str) and mode_b_render_cov_override != "none":
        base["mode_b_render_cov_override"] = str(mode_b_render_cov_override)
        base["mode_b_render_cov_override_scope"] = str(mode_b_render_cov_override_scope)
        base["mode_b_render_tiny_splats_var"] = float(mode_b_render_tiny_splats_var)
        base["mode_b_render_cov_diag_min"] = float(mode_b_render_cov_diag_min)
        base["mode_b_render_cov_diag_max"] = float(mode_b_render_cov_diag_max)
    if bool(mode_b_disp_propagation_enabled):
        base["mode_b_disp_propagation_enabled"] = True
        base["mode_b_disp_propagation_k"] = int(mode_b_disp_propagation_k)
        base["mode_b_disp_low_percentile"] = float(mode_b_disp_low_percentile)
        base["mode_b_disp_min_neighbor_moved_m"] = float(mode_b_disp_min_neighbor_moved_m)
        base["mode_b_disp_propagation_alpha"] = float(mode_b_disp_propagation_alpha)
    if bool(mode_b_save_selected_means3d_per_frame):
        base["mode_b_save_selected_means3d_per_frame"] = True
    if bool(mode_b_debug_dump_selected_only_renders):
        base["mode_b_debug_dump_selected_only_renders"] = True
        if isinstance(mode_b_extra_render_dump_variants, str) and mode_b_extra_render_dump_variants != "all":
            base["mode_b_extra_render_dump_variants"] = str(mode_b_extra_render_dump_variants)
    # Sand render override knobs are consumed by the main-repo shim (no submodule edits).
    if bool(mode_b_sand_render_override):
        base["mode_b_sand_render_override"] = True
        base["mode_b_sand_opacity_scale"] = float(mode_b_sand_opacity_scale)
        base["mode_b_sand_opacity_min"] = float(mode_b_sand_opacity_min)
        base["mode_b_sand_opacity_max"] = float(mode_b_sand_opacity_max)
        base["mode_b_sand_cov_scale"] = float(mode_b_sand_cov_scale)
        if mode_b_sand_color_override_rgb is not None:
            base["mode_b_sand_color_override_rgb"] = [float(x) for x in mode_b_sand_color_override_rgb[:3]]
    # Sand motion correction knobs are also consumed by the shim (means3D-only render-time correction).
    if bool(mode_b_sand_motion_correction):
        base["mode_b_sand_motion_correction"] = True
        base["mode_b_sand_motion_alpha"] = float(mode_b_sand_motion_alpha)
        base["mode_b_sand_global_dy_percentile"] = float(mode_b_sand_global_dy_percentile)
        if mode_b_sand_min_dy_as_global_frac is not None:
            base["mode_b_sand_min_dy_as_global_frac"] = float(mode_b_sand_min_dy_as_global_frac)
    if bool(mode_b_sand_extreme_all_selected_down):
        base["mode_b_sand_extreme_all_selected_down"] = True
        base["mode_b_sand_extreme_total_down_m"] = float(mode_b_sand_extreme_total_down_m)
    if isinstance(mode_b_render_gaussian_subset, str) and mode_b_render_gaussian_subset != "all":
        base["mode_b_render_gaussian_subset"] = str(mode_b_render_gaussian_subset)
    if mode_b_mpm_kabsch_rigid_strip:
        base["mode_b_mpm_kabsch_rigid_strip"] = True
        base["mode_b_mpm_kabsch_elastic_amp"] = float(mode_b_mpm_kabsch_elastic_amp)
    if mode_b_mpm_anchor_feet_y_percentile is not None:
        ap = float(mode_b_mpm_anchor_feet_y_percentile)
        if 0.0 < ap < 100.0:
            base["mode_b_mpm_anchor_feet_y_percentile"] = ap
    if mode_b_mpm_tilt_diagnostics:
        base["mode_b_mpm_tilt_diagnostics"] = True
        base["mode_b_mpm_tilt_top_y_percentile"] = float(mode_b_mpm_tilt_top_y_percentile)
    if phys_sustained_wobble and in_place_wobble:
        base["mode_b_phys_sustained_wobble"] = True
        base["mode_b_phys_wobble_frequency_hz"] = float(phys_wobble_frequency_hz)
        base["mode_b_phys_wobble_velocity_peak"] = float(phys_wobble_velocity_peak)
        base["mode_b_phys_wobble_velocity_peak_x"] = float(
            phys_wobble_velocity_peak
            if phys_wobble_velocity_peak_x is None
            else phys_wobble_velocity_peak_x
        )
        base["mode_b_phys_wobble_velocity_peak_y"] = float(phys_wobble_velocity_peak_y)
        base["mode_b_phys_wobble_velocity_peak_z"] = float(phys_wobble_velocity_peak_z)
        base["mode_b_phys_wobble_velocity_peak_diag1"] = float(
            phys_wobble_velocity_peak_diag1
        )
        base["mode_b_phys_wobble_velocity_peak_diag2"] = float(
            phys_wobble_velocity_peak_diag2
        )
        base["mode_b_phys_wobble_velocity_peak_twist"] = float(
            phys_wobble_velocity_peak_twist
        )
        base["mode_b_phys_wobble_phase_y_rad"] = float(phys_wobble_phase_y_rad)
        base["mode_b_phys_wobble_phase_z_rad"] = float(phys_wobble_phase_z_rad)
        base["mode_b_phys_wobble_force_scale"] = float(phys_wobble_force_scale)
        base["mode_b_phys_wobble_decay_lambda_per_s"] = float(
            phys_wobble_decay_lambda_per_s
        )
        if phys_wobble_duration_s is not None:
            base["mode_b_phys_wobble_duration_s"] = float(phys_wobble_duration_s)
        base["mode_b_phys_wobble_ramp_time_s"] = float(phys_wobble_ramp_time_s)
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
