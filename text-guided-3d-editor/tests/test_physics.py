from __future__ import annotations

from pathlib import Path

import numpy as np
import json

from physics.config_generator import generate_phys_config
from physics.material_presets import get_preset
from physics.mode_b_jelly import (
    apply_com_pin_mpm,
    estimate_mode_b_support_contact_y,
    max_vertical_com_drift_after_pinning,
)


def test_material_preset() -> None:
    j = get_preset("jelly")
    assert j["material"] == "jelly"
    r = get_preset("rubber")
    assert r["material"] == "jelly"
    assert r["E"] > j["E"]


def test_generate_phys_config(tmp_path: Path) -> None:
    pos = np.random.randn(100, 3).astype(np.float64)
    idx = np.arange(10, dtype=np.int64)
    p = tmp_path / "c.json"
    path, sim_area = generate_phys_config(idx, pos, "foam", p)
    assert p.is_file()
    assert path == str(p.resolve())
    assert len(sim_area) == 6


def test_in_place_wobble_shear_boundary_conditions(tmp_path: Path) -> None:
    """Two opposite lateral velocity slabs (upper vs mid); feet band has no enforce BC."""
    np.random.seed(0)
    pos = np.zeros((50, 3), dtype=np.float64)
    pos[:, 1] = np.linspace(0.0, 0.5, 50)
    pos[:, 0] = np.random.randn(50) * 0.01
    pos[:, 2] = np.random.randn(50) * 0.01
    idx = np.arange(50, dtype=np.int64)
    p = tmp_path / "wobble_shear.json"
    generate_phys_config(
        idx,
        pos,
        "rubber",
        p,
        in_place_wobble=True,
        wobble_velocity=0.05,
        wobble_end_time=0.08,
        enable_internal_particle_fill=False,
    )
    data = json.loads(p.read_text())
    en = [b for b in data["boundary_conditions"] if b["type"] == "enforce_particle_translation"]
    assert len(en) == 2
    assert en[0]["velocity"][0] * en[1]["velocity"][0] < 0
    assert abs(en[0]["velocity"][0]) > abs(en[1]["velocity"][0])


def test_generate_phys_config_omits_filling_when_disabled(tmp_path: Path) -> None:
    pos = np.random.randn(50, 3).astype(np.float64)
    idx = np.arange(5, dtype=np.int64)
    p = tmp_path / "no_fill.json"
    generate_phys_config(idx, pos, "foam", p, enable_internal_particle_fill=False)
    cfg = json.loads(p.read_text())
    assert "particle_filling" not in cfg


def test_generate_phys_config_material_overrides(tmp_path: Path) -> None:
    pos = np.random.randn(50, 3).astype(np.float64)
    idx = np.arange(5, dtype=np.int64)
    p = tmp_path / "jelly_override.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        material_overrides={"E": 12345.0, "grid_v_damping_scale": 0.99991},
        enable_internal_particle_fill=False,
    )
    cfg = json.loads(p.read_text())
    assert cfg["E"] == 12345.0
    assert cfg["grid_v_damping_scale"] == 0.99991


def test_generate_phys_config_with_explicit_indices(tmp_path: Path) -> None:
    pos = np.random.randn(100, 3).astype(np.float64)
    idx = np.arange(10, dtype=np.int64)
    indices_path = tmp_path / "indices.npy"
    np.save(indices_path, idx)
    p = tmp_path / "c.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        simulate_indices_npy=indices_path,
        particle_filling={"smooth": True, "max_partciels_per_cell": 4},
        subtract_rigid_drift=True,
        enable_internal_particle_fill=True,
    )
    cfg = json.loads(p.read_text())
    assert cfg["simulate_indices_npy"] == str(indices_path.resolve())
    assert cfg["subtract_rigid_drift"] is True
    assert cfg["particle_filling"]["smooth"] is True
    assert cfg["particle_filling"]["max_partciels_per_cell"] == 4


def test_generate_phys_config_freeze_render_cov(tmp_path: Path) -> None:
    pos = np.random.randn(20, 3).astype(np.float64)
    idx = np.arange(20, dtype=np.int64)
    p = tmp_path / "fcov.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        mode_b_render_freeze_gaussian_cov=True,
        enable_internal_particle_fill=False,
    )
    cfg = json.loads(p.read_text())
    assert cfg["mode_b_render_freeze_gaussian_cov"] is True


def test_generate_phys_config_kabsch_rigid_strip(tmp_path: Path) -> None:
    pos = np.random.randn(25, 3).astype(np.float64)
    idx = np.arange(25, dtype=np.int64)
    p = tmp_path / "kabsch.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        mode_b_mpm_kabsch_rigid_strip=True,
        mode_b_mpm_kabsch_elastic_amp=0.92,
        enable_internal_particle_fill=False,
    )
    cfg = json.loads(p.read_text())
    assert cfg["mode_b_mpm_kabsch_rigid_strip"] is True
    assert cfg["mode_b_mpm_kabsch_elastic_amp"] == 0.92


def test_generate_phys_config_anchor_feet_percentile(tmp_path: Path) -> None:
    pos = np.random.randn(20, 3).astype(np.float64)
    idx = np.arange(20, dtype=np.int64)
    p = tmp_path / "feet.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        mode_b_mpm_anchor_feet_y_percentile=88.0,
        enable_internal_particle_fill=False,
    )
    cfg = json.loads(p.read_text())
    assert cfg["mode_b_mpm_anchor_feet_y_percentile"] == 88.0


def test_estimate_mode_b_support_contact_y_high_percentile() -> None:
    pos = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    idx = np.arange(4, dtype=np.int64)
    y92 = estimate_mode_b_support_contact_y(pos, idx, contact_percentile=92.0)
    assert y92 >= 2.5
    y50 = estimate_mode_b_support_contact_y(pos, idx, contact_percentile=50.0)
    assert 1.0 < y50 < 2.5


def test_mode_b_phys_config_includes_com_pin_and_support_collider(tmp_path: Path) -> None:
    np.random.seed(1)
    pos = np.random.randn(80, 3).astype(np.float64)
    pos[:, 1] += 2.0
    idx = np.arange(80, dtype=np.int64)
    support_y = estimate_mode_b_support_contact_y(pos, idx, contact_percentile=90.0)
    p = tmp_path / "mode_b.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        floor_y=support_y,
        floor_collider=True,
        world_up=(0.0, -1.0, 0.0),
        pin_initial_com_mpm=True,
        pin_com_vertical_only=False,
        pin_zero_mean_velocity_gs=True,
        enable_internal_particle_fill=False,
    )
    cfg = json.loads(p.read_text())
    assert cfg["pin_initial_com_mpm"] is True
    assert cfg["pin_zero_mean_velocity_gs"] is True
    assert cfg["mpm_space_vertical_upward_axis"] == [0.0, -1.0, 0.0]
    assert any(b.get("type") == "surface_collider" for b in cfg["boundary_conditions"])


def test_com_pin_keeps_centroid_after_biased_steps() -> None:
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((60, 3)).astype(np.float64) * 0.03
    com0 = pts.mean(axis=0)
    for _ in range(200):
        pts[:, 1] += 0.004
        pts += rng.standard_normal(pts.shape) * 0.001
        pts, _ = apply_com_pin_mpm(pts, com0, vertical_only=False)
    assert np.allclose(pts.mean(axis=0), com0, atol=1e-9)


def test_generate_phys_config_displacement_retention(tmp_path: Path) -> None:
    pos = np.random.randn(30, 3).astype(np.float64)
    idx = np.arange(30, dtype=np.int64)
    p = tmp_path / "ret.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        mode_b_mpm_displacement_retention=0.88,
        enable_internal_particle_fill=False,
    )
    cfg = json.loads(p.read_text())
    assert cfg["mode_b_mpm_displacement_retention"] == 0.88


def test_mode_b_vertical_drift_bounded_under_vertical_only_pin() -> None:
    max_pre, final_dy = max_vertical_com_drift_after_pinning(
        initial_y=1.0,
        n_steps=400,
        gravity_step=0.04,
        noise=0.008,
        vertical_only=True,
    )
    assert max_pre > 0.01
    assert final_dy < 1e-7


def test_generate_phys_config_no_displacement_retention_at_one(tmp_path: Path) -> None:
    pos = np.random.randn(12, 3).astype(np.float64)
    idx = np.arange(12, dtype=np.int64)
    p = tmp_path / "r1.json"
    generate_phys_config(
        idx,
        pos,
        "jelly",
        p,
        mode_b_mpm_displacement_retention=1.0,
        enable_internal_particle_fill=False,
    )
    cfg = json.loads(p.read_text())
    assert "mode_b_mpm_displacement_retention" not in cfg
