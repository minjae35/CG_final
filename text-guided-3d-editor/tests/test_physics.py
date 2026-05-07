from __future__ import annotations

from pathlib import Path

import numpy as np
import json

from physics.config_generator import generate_phys_config
from physics.material_presets import get_preset


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
