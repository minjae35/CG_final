"""PhysGaussian-style material presets (PRD 4.1)."""
from __future__ import annotations

from typing import Any

MATERIAL_PRESETS: dict[str, dict[str, Any]] = {
    "jelly": {
        "material": "jelly",
        "E": 1.2e5,
        "nu": 0.4,
        "density": 500,
        "grid_v_damping_scale": 0.999,
    },
    # Same MPM branch as jelly (hyperelastic FCR, material id 0): stiffer than jelly
    # but softer than ultra-rigid rubber so COM-visible wobble + modest shape modes show.
    "rubber": {
        "material": "jelly",
        "E": 2.4e6,
        "nu": 0.34,
        "density": 1050,
        "grid_v_damping_scale": 0.99976,
    },
    "metal": {
        "material": "metal",
        "E": 1e7,
        "nu": 0.30,
        "density": 1200,
    },
    "rigid": {
        "material": "metal",
        "E": 5e5,
        "nu": 0.35,
        "density": 1200,
        "yield_stress": 1e5,
        "grid_v_damping_scale": 0.9999,
    },
    "sand": {
        "material": "sand",
        "E": 1e4,
        "nu": 0.2,
        "density": 1500,
    },
    "foam": {
        "material": "foam",
        "E": 2e5,
        "nu": 0.25,
        "density": 250,
        "yield_stress": 3e4,
        "grid_v_damping_scale": 0.9988,
    },
}


def get_preset(name: str) -> dict[str, Any]:
    if name not in MATERIAL_PRESETS:
        raise KeyError(f"Unknown material preset: {name}")
    return dict(MATERIAL_PRESETS[name])
