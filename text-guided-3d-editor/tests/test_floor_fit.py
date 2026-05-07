from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from placement.surface_aware import (
    estimate_floor_slopes_dy_dx_dz,
    floor_downward_normal_from_slopes,
    rotate_unit_vector_around_axis_deg,
)


def test_floor_downward_normal_flat() -> None:
    n = floor_downward_normal_from_slopes(0.0, 0.0)
    np.testing.assert_allclose(n, [0.0, 1.0, 0.0], atol=1e-7)


def test_rotate_unit_vector_preserves_length() -> None:
    v = np.array([0.0, 1.0, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    w = rotate_unit_vector_around_axis_deg(v, axis, 12.0)
    np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-7)
    assert w[0] < 1e-6


def test_estimate_floor_slopes_recovery(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    a_true, b_true = 0.08, -0.05
    px, pz = 0.2, -0.1
    n = 500
    xs = rng.uniform(-0.18, 0.18, n) + px
    zs = rng.uniform(-0.18, 0.18, n) + pz
    ys = (
        2.0
        + a_true * (xs - px)
        + b_true * (zs - pz)
        + rng.normal(0.0, 0.002, n)
    )
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("opacity", "f8")]
    v = np.zeros(n, dtype=dtype)
    v["x"] = xs
    v["y"] = ys
    v["z"] = zs
    v["opacity"] = 1.0
    path = tmp_path / "floor.ply"
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))
    floor_y = float(np.median(ys))
    a, b = estimate_floor_slopes_dy_dx_dz(
        path, px, pz, floor_y_ref=floor_y, radius=0.35, y_band=0.08
    )
    assert abs(a - a_true) < 0.025
    assert abs(b - b_true) < 0.025
