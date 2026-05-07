from __future__ import annotations

import numpy as np

from physics.kinematic_wobble import object_displacement_kinematic


def test_kinematic_displacement_mean_free() -> None:
    y = np.linspace(1.0, 2.0, 500, dtype=np.float64)
    d = object_displacement_kinematic(
        y,
        frame_index=3,
        frame_dt=0.01,
        amp=0.02,
        freq_hz=1.0,
        height_gamma=1.5,
        bottom_pin=0.25,
    )
    assert d.shape == (500, 3)
    assert np.allclose(d.mean(axis=0), 0.0, atol=1e-9)


def test_kinematic_nearly_flat_y_with_xyz_still_nonzero() -> None:
    """Degenerate bbox in +Y: phase from xz must not yield all-zero displacement."""
    rng = np.random.default_rng(42)
    y = 1.0 + rng.random(400) * 1e-7
    x = rng.standard_normal(400)
    z = rng.standard_normal(400)
    xyz = np.stack([x, y, z], axis=1)
    d = object_displacement_kinematic(
        y,
        xyz_world=xyz,
        frame_index=10,
        frame_dt=0.01,
        amp=0.02,
        freq_hz=1.0,
        height_gamma=1.5,
        bottom_pin=0.25,
    )
    assert d.shape == (400, 3)
    assert np.linalg.norm(d) > 1e-5
    assert np.allclose(d.mean(axis=0), 0.0, atol=1e-8)


def test_kinematic_feet_pinned_smaller_motion() -> None:
    """Larger y (feet) should get smaller displacement magnitude on average than head."""
    y = np.concatenate([np.full(200, 2.5), np.full(200, 1.0)])
    d = object_displacement_kinematic(
        y,
        frame_index=10,
        frame_dt=0.01,
        amp=0.05,
        freq_hz=1.0,
        height_gamma=2.0,
        bottom_pin=0.3,
    )
    m_feet = np.linalg.norm(d[:200], axis=1).mean()
    m_head = np.linalg.norm(d[200:], axis=1).mean()
    assert m_head > m_feet
