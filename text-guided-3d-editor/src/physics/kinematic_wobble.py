"""Shape-preserving kinematic wobble (no MPM): lateral + squash driven by height weights."""
from __future__ import annotations

import numpy as np


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def object_displacement_kinematic(
    y_world: np.ndarray,
    *,
    frame_index: int,
    frame_dt: float,
    amp: float,
    freq_hz: float,
    height_gamma: float,
    bottom_pin: float,
    xyz_world: np.ndarray | None = None,
) -> np.ndarray:
    """Return (N,3) displacement in world metres for object Gaussians.

    ``y_world`` is scene +Y (down).  Larger ``y`` = closer to floor/feet.
    Motion is strongest toward smaller ``y`` (head) and fades in a ``bottom_pin``
    band at the feet.  Displacement is **mean-free** so the object COM does not drift.

    ``xyz_world`` (optional, shape ``(N,3)``) breaks **parallel** displacement
    ``w_i * V`` across particles: after ``D -= D.mean`` that pattern cancels to
    ``(w_i - mean(w)) * V``, which is ~0 when ``w`` is nearly constant.  Per-particle
    phase from height ``t`` and normalized ``x,z`` fixes the render looking static.
    """
    y = np.asarray(y_world, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    y_max = float(y.max())
    y_min = float(y.min())
    span = max(y_max - y_min, 1e-8)
    height_degenerate = span < 1e-5

    # t = 0 at feet (+Y down: y_max), 1 toward head (y_min).  Foot pin uses y span;
    # if height is degenerate, derive t from xz and skip the pin band.
    if not height_degenerate:
        t = (y_max - y) / span
        t = np.clip(t, 0.0, 1.0)
        pin = float(np.clip(bottom_pin, 0.0, 0.95))
        edge = y_max - pin * span
        u = (y - edge) / (pin * span + 1e-8)
        pin_mask = 1.0 - _smoothstep01(np.clip(u, 0.0, 1.0))
    elif xyz_world is not None:
        X = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
        ax, az = X[:, 0], X[:, 2]
        sx = max(float(ax.max() - ax.min()), 1e-8)
        sz = max(float(az.max() - az.min()), 1e-8)
        t = 0.5 * (ax - float(ax.min())) / sx + 0.5 * (az - float(az.min())) / sz
        t = np.clip(t, 0.0, 1.0)
        pin_mask = np.ones_like(t, dtype=np.float64)
    else:
        t = np.linspace(0.0, 1.0, y.size, endpoint=False, dtype=np.float64)
        pin_mask = np.ones_like(t, dtype=np.float64)

    w_h = np.power(t, float(max(height_gamma, 0.05)))
    w = w_h * pin_mask

    t_sim = float(frame_index) * float(frame_dt)
    phase = 2.0 * np.pi * float(freq_hz) * t_sim
    # Per-particle phase so D is not (w_i * same_vector); COM subtraction then
    # does not wipe almost all motion when w has low variance.
    if xyz_world is not None:
        X = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
        x, z = X[:, 0], X[:, 2]
        xs = (x - np.mean(x)) / (np.std(x) + 1e-8)
        zs = (z - np.mean(z)) / (np.std(z) + 1e-8)
        phase_i = phase + 2.0 * np.pi * (0.55 * t + 0.12 * xs + 0.09 * zs)
    else:
        phase_i = phase + 2.0 * np.pi * (1.22 * t)
    lateral = float(amp) * np.sin(phase_i) * w
    squash = float(amp) * 0.14 * np.sin(2.0 * phase_i + 0.9) * w
    sway_z = float(amp) * 0.09 * np.cos(1.4 * phase_i + 0.2) * w
    D = np.stack([lateral, squash, sway_z], axis=1)
    D -= D.mean(axis=0, keepdims=True)
    return D.astype(np.float64)
