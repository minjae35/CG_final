"""Mode-b in-place jelly helpers: support height from selection, COM drift bounds."""
from __future__ import annotations

import numpy as np


def estimate_mode_b_support_contact_y(
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    world_down_axis: int = 1,
    contact_percentile: float = 92.0,
) -> float:
    """Return world-space Y of the object–support contact band (+Y-down scenes).

    Gravity pulls along +Y. The side that meets a table/floor first sits at **larger**
    ``y`` than the upper surface, so use a **high** percentile of selected ``y``.
    """
    if len(indices) == 0:
        raise ValueError("empty indices for support estimation")
    ys = np.asarray(positions[indices, world_down_axis], dtype=np.float64)
    p = float(np.clip(contact_percentile, 0.0, 100.0))
    return float(np.percentile(ys, p))


def apply_com_pin_mpm(
    positions_mpm: np.ndarray,
    com_init: np.ndarray,
    *,
    vertical_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract rigid COM drift in MPM coordinates (vectorized, for tests)."""
    com = np.mean(positions_mpm, axis=0)
    delta = com - com_init
    if vertical_only:
        delta = np.array([0.0, float(delta[1]), 0.0], dtype=np.float64)
    return positions_mpm - delta, delta


def max_vertical_com_drift_after_pinning(
    *,
    initial_y: float,
    n_steps: int,
    gravity_step: float = 0.01,
    noise: float = 0.002,
    seed: int = 0,
    vertical_only: bool = False,
) -> tuple[float, float]:
    """Toy integration: downward bias + noise each step, then COM pin (used by tests).

    Returns (max pre-pin |Δy| before correction in a step, final |Δy| after last pin).
    """
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((40, 3)).astype(np.float64) * 0.02
    pts -= pts.mean(axis=0)
    pts[:, 1] += float(initial_y)
    com0 = pts.mean(axis=0)
    max_pre = 0.0
    for _ in range(n_steps):
        pts[:, 1] += gravity_step
        pts += rng.standard_normal(pts.shape) * noise
        pre_com = pts.mean(axis=0)
        max_pre = max(max_pre, abs(float(pre_com[1] - com0[1])))
        pts, _ = apply_com_pin_mpm(pts, com0, vertical_only=vertical_only)
    final_dy = abs(float(pts.mean(axis=0)[1] - com0[1]))
    return max_pre, final_dy
