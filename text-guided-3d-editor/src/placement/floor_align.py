"""Dominant floor plane -> rotation for PhysGaussian (PRD 4.4)."""
from __future__ import annotations

import numpy as np
from plyfile import PlyData


def estimate_floor_rotation(ply_path: str, max_points: int = 50000) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Returns (axis, angle_degrees, translation) to apply so floor approx z=0.
    Simplified: use lowest 10% z points, fit plane n·x+d=0, rotate so n -> +z.
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
    z_thresh = np.quantile(xyz[:, 2], 0.1)
    floor_pts = xyz[xyz[:, 2] <= z_thresh + 0.02]
    if len(floor_pts) < 100:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(3)

    c = floor_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(floor_pts - c)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-8)
    # target +z
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.dot(n, z_axis) < 0:
        n = -n
    cross = np.cross(n, z_axis)
    s = np.linalg.norm(cross)
    if s < 1e-6:
        return z_axis, 0.0, np.zeros(3)
    axis = cross / s
    angle = np.arccos(np.clip(np.dot(n, z_axis), -1.0, 1.0))
    return axis, float(np.degrees(angle)), -c
