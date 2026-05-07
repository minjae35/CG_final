"""Surface-aware placement (PRD 3.2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation as SciRotation


def load_gaussian_xyz_opacity(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)
    pos = np.stack([x, y, z], axis=1)
    op = np.asarray(v["opacity"], dtype=np.float64) if "opacity" in v.dtype.names else np.ones(len(x))
    return pos, op


def find_surface_height(
    scene_gaussians: np.ndarray,
    x: float,
    y: float,
    search_radius: float = 0.1,
    opacity_threshold: float = 0.5,
) -> float:
    """
    scene_gaussians: (N, 4) optional opacity in col4, or (N,3) xyz only.
    """
    if scene_gaussians.shape[1] == 4:
        xyz, op = scene_gaussians[:, :3], scene_gaussians[:, 3]
    else:
        xyz, op = scene_gaussians[:, :3], np.ones(len(scene_gaussians))
    dx = xyz[:, 0] - x
    dy = xyz[:, 1] - y
    mask = (dx * dx + dy * dy) <= search_radius**2
    if not np.any(mask):
        return float(np.median(xyz[:, 2]))
    sub = xyz[mask]
    sub_op = op[mask]
    hi = sub[sub_op > opacity_threshold]
    if len(hi) == 0:
        return float(np.max(sub[:, 2]))
    return float(np.max(hi[:, 2]))


def find_surface_height_from_ply(
    ply_path: str,
    x: float,
    y: float,
    search_radius: float = 0.1,
    opacity_threshold: float = 0.5,
) -> float:
    pos, op = load_gaussian_xyz_opacity(ply_path)
    g = np.concatenate([pos, op[:, None]], axis=1)
    return find_surface_height(g, x, y, search_radius, opacity_threshold)


def find_camera_visible_surface_point(
    ply_path: str | Path,
    cameras_json: str | Path,
    camera_index: int = 0,
    target_u_frac: float = 0.58,
    target_v_frac: float = 0.68,
    nearest_count: int = 2500,
    opacity_threshold: float = 0.0,
    z_quantile: float = 75.0,
) -> tuple[float, float, float]:
    """Pick a robust visible support point near a target pixel in a training camera.

    The 3DGS camera JSON stores camera center and rotation.  For this repository's
    trained scenes, ``rotation.T @ (point - center)`` projects into camera space.
    """
    with open(cameras_json, encoding="utf-8") as f:
        cam = json.load(f)[camera_index]

    pos, op = load_gaussian_xyz_opacity(str(ply_path))
    center = np.asarray(cam["position"], dtype=np.float64)
    rot = np.asarray(cam["rotation"], dtype=np.float64)
    xyz_cam = (rot.T @ (pos - center).T).T
    depth = xyz_cam[:, 2]
    w, h = float(cam["width"]), float(cam["height"])
    u = float(cam["fx"]) * xyz_cam[:, 0] / np.maximum(depth, 1e-8) + w * 0.5
    v = float(cam["fy"]) * xyz_cam[:, 1] / np.maximum(depth, 1e-8) + h * 0.5

    visible = (
        (depth > 0)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
        & (op > opacity_threshold)
    )
    if not np.any(visible):
        raise ValueError(f"No visible Gaussians found for camera {camera_index}: {cameras_json}")

    tu, tv = w * target_u_frac, h * target_v_frac
    dist2 = (u - tu) ** 2 + (v - tv) ** 2
    idx = np.where(visible)[0]
    idx = idx[np.argsort(dist2[idx])[:nearest_count]]
    pts = pos[idx]
    return (
        float(np.median(pts[:, 0])),
        float(np.median(pts[:, 1])),
        float(np.percentile(pts[:, 2], z_quantile)),
    )


def refine_floor_contact_y(
    ply_path: str | Path,
    px: float,
    pz: float,
    *,
    floor_y_hint: float,
    xz_radius: float = 0.14,
    y_span_smaller_than_hint: float = 0.12,
    y_span_larger_than_hint: float = 0.65,
    opacity_threshold: float = 0.02,
    contact_percentile: float = 93.0,
    min_points: int = 50,
) -> float | None:
    """Robust floor height (+Y-down scenes) near ``(px, pz)`` from dense scene Gaussians.

    :func:`find_camera_visible_surface_point` can return a ``y`` that mixes rug
    support with slightly nearer clutter along the ray.  Here we take Gaussians
    in an XZ disk around the placement footprint and use a high percentile of
    ``y`` (support is toward **larger** ``y`` when +Y points down).
    """
    pos, op = load_gaussian_xyz_opacity(str(ply_path))
    dx = pos[:, 0] - float(px)
    dz = pos[:, 2] - float(pz)
    disk = (dx * dx + dz * dz) <= float(xz_radius) ** 2
    # True floor is often at *larger* y than a frustum-median hint; keep a wide
    # upward (+y) span and a small downward span for floating outliers.
    band = (pos[:, 1] >= float(floor_y_hint) - float(y_span_smaller_than_hint)) & (
        pos[:, 1] <= float(floor_y_hint) + float(y_span_larger_than_hint)
    )
    m = disk & band & (op > float(opacity_threshold))
    if int(m.sum()) < int(min_points):
        return None
    ys = pos[m, 1]
    return float(np.percentile(ys, float(contact_percentile)))


def estimate_floor_slopes_dy_dx_dz(
    ply_path: str | Path,
    px: float,
    pz: float,
    floor_y_ref: float,
    *,
    radius: float = 0.16,
    y_band: float = 0.10,
    opacity_threshold: float = 0.0,
    max_slope: float = 0.45,
) -> tuple[float, float]:
    """Estimate local floor plane ``y ≈ c + a*(x-px) + b*(z-pz)`` near placement.

    Scene convention matches the rest of this repo: **+Y points down**; the rug
    is at large ``y``. ``floor_y_ref`` should be the contact height from
    :func:`find_camera_visible_surface_point`.

    Returns ``(a, b)`` = ``(∂y/∂x, ∂y/∂z)`` in scene units. For a perfectly level
    floor, both are ~0. Small slopes let us rotate the inserted object so its
    base matches the rug instead of staying axis-aligned to XZ.
    """
    pos, op = load_gaussian_xyz_opacity(str(ply_path))
    dx = pos[:, 0] - px
    dz = pos[:, 2] - pz
    horiz = dx * dx + dz * dz
    mask = (
        (horiz <= radius * radius)
        & (pos[:, 1] >= floor_y_ref - y_band)
        & (pos[:, 1] <= floor_y_ref + 0.04)
        & (op > opacity_threshold)
    )
    if int(mask.sum()) < 80:
        return 0.0, 0.0
    xs = pos[mask, 0] - px
    zs = pos[mask, 2] - pz
    ys = pos[mask, 1]
    w = np.asarray(op[mask], dtype=np.float64)
    w = np.clip(w, 1e-6, None)
    # Weight nearer points more so the fit tracks the rug under the object.
    w = w / (1.0 + horiz[mask] / (0.25 * radius * radius + 1e-8))
    X = np.stack([np.ones_like(xs), xs, zs], axis=1)
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = ys * sqrt_w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    a = float(beta[1])
    b = float(beta[2])
    a = float(np.clip(a, -max_slope, max_slope))
    b = float(np.clip(b, -max_slope, max_slope))
    return a, b


def floor_downward_normal_from_slopes(a: float, b: float) -> np.ndarray:
    """Unit vector into the floor (+Y side of the tangent plane), from (a,b)=∂y/∂x,z."""
    v = np.array([-a, 1.0, -b], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return v / n


def camera_right_horizontal_world(cameras_json: str | Path, camera_index: int = 0) -> np.ndarray:
    """World-space axis horizontal to the ground, pointing camera-right (+X in image).

    Uses the same ``rotation`` convention as :func:`find_camera_visible_surface_point`.
    """
    with open(cameras_json, encoding="utf-8") as f:
        cam = json.load(f)[camera_index]
    r = np.asarray(cam["rotation"], dtype=np.float64)
    r_cam = r[:, 0]
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = r_cam - float(np.dot(r_cam, ey)) * ey
    nrm = np.linalg.norm(u)
    if nrm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return u / nrm


def rotate_unit_vector_around_axis_deg(v: np.ndarray, unit_axis: np.ndarray, deg: float) -> np.ndarray:
    """Right-hand rule rotation of unit vector ``v`` about ``unit_axis``."""
    if abs(deg) < 1e-9:
        return np.asarray(v, dtype=np.float64) / (np.linalg.norm(v) + 1e-12)
    u = np.asarray(unit_axis, dtype=np.float64).reshape(3)
    u = u / (np.linalg.norm(u) + 1e-12)
    rot = SciRotation.from_rotvec(np.deg2rad(float(deg)) * u)
    out = rot.apply(np.asarray(v, dtype=np.float64).reshape(3))
    return out / (np.linalg.norm(out) + 1e-12)
