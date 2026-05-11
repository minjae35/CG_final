from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass(frozen=True)
class RegionStats:
    name: str
    n: int
    mean: float
    p50: float
    p90: float
    p99: float
    max: float


def _percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else float("nan")


def _stats(name: str, d: np.ndarray) -> RegionStats:
    if d.size == 0:
        return RegionStats(name=name, n=0, mean=float("nan"), p50=float("nan"), p90=float("nan"), p99=float("nan"), max=float("nan"))
    return RegionStats(
        name=name,
        n=int(d.size),
        mean=float(d.mean()),
        p50=_percentile(d, 50),
        p90=_percentile(d, 90),
        p99=_percentile(d, 99),
        max=float(d.max()),
    )


def _orthonormal_basis_from_up(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (front, left, up) basis.
    - up is provided (unit)
    - front is the principal axis projected onto horizontal plane (orthogonal to up)
    - left completes right-handed basis: left = up x front
    """
    up = up.astype(np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)

    # PCA on centered points
    c = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
    # SVD gives principal directions in Vt rows
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    p0 = vt[0]
    # project onto horizontal plane
    front = p0 - up * float(np.dot(p0, up))
    n = np.linalg.norm(front)
    if n < 1e-8:
        # fallback: pick any axis not parallel to up
        a = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, up))) > 0.9:
            a = np.array([0.0, 0.0, 1.0])
        front = a - up * float(np.dot(a, up))
        n = np.linalg.norm(front)
    front = front / (n + 1e-12)
    left = np.cross(up, front)
    left = left / (np.linalg.norm(left) + 1e-12)
    return front.astype(np.float32), left.astype(np.float32), up.astype(np.float32)


def _color_map(d: np.ndarray) -> np.ndarray:
    """
    Blue (low) -> Green -> Yellow -> Red (high).
    Uses robust scaling based on p95 to avoid a single outlier dominating.
    """
    if d.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    hi = float(np.percentile(d, 95))
    hi = max(hi, 1e-8)
    t = np.clip(d / hi, 0.0, 1.0)

    rgb = np.zeros((d.size, 3), dtype=np.float32)
    # piecewise: [0,0.5]: blue->green, [0.5,0.75]: green->yellow, [0.75,1]: yellow->red
    a = t <= 0.5
    b = (t > 0.5) & (t <= 0.75)
    c = t > 0.75
    # blue->green
    tt = (t[a] / 0.5)[:, None]
    rgb[a] = (1 - tt) * np.array([0, 0, 1]) + tt * np.array([0, 1, 0])
    # green->yellow
    tt = ((t[b] - 0.5) / 0.25)[:, None]
    rgb[b] = (1 - tt) * np.array([0, 1, 0]) + tt * np.array([1, 1, 0])
    # yellow->red
    tt = ((t[c] - 0.75) / 0.25)[:, None]
    rgb[c] = (1 - tt) * np.array([1, 1, 0]) + tt * np.array([1, 0, 0])
    return (rgb * 255.0).clip(0, 255).astype(np.uint8)


def analyze_and_write(
    *,
    out_dir: Path,
    model_ply: Path,
    up_axis: np.ndarray,
) -> Path:
    debug_dir = out_dir / "debug"
    first = np.load(debug_dir / "selected_means3d_first.npy").astype(np.float32, copy=False)
    last = np.load(debug_dir / "selected_means3d_last.npy").astype(np.float32, copy=False)
    if first.shape != last.shape:
        raise ValueError(f"shape mismatch: first={first.shape}, last={last.shape}")

    disp = last - first
    mag = np.linalg.norm(disp, axis=1)

    front, left, up = _orthonormal_basis_from_up(first, up_axis.astype(np.float32))
    # World-down displacement component. Positive means moving "down" (sinking).
    down = (-up).astype(np.float32)
    down_disp = (disp @ down).astype(np.float32)
    down_sink = np.maximum(down_disp, 0.0)
    # signed coords in this basis
    c = first - first.mean(axis=0, keepdims=True)
    s_front = c @ front
    s_left = c @ left
    s_up = c @ up

    # region splits
    med_front = float(np.median(s_front))
    med_left = float(np.median(s_left))
    med_up = float(np.median(s_up))

    # seat/backrest heuristic: backrest = (upper half) & (rear half)
    rear = s_front < med_front
    front_half = ~rear
    upper = s_up > med_up
    lower = ~upper
    left_side = s_left > med_left
    right_side = ~left_side

    # arms heuristic: extreme left/right (top 20% by |s_left|)
    abs_lr = np.abs(s_left)
    arm_thr = float(np.percentile(abs_lr, 80))
    arms = abs_lr >= arm_thr
    left_arm = arms & left_side
    right_arm = arms & right_side

    # seat heuristic: lower & near center in front/back (middle 60% of s_front)
    f_lo, f_hi = np.percentile(s_front, [20, 80])
    seat = lower & (s_front >= f_lo) & (s_front <= f_hi)
    backrest = upper & rear

    stats = [
        _stats("all_selected", mag),
        _stats("front_half", mag[front_half]),
        _stats("rear_half", mag[rear]),
        _stats("upper_half", mag[upper]),
        _stats("lower_half", mag[lower]),
        _stats("left_half", mag[left_side]),
        _stats("right_half", mag[right_side]),
        _stats("left_arm", mag[left_arm]),
        _stats("right_arm", mag[right_arm]),
        _stats("seat", mag[seat]),
        _stats("backrest", mag[backrest]),
    ]

    out_json = debug_dir / "displacement_region_stats.json"
    out_json.write_text(json.dumps([s.__dict__ for s in stats], indent=2), encoding="utf-8")

    # Downward (sinking) component stats by region.
    down_stats = [
        _stats("all_selected_down", down_sink),
        _stats("front_half_down", down_sink[front_half]),
        _stats("rear_half_down", down_sink[rear]),
        _stats("upper_half_down", down_sink[upper]),
        _stats("lower_half_down", down_sink[lower]),
        _stats("left_arm_down", down_sink[left_arm]),
        _stats("right_arm_down", down_sink[right_arm]),
        _stats("seat_down", down_sink[seat]),
        _stats("backrest_down", down_sink[backrest]),
    ]
    (debug_dir / "displacement_down_region_stats.json").write_text(
        json.dumps([s.__dict__ for s in down_stats], indent=2), encoding="utf-8"
    )

    # Sideways-vs-down check (are some regions moving mostly laterally?).
    lateral = disp - (down_disp[:, None] * down[None, :])
    lateral_mag = np.linalg.norm(lateral, axis=1).astype(np.float32)
    eps = 1e-8
    ratio_lat_over_down = (lateral_mag / (down_sink + eps)).astype(np.float32)

    def _ratio_summary(name: str, mask: np.ndarray) -> dict:
        r = ratio_lat_over_down[mask]
        return {
            "name": name,
            "n": int(mask.sum()),
            "p50": float(np.percentile(r, 50)) if r.size else float("nan"),
            "p90": float(np.percentile(r, 90)) if r.size else float("nan"),
            "p99": float(np.percentile(r, 99)) if r.size else float("nan"),
            "mean": float(r.mean()) if r.size else float("nan"),
        }

    (debug_dir / "lateral_over_down_ratio.json").write_text(
        json.dumps(
            [
                _ratio_summary("all_selected", np.ones_like(down_sink, dtype=bool)),
                _ratio_summary("upper_half", upper),
                _ratio_summary("lower_half", lower),
                _ratio_summary("seat", seat),
                _ratio_summary("backrest", backrest),
                _ratio_summary("front_half", front_half),
                _ratio_summary("rear_half", rear),
                _ratio_summary("left_arm", left_arm),
                _ratio_summary("right_arm", right_arm),
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # displacement-colored PLY for selected points (color by mag)
    colors = _color_map(mag)
    ply = PlyData.read(str(model_ply))
    v = ply["vertex"]
    pos = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32, copy=False)
    # selected indices are global indices into model ply
    selected = np.load(out_dir / "selected_indices.npy").astype(np.int64, copy=False)
    pts = pos[selected]

    verts = np.empty(
        pts.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    out_ply = debug_dir / "displacement_colored_selected_points.ply"
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(out_ply))

    # sinking-colored PLY for selected points (color by down_sink)
    colors_down = _color_map(down_sink)
    verts2 = np.empty(
        pts.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts2["x"], verts2["y"], verts2["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    verts2["red"], verts2["green"], verts2["blue"] = (
        colors_down[:, 0],
        colors_down[:, 1],
        colors_down[:, 2],
    )
    out_ply_down = debug_dir / "displacement_down_colored_selected_points.ply"
    PlyData([PlyElement.describe(verts2, "vertex")], text=False).write(str(out_ply_down))
    return out_ply

