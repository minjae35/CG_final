#!/usr/bin/env python3
"""Compare ``uv_debug_points.ply`` (raw surface RGB) vs ``object_mesh_gaussians.ply`` (SH DC + scales).

Run from ``text-guided-3d-editor/`` (or pass absolute paths):

  export PYTHONPATH=src
  python scripts/diagnose_uv_debug_vs_gaussian_ply.py \\
    --debug /tmp/uv_debug_points.ply \\
    --gaussian output/generated_objects/sds_run/object_mesh_gaussians.ply \\
    --tiny-scale-out output/generated_objects/sds_run/object_mesh_gaussians_debug_tiny_scale.ply

Interpretation:
  * If per-point RGB (decoded from ``f_dc_*``) matches the debug PLY within ~1–2 levels,
    UV sampling is fine and any viewer smear is likely **splat scale / alpha / rasterization**.
  * If RGB differs a lot row-wise, check UV path or that both files are from the **same** run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# Same as generation.mesh_to_gaussian.C0
C0 = 0.28209479177387814


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """rgb uint8 (N,3) -> SH DC (N,3) float32."""
    return (rgb.astype(np.float32) / 255.0 - 0.5) / C0


def _sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """Inverse of mesh_to_gaussian ``_rgb_to_sh``."""
    x = f_dc.astype(np.float64) * C0 + 0.5
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.float32)


def _load_debug_rgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = list(v.dtype.names or ())
    xyz = np.column_stack([v["x"].astype(np.float64), v["y"].astype(np.float64), v["z"].astype(np.float64)])
    r, g, b = None, None, None
    for cr, cg, cb in (("red", "green", "blue"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if cr in names and cg in names and cb in names:
            r, g, b = v[cr], v[cg], v[cb]
            break
    if r is None:
        raise ValueError(f"{path}: need uchar columns red/green/blue (debug export). Found: {names}")
    rgb = np.column_stack([r.astype(np.float64), g.astype(np.float64), b.astype(np.float64)])
    return xyz, rgb


def _load_gaussian_ply(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = list(v.dtype.names or ())
    need = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")
    for k in need:
        if k not in names:
            raise ValueError(f"{path}: missing vertex field {k!r}. Found: {names}")
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    f_dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float64)
    extra: dict[str, np.ndarray] = {}
    for k in ("scale_0", "scale_1", "scale_2", "opacity", "nx", "ny", "nz", "rot_0", "rot_1", "rot_2", "rot_3"):
        if k in names:
            extra[k] = np.asarray(v[k])
    return xyz, f_dc, extra


def _color_stats(rgb: np.ndarray, label: str) -> None:
    """rgb (N,3) float or int."""
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    n = len(rgb)
    print(f"\n--- {label} (N={n}) ---")
    print(f"  RGB min:    ({r.min():.1f}, {g.min():.1f}, {b.min():.1f})")
    print(f"  RGB mean:   ({r.mean():.1f}, {g.mean():.1f}, {b.mean():.1f})")
    print(f"  RGB max:    ({r.max():.1f}, {g.max():.1f}, {b.max():.1f})")

    dark = (r < 45) & (g < 45) & (b < 45)
    orange = (r > 120) & (g > 60) & (g < 220) & (b < 110) & ~dark
    yellow = (r > 170) & (g > 140) & (b < 140) & ~dark

    def pct(mask: np.ndarray) -> float:
        return 100.0 * float(np.count_nonzero(mask)) / max(n, 1)

    print(f"  very dark (R,G,B<45):     {np.count_nonzero(dark):6d}  ({pct(dark):.2f}%)")
    print(f"  orange-ish (heuristic):   {np.count_nonzero(orange):6d}  ({pct(orange):.2f}%)")
    print(f"  yellow-ish (heuristic):   {np.count_nonzero(yellow):6d}  ({pct(yellow):.2f}%)")


def _compare_aligned(
    xyz_d: np.ndarray,
    rgb_d: np.ndarray,
    xyz_g: np.ndarray,
    rgb_g_from_sh: np.ndarray,
) -> None:
    n = min(len(xyz_d), len(xyz_g))
    if len(xyz_d) != len(xyz_g):
        print(
            f"\n[WARN] vertex count mismatch: debug={len(xyz_d)} gaussian={len(xyz_g)} — "
            f"comparing first {n} rows only."
        )
    d_xyz = np.abs(xyz_d[:n] - xyz_g[:n])
    d_rgb = np.abs(rgb_d[:n] - rgb_g_from_sh[:n])
    print("\n--- aligned comparison (same row index, first N rows) ---")
    print(f"  |Δxyz| max:  {d_xyz.max(axis=0)}")
    print(f"  |Δxyz| mean: {d_xyz.mean(axis=0)}")
    print(f"  |ΔRGB| max:  {d_rgb.max(axis=0)}")
    print(f"  |ΔRGB| mean:{d_rgb.mean(axis=0)}")
    med = np.median(np.linalg.norm(d_rgb, axis=1))
    p95 = float(np.percentile(np.linalg.norm(d_rgb, axis=1), 95))
    print(f"  L2 RGB error median: {med:.3f}  p95: {p95:.3f} (0–255 scale)")

    mean_rgb_err = float(d_rgb.mean())
    max_rgb_err = float(d_rgb.max())
    xyz_ok = float(d_xyz.mean()) < 1e-3
    rgb_ok = max_rgb_err <= 2.5 and mean_rgb_err < 1.5
    if xyz_ok and rgb_ok:
        verdict = (
            "B) UV/debug RGB and ``f_dc_*`` round-trip **match** at splat centres (within float noise).\n"
            "   → Viewer smear is likely **Gaussian scale / opacity / alpha blending**, not UV lookup. "
            "Open the ``--tiny-scale-out`` PLY in SuperSplat to confirm."
        )
    elif rgb_ok and not xyz_ok:
        verdict = (
            "**Position mismatch** — files may be from different runs or reordered; "
            "regenerate ``uv_debug_points.ply`` with ``MESH_GAUSSIAN_DEBUG_UV_POINTS`` in the same "
            "``regen-mesh-assets`` / ``mode-a`` invocation as ``object_mesh_gaussians.ply``."
        )
    else:
        verdict = (
            "A) **RGB mismatch** between debug PLY and SH-decoded Gaussian colours.\n"
            "   → UV sampling / ``f_dc`` encoding / or mismatched file pair — fix before blaming splat size."
        )
    print("\n*** Verdict ***")
    print(verdict)


def _scale_report(extra: dict[str, np.ndarray]) -> None:
    keys = ("scale_0", "scale_1", "scale_2")
    if not all(k in extra for k in keys):
        print("\n--- Gaussian scales: not all of scale_0..2 present ---")
        return
    s = np.column_stack([extra[k].astype(np.float64) for k in keys])
    eff = np.exp(s)
    print("\n--- object_mesh_gaussians scale_* (log-space; viewer uses exp) ---")
    print(f"  scale_0..2 min:  {s.min(axis=0)}")
    print(f"  scale_0..2 mean: {s.mean(axis=0)}")
    print(f"  scale_0..2 max:  {s.max(axis=0)}")
    print(f"  exp(scale) mean (world-ish radius scale): {eff.mean(axis=0)}")
    print(f"  exp(scale) max:  {eff.max(axis=0)}")
    if float(s.mean()) > -4.0:
        print(
            "  [note] mean log-scale > -4 → relatively **large** isotropic Gaussians → more neighbour blending."
        )


def _write_tiny_scale_gaussian_ply(
    gaussian_path: Path,
    debug_path: Path,
    out_path: Path,
    *,
    scale_logit: float,
) -> None:
    """Copy Gaussian PLY; replace ``f_dc_*`` from debug RGB (first N rows); set isotropic ``scale_*`` and identity ``rot_*``."""
    ply = PlyData.read(str(gaussian_path))
    el = ply["vertex"]
    v = np.array(el.data, copy=True)
    names = list(v.dtype.names or ())
    _xyz_d, rgb_d = _load_debug_rgb_ply(debug_path)
    n = min(len(v), len(rgb_d))
    if len(v) != len(rgb_d):
        print(
            f"[WARN] tiny-scale PLY: only first N={n} splats get colours from debug "
            f"(gaussian={len(v)} debug={len(rgb_d)}); tail keeps original f_dc."
        )
    orig_dc = np.column_stack([v["f_dc_0"].copy(), v["f_dc_1"].copy(), v["f_dc_2"].copy()])
    sh = _rgb_to_sh_dc(rgb_d[:n].astype(np.uint8))
    for j in range(3):
        v[f"f_dc_{j}"][:n] = sh[:, j]
        v[f"f_dc_{j}"][n:] = orig_dc[n:, j]
    for j in range(3):
        if f"scale_{j}" in names:
            v[f"scale_{j}"][:] = float(scale_logit)
    for j, val in enumerate((1.0, 0.0, 0.0, 0.0)):
        rk = f"rot_{j}"
        if rk in names:
            v[rk][:] = val
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(out_path))
    print(f"\nWrote tiny-scale debug Gaussian PLY: {out_path.resolve()}")


def main() -> None:
    _ensure_src_on_path()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debug", type=Path, required=True, help="uv_debug_points.ply (x,y,z + red,green,blue)")
    p.add_argument("--gaussian", type=Path, required=True, help="object_mesh_gaussians.ply")
    p.add_argument(
        "--tiny-scale-out",
        type=Path,
        default=None,
        help="Optional: write Gaussian PLY with debug RGB as f_dc and scale_* = -6, identity rot.",
    )
    p.add_argument(
        "--tiny-scale-logit",
        type=float,
        default=-6.0,
        help="Isotropic log-scale for --tiny-scale-out (default -6.0).",
    )
    args = p.parse_args()

    xyz_d, rgb_d = _load_debug_rgb_ply(args.debug)
    xyz_g, f_dc, extra = _load_gaussian_ply(args.gaussian)
    rgb_from_sh = _sh_dc_to_rgb(f_dc)

    _color_stats(rgb_d, "debug PLY (surface RGB)")
    _color_stats(rgb_from_sh, "Gaussian PLY (RGB decoded from f_dc_0..2)")
    _compare_aligned(xyz_d, rgb_d, xyz_g, rgb_from_sh)
    _scale_report(extra)

    if args.tiny_scale_out is not None:
        _write_tiny_scale_gaussian_ply(
            args.gaussian,
            args.debug,
            args.tiny_scale_out,
            scale_logit=float(args.tiny_scale_logit),
        )


if __name__ == "__main__":
    main()
