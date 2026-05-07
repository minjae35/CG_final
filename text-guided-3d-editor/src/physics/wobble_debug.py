"""Optional diagnostics after PhysGaussian when ``--wobble-debug`` saves per-frame tracks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console


def _gauss_frame_shapes(gauss_dir: Path) -> dict[int, tuple[int, ...]]:
    """Return frame_index -> (N, 3) without loading full arrays into RAM."""
    out: dict[int, tuple[int, ...]] = {}
    for p in sorted(gauss_dir.glob("gauss_*.npy")):
        try:
            fi = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        mm = np.load(p, mmap_mode="r")
        if mm.ndim != 2 or mm.shape[1] != 3:
            del mm
            continue
        out[fi] = (int(mm.shape[0]), int(mm.shape[1]))
        del mm
    return out


def log_wobble_motion_from_gauss_npy(
    sim_run: Path,
    *,
    n_obj: int,
    console: Console,
    sample_frames: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 8, 12, 20, 40, 80, 120, 160, 199),
) -> None:
    """Load ``mesh_gauss_xyz/gauss_XXXX.npy`` (PhysGaussian export) and print motion stats.

    PhysGaussian writes these **before** concatenating static background Gaussians into
    the render buffer — each file should be the simulated object slice only, with a
    stable row count per run. If ``mesh_gauss_xyz/`` mixes files from different runs,
    row counts can differ; we scan shapes, pick a reference frame with the largest
    ``N``, and compare only the common prefix ``[:n_common]`` so analysis never crashes.
    """
    d = sim_run / "mesh_gauss_xyz"
    if not d.is_dir():
        console.print("[yellow]wobble-debug:[/] no mesh_gauss_xyz/ (re-run with --wobble-debug)")
        return
    paths = sorted(d.glob("gauss_*.npy"))
    if len(paths) < 2:
        console.print("[yellow]wobble-debug:[/] expected gauss_*.npy sequence under mesh_gauss_xyz/")
        return

    def _idx(stem: str) -> int:
        return int(stem.split("_")[-1])

    by_idx: dict[int, Path] = {_idx(p.stem): p for p in paths}
    shape_by_fi = _gauss_frame_shapes(d)

    if not shape_by_fi:
        console.print("[yellow]wobble-debug:[/] no valid gauss_*.npy (expected N×3 float)")
        return

    # Log a compact shape table (first / last few + any outliers).
    max_idx = max(shape_by_fi)
    rows_shape: list[str] = []
    counts = sorted({s[0] for s in shape_by_fi.values()})
    for fi in sorted(shape_by_fi)[:5]:
        p = by_idx[fi]
        rows_shape.append(f"    gauss_{fi:04d}.npy  shape={shape_by_fi[fi]}  path={p}")
    if max_idx > 5:
        rows_shape.append("    …")
    for fi in sorted(shape_by_fi)[-3:]:
        if fi <= 5:
            continue
        p = by_idx[fi]
        rows_shape.append(f"    gauss_{fi:04d}.npy  shape={shape_by_fi[fi]}  path={p}")
    console.print(
        f"[bold]wobble-debug[/] pipeline n_obj={int(n_obj)}  "
        f"distinct_row_counts={counts}  frames_on_disk={len(shape_by_fi)}  max_index={max_idx}"
    )
    console.print("[dim]Per-frame gauss npy (sample):[/]\n" + "\n".join(rows_shape))
    if len(counts) > 1:
        console.print(
            "[yellow]wobble-debug:[/] row counts differ across frames — often leftover "
            "`mesh_gauss_xyz/*.npy` from an older run mixed with this one. "
            "Metrics use the **largest common prefix** vs the reference frame below."
        )

    # Reference: prefer frame 0 if present, else the frame with the most rows.
    ref_fi = 0 if 0 in shape_by_fi else max(shape_by_fi, key=lambda k: shape_by_fi[k][0])
    ref_path = by_idx[ref_fi]
    ref = np.asarray(np.load(ref_path), dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 3:
        console.print(f"[yellow]wobble-debug:[/] bad reference shape {ref.shape} at {ref_path}")
        return

    n_ref = ref.shape[0]
    n_expect = min(int(n_obj), n_ref)
    console.print(
        f"[cyan]wobble-debug reference[/] frame={ref_fi}  rows={n_ref}  "
        f"(compare min(rows, pipeline n_obj)={n_expect})"
    )

    want = sorted(set(sample_frames) | {0, ref_fi, max_idx})
    out_lines: list[str] = []
    out_lines.append(
        "[dim]Per-frame vs ref: |COM−COM0| = ‖mean(Δ)‖; raw = x−x0; "
        "NRᵢ = Δᵢ − mean(Δ) (bulk translation of the displacement field removed).[/]"
    )
    for fi in want:
        if fi not in by_idx or fi not in shape_by_fi:
            continue
        cur_path = by_idx[fi]
        cur = np.asarray(np.load(cur_path), dtype=np.float64)
        if cur.ndim != 2 or cur.shape[1] != 3:
            out_lines.append(f"  frame {fi:4d}  skip bad shape {cur.shape}  {cur_path}")
            continue
        n_cur = cur.shape[0]
        n_common = min(n_ref, n_cur)
        if n_common == 0:
            out_lines.append(f"  frame {fi:4d}  skip empty  {cur_path}")
            continue
        ref_s = ref[:n_common]
        cur_s = cur[:n_common]
        delta = cur_s - ref_s
        d_mean = delta.mean(axis=0)
        delta_nr = delta - d_mean
        norms_raw = np.linalg.norm(delta, axis=1)
        norms_nr = np.linalg.norm(delta_nr, axis=1)
        com = cur_s.mean(axis=0)
        com0 = ref_s.mean(axis=0)
        trunc = ""
        if n_cur != n_ref:
            # Avoid Rich markup interpreting ``[...]`` as a style tag.
            trunc = f"  (truncated to first {n_common} rows vs ref)"
        out_lines.append(
            f"  frame {fi:4d}  rows={n_cur}{trunc}\n"
            f"      |COM−COM0|={float(np.linalg.norm(com - com0)):.6f}  "
            f"‖mean(Δ)‖={float(np.linalg.norm(d_mean)):.6f}\n"
            f"      raw max/mean ‖Δᵢ‖={float(norms_raw.max()):.6f} / {float(norms_raw.mean()):.6f}  "
            f"NR max/mean ‖Δᵢ−mean(Δ)‖={float(norms_nr.max()):.6f} / {float(norms_nr.mean()):.6f}"
        )
    console.print("\n".join(out_lines))
    console.print(
        "[dim]Simulation / mp4 completed before this step; wobble-debug is analysis-only.[/]"
    )
