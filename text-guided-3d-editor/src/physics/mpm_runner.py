"""Run PhysGaussian gs_simulation.py (PRD 4.3)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import json

from config import PROJECT_ROOT


def _backup_existing_video(video_dir: Path) -> Path | None:
    """Rename ``video_dir/output.mp4`` to ``output_<timestamp>.mp4`` if present.

    PhysGaussian's own renderer also renames the previous ``output.mp4`` before
    writing, but the optional ``playback_seconds`` re-encode below uses
    ``ffmpeg -y`` which would otherwise blow the prior result away.  Backing it
    up here covers both paths.
    """
    current = video_dir / "output.mp4"
    if not current.is_file():
        return None
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    backup = current.with_name(f"output_{stamp}.mp4")
    n = 1
    while backup.is_file():
        backup = current.with_name(f"output_{stamp}_{n}.mp4")
        n += 1
    current.rename(backup)
    return backup


def _write_motion_debug_ply(*, out_dir: Path, model_ply: Path) -> None:
    """
    Produce a quick debug point-cloud visualization:
    - selected gaussians: red
    - moved (subset of selected): green
    - selected but not moved: blue
    """
    import numpy as np
    from plyfile import PlyData, PlyElement

    moved_path = out_dir / "debug" / "moved_selected_local_indices.npy"
    selected_path = out_dir / "selected_indices.npy"
    if not moved_path.is_file() or not selected_path.is_file():
        return
    if not model_ply.is_file():
        return

    moved_local = np.load(moved_path).astype(np.int64, copy=False)
    selected = np.load(selected_path).astype(np.int64, copy=False)
    moved_local_set = set(map(int, moved_local.tolist()))

    ply = PlyData.read(str(model_ply))
    v = ply["vertex"]
    pos = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1
    ).astype(np.float32, copy=False)
    pts = pos[selected]

    rgb = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = 255  # selected = red by default

    moved_mask = np.zeros((pts.shape[0],), dtype=bool)
    if moved_local.size:
        moved_mask = np.array(
            [i in moved_local_set for i in range(len(selected))], dtype=bool
        )

    # selected-but-not-moved (pinned/static) → blue
    rgb[~moved_mask] = np.array([0, 0, 255], dtype=np.uint8)
    # moved → green
    rgb[moved_mask] = np.array([0, 255, 0], dtype=np.uint8)

    verts = np.empty(
        pts.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_ply = debug_dir / "motion_debug_selected_points.ply"
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(out_ply))


def _write_displacement_debug(*, out_dir: Path, model_ply: Path, config_path: Path) -> None:
    """
    Spatial displacement analysis for selected gaussians.

    Requires shim outputs:
    - debug/selected_means3d_first.npy
    - debug/selected_means3d_last.npy
    """
    import numpy as np

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        up = np.array(
            cfg.get("mpm_space_vertical_upward_axis", [0.0, 1.0, 0.0]),
            dtype=np.float32,
        )
    except Exception:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    from physics.displacement_analysis import analyze_and_write

    analyze_and_write(out_dir=out_dir, model_ply=model_ply, up_axis=up)


def _sorted_numeric_png_frames(frames_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(index, path)`` for ``<digits>.png`` only, sorted by index.

    Ignores unrelated ``*.png`` (overlays, etc.) so ffmpeg's ``%04d`` sequence
    is not confused. PhysGaussian uses names like ``0000.png``.
    """
    out: list[tuple[int, Path]] = []
    for p in frames_dir.glob("*.png"):
        m = re.fullmatch(r"(\d+)\.png", p.name, flags=re.IGNORECASE)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _indices_contiguous(ids: list[int]) -> bool:
    if not ids:
        return False
    start = ids[0]
    return ids == list(range(start, start + len(ids)))


def _ffmpeg_reencode_playback_pngs(
    *,
    frames_dir: Path,
    out_mp4: Path,
    pairs: list[tuple[int, Path]],
    playback_seconds: float,
) -> None:
    """Build ``out_mp4`` from numbered PNGs, stretching to ``playback_seconds``."""
    fps = max(1.0, len(pairs) / float(playback_seconds))
    ids = [i for i, _ in pairs]
    scale_vf = "scale='trunc(min(1280,iw)/2)*2':-2"
    enc = [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-profile:v",
        "main",
        "-level:v",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
    ]

    if _indices_contiguous(ids):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                f"{fps:.6f}",
                "-start_number",
                str(ids[0]),
                "-i",
                str(frames_dir / "%04d.png"),
                "-frames:v",
                str(len(ids)),
                "-vf",
                scale_vf,
                *enc,
                str(out_mp4),
            ],
            check=True,
        )
        return

    frame_dur = 1.0 / fps
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for _, p in pairs:
            ap = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{ap}'\n")
            f.write(f"duration {frame_dur:.9f}\n")
        if pairs:
            ap = str(pairs[-1][1].resolve()).replace("'", "'\\''")
            f.write(f"file '{ap}'\n")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-vf",
                scale_vf,
                *enc,
                str(out_mp4),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def run_simulation(
    model_path: str,
    config_path: str,
    output_path: str,
    render_img: bool = True,
    compile_video: bool = True,
    camera_scene_path: str | None = None,
    playback_seconds: float | None = None,
    save_obj_centroid: bool = False,
    save_obj_gauss_xyz: bool = False,
    *,
    save_obj_tracks_for_debug: bool = False,
    taichi_device_memory_gb: float | None = None,
) -> str:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    sim_py = PROJECT_ROOT / "submodules" / "PhysGaussian" / "gs_simulation.py"
    if not sim_py.is_file():
        raise FileNotFoundError(sim_py)

    # PhysGaussian expects `--model_path` to point at the 3DGS *model directory*
    # containing `point_cloud/iteration_*/point_cloud.ply`. Our pipeline often
    # passes the PLY path directly, so normalize it here.
    model_ply = Path(model_path)
    mp = model_ply
    if mp.is_file() and mp.suffix.lower() == ".ply":
        # <scene>/point_cloud/iteration_x/point_cloud.ply -> <scene>
        try:
            mp = mp.parents[2]
        except IndexError:
            pass

    frames_dir = out / "frames"
    # Mode-b layout: frames/ + video/ (single folder name, no plural).
    video_dir = out / "video"
    legacy_video_dir = out / "videos"
    if legacy_video_dir.is_dir() and not video_dir.is_dir():
        # One-time migration: rename videos/ -> video/ so downstream paths are stable.
        legacy_video_dir.rename(video_dir)

    # PhysGaussian (this repo's `gs_simulation.py`) writes PNG frames directly
    # under `--output_path` and writes `output.mp4` into the same folder.
    # So: pass frames_dir to PhysGaussian so frames land in <out>/frames/,
    # then move frames_dir/output.mp4 → <out>/video/output.mp4.
    phys_out = frames_dir

    cmd = [
        sys.executable,
        str(sim_py),
        "--model_path",
        str(mp),
        "--output_path",
        str(phys_out),
        "--config",
        str(config_path),
    ]
    if render_img:
        cmd.append("--render_img")
    if compile_video:
        cmd.append("--compile_video")
    # NOTE: Upstream PhysGaussian `gs_simulation.py` in this repo does not accept
    # `--camera_scene_path`. Rendering/compilation is driven by the sim config.
    # Keep the parameter for backward compatibility, but do not forward it.
    if save_obj_centroid or save_obj_tracks_for_debug:
        cmd.append("--save_obj_centroid")
    if save_obj_gauss_xyz or save_obj_tracks_for_debug:
        cmd.append("--save_obj_gauss_xyz")

    if render_img and frames_dir.exists():
        shutil.rmtree(frames_dir)
    backup_path: Path | None = None
    if compile_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        backup_path = _backup_existing_video(video_dir)

    env = os.environ.copy()
    pg = str(sim_py.parent)
    # Keep PhysGaussian first on sys.path, and inject a compatibility shim for
    # `diff_gaussian_rasterization` API differences (e.g. newer builds require
    # `antialiasing` in `GaussianRasterizationSettings`).
    prior_pp = env.get("PYTHONPATH", "")
    if prior_pp:
        parts = [p for p in prior_pp.split(os.pathsep) if p]
        blocked = (
            f"{os.sep}submodules{os.sep}gaussian-splatting{os.sep}submodules{os.sep}diff-gaussian-rasterization",
            f"{os.sep}submodules{os.sep}gaussian-splatting",
        )
        parts = [p for p in parts if not any(b in p for b in blocked)]
        prior_pp = os.pathsep.join(parts)
    shim = str(PROJECT_ROOT / "src" / "physgaussian_shim")
    env["PYTHONPATH"] = os.pathsep.join([p for p in (shim, pg, prior_pp) if p])
    # Ensure PyTorch shared libs (libc10.so, libtorch_cuda.so, ...) are discoverable in subprocess.
    # Some environments don't populate LD_LIBRARY_PATH for non-interactive subprocesses.
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            env["LD_LIBRARY_PATH"] = str(torch_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    except Exception:
        pass
    # Always propagate the intended Taichi memory budget to the subprocess.
    # The PhysGaussian submodule may hard-code `ti.init(device_memory_GB=8.0)`, so the
    # shim (`src/physgaussian_shim/sitecustomize.py`) uses this env var to override it.
    if taichi_device_memory_gb is not None:
        env["PHYSGAUSSIAN_TAICHI_DEVICE_MEMORY_GB"] = str(float(taichi_device_memory_gb))

    # PhysGaussian debug: record which selected gaussians actually move (via shim wrapper).
    # We save moved indices under <out>/debug/ for post-run visualization.
    env["PHYSGAUSSIAN_DEBUG_DIR"] = str((out / "debug").resolve())
    # Provide config path to shim so it can apply render overrides without submodule edits.
    env["PHYSGAUSSIAN_CONFIG_PATH"] = str(Path(config_path).resolve())
    # Tell the shim how many gaussians are "selected" (in the concatenated tensor layout).
    # Prefer <out>/selected_indices.npy, which always exists for mode-b.
    try:
        import numpy as np

        sel_path = out / "selected_indices.npy"
        if sel_path.is_file():
            sel = np.load(sel_path)
            env["PHYSGAUSSIAN_SELECTED_N"] = str(int(sel.shape[0]))
    except Exception:
        pass

    subprocess.run(cmd, check=True, cwd=pg, env=env, timeout=86400)
    # If the shim wrote moved indices, turn it into a simple colored PLY.
    try:
        _write_motion_debug_ply(out_dir=out, model_ply=model_ply)
    except Exception:
        pass
    # If the shim wrote first/last means, write displacement distribution debug outputs.
    try:
        _write_displacement_debug(out_dir=out, model_ply=model_ply, config_path=Path(config_path))
    except Exception:
        pass

    numbered_pairs = (
        _sorted_numeric_png_frames(frames_dir) if frames_dir.is_dir() else []
    )

    # Normalize all variants to <out>/video/output.mp4 so downstream paths stay consistent.
    primary = video_dir / "output.mp4"
    phys_mp4 = frames_dir / "output.mp4"
    if playback_seconds and numbered_pairs:
        video_dir.mkdir(parents=True, exist_ok=True)
        if backup_path is None:
            _backup_existing_video(video_dir)
        try:
            _ffmpeg_reencode_playback_pngs(
                frames_dir=frames_dir,
                out_mp4=primary,
                pairs=numbered_pairs,
                playback_seconds=float(playback_seconds),
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            if phys_mp4.is_file():
                if backup_path is None:
                    _backup_existing_video(video_dir)
                phys_mp4.replace(primary)
            else:
                raise FileNotFoundError(
                    "Playback re-encode from PNGs failed and no "
                    f"frames/output.mp4 fallback exists ({exc})."
                ) from exc
        else:
            if phys_mp4.is_file():
                phys_mp4.unlink(missing_ok=True)
    elif phys_mp4.is_file():
        video_dir.mkdir(parents=True, exist_ok=True)
        if backup_path is None:
            _backup_existing_video(video_dir)
        phys_mp4.replace(primary)
    # Backward compatibility: some variants may still write to legacy paths.
    legacy_root = out / "output.mp4"
    legacy_video_primary = legacy_video_dir / "output.mp4"
    if legacy_root.is_file() and not primary.is_file():
        video_dir.mkdir(parents=True, exist_ok=True)
        _backup_existing_video(video_dir)
        legacy_root.replace(primary)
    if legacy_video_primary.is_file() and not primary.is_file():
        video_dir.mkdir(parents=True, exist_ok=True)
        _backup_existing_video(video_dir)
        legacy_video_primary.replace(primary)
    if primary.is_file():
        return str(primary)
    raise FileNotFoundError(f"PhysGaussian did not produce a video at {primary}")
