"""Run PhysGaussian gs_simulation.py (PRD 4.3)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
    mp = Path(model_path)
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
    # Keep PhysGaussian first on sys.path, but avoid accidentally importing *repo* submodule
    # CUDA extensions (diff-gaussian-rasterization / gaussian-splatting) when a compiled
    # site-packages version exists. Those repo copies may lack compiled `_C` and cause
    # circular-import / missing-extension errors.
    prior_pp = env.get("PYTHONPATH", "")
    if prior_pp:
        parts = [p for p in prior_pp.split(os.pathsep) if p]
        blocked = (
            f"{os.sep}submodules{os.sep}gaussian-splatting{os.sep}submodules{os.sep}diff-gaussian-rasterization",
            f"{os.sep}submodules{os.sep}gaussian-splatting",
        )
        parts = [p for p in parts if not any(b in p for b in blocked)]
        prior_pp = os.pathsep.join(parts)
    env["PYTHONPATH"] = pg + (os.pathsep + prior_pp if prior_pp else "")
    # Ensure PyTorch shared libs (libc10.so, libtorch_cuda.so, ...) are discoverable in subprocess.
    # Some environments don't populate LD_LIBRARY_PATH for non-interactive subprocesses.
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            env["LD_LIBRARY_PATH"] = str(torch_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    except Exception:
        pass
    if taichi_device_memory_gb is not None:
        env["PHYSGAUSSIAN_TAICHI_DEVICE_MEMORY_GB"] = str(float(taichi_device_memory_gb))
    subprocess.run(cmd, check=True, cwd=pg, env=env, timeout=86400)

    # Normalize all variants to <out>/video/output.mp4 so downstream paths stay consistent.
    primary = video_dir / "output.mp4"
    phys_mp4 = frames_dir / "output.mp4"
    if phys_mp4.is_file():
        video_dir.mkdir(parents=True, exist_ok=True)
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
    if playback_seconds and frames_dir.is_dir():
        frames = sorted(frames_dir.glob("*.png"))
        if frames:
            fps = max(1.0, len(frames) / float(playback_seconds))
            # HD cap + Main@L4.0 + faststart — plays in most players (full 3K High@L5.x often won't).
            video_dir.mkdir(parents=True, exist_ok=True)
            _backup_existing_video(video_dir)
            subprocess.run(
                [
                    "ffmpeg",
                    "-framerate",
                    f"{fps:.6f}",
                    "-i",
                    str(frames_dir / "%04d.png"),
                    "-vf",
                    # Keep width ≤ 1280 and force even dimensions for H.264/yuv420p.
                    "scale='trunc(min(1280,iw)/2)*2':-2",
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
                    "-y",
                    str(primary),
                ],
                check=True,
            )
    if primary.is_file():
        return str(primary)
    raise FileNotFoundError(f"PhysGaussian did not produce a video at {primary}")
