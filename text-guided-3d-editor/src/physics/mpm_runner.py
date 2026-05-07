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

    cmd = [
        sys.executable,
        str(sim_py),
        "--model_path",
        str(model_path),
        "--output_path",
        str(out),
        "--config",
        str(config_path),
    ]
    if render_img:
        cmd.append("--render_img")
    if compile_video:
        cmd.append("--compile_video")
    if camera_scene_path:
        cmd.extend(["--camera_scene_path", str(camera_scene_path)])
    if save_obj_centroid or save_obj_tracks_for_debug:
        cmd.append("--save_obj_centroid")
    if save_obj_gauss_xyz or save_obj_tracks_for_debug:
        cmd.append("--save_obj_gauss_xyz")

    frames_dir = out / "frames"
    video_dir = out / "video"
    if render_img and frames_dir.exists():
        shutil.rmtree(frames_dir)
    backup_path: Path | None = None
    if compile_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        backup_path = _backup_existing_video(video_dir)

    env = os.environ.copy()
    pg = str(sim_py.parent)
    env["PYTHONPATH"] = pg + os.pathsep + env.get("PYTHONPATH", "")
    if taichi_device_memory_gb is not None:
        env["PHYSGAUSSIAN_TAICHI_DEVICE_MEMORY_GB"] = str(float(taichi_device_memory_gb))
    subprocess.run(cmd, check=True, cwd=pg, env=env, timeout=86400)

    # PhysGaussian writes frames under <out>/frames/ and video to <out>/video/output.mp4
    primary = out / "video" / "output.mp4"
    legacy_root = out / "output.mp4"
    if playback_seconds and frames_dir.is_dir():
        frames = sorted(frames_dir.glob("*.png"))
        if frames:
            fps = max(1.0, len(frames) / float(playback_seconds))
            # HD cap + Main@L4.0 + faststart — plays in most players (full 3K High@L5.x often won't).
            subprocess.run(
                [
                    "ffmpeg",
                    "-framerate",
                    f"{fps:.6f}",
                    "-i",
                    str(frames_dir / "%04d.png"),
                    "-vf",
                    "scale='min(1280,iw)':-2",
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
    if legacy_root.is_file():
        return str(legacy_root)
    raise FileNotFoundError(f"PhysGaussian did not produce a video at {primary}")
