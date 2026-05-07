"""DreamGaussian subprocess wrapper (PRD 3.1)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT


def find_latest_ply(output_dir: Path, min_mtime_ns: int = 0) -> Path | None:
    plies = [
        p for p in Path(output_dir).rglob("*.ply")
        if p.stat().st_mtime_ns >= min_mtime_ns
    ]
    if not plies:
        return None
    return max(plies, key=lambda p: (p.stat().st_mtime_ns, p.name))


def find_best_ply(output_dir: Path, min_mtime_ns: int = 0) -> Path | None:
    """Prefer the stage-1 Gaussian model (full 3DGS properties) over mtime-latest.

    DreamGaussian stage-1 saves `{save_path}.ply` with all Gaussian properties
    needed for PhysGaussian. Stage-2 outputs (mesh, albedo) are not useful here.
    Fall back to mtime-latest if the preferred names are absent.
    """
    preferred = ["object_model.ply", "object.ply", "object_ready.ply"]
    for name in preferred:
        candidate = output_dir / name
        if (
            candidate.is_file()
            and candidate.stat().st_size > 4096
            and candidate.stat().st_mtime_ns >= min_mtime_ns
        ):
            return candidate
    return find_latest_ply(output_dir, min_mtime_ns=min_mtime_ns)


def _write_minimal_ply(path: Path) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    v = np.zeros(1, dtype=dtype)
    v["opacity"] = 1.0
    v["scale_0"] = v["scale_1"] = v["scale_2"] = 0.01
    v["rot_0"] = 1.0
    el = PlyElement.describe(v, "vertex")
    PlyData([el], text=False).write(str(path))


def _mvdream_available() -> bool:
    try:
        import mvdream  # noqa: F401
        return True
    except ImportError:
        return False


def generate_object(
    text_prompt: str,
    output_dir: Path | str,
    num_steps: int = 500,
    *,
    hf_key: str | None = None,
    sd_version: str | None = None,
    use_mvdream: bool | None = None,
) -> Path:
    """Run DreamGaussian text-to-3D; returns path to produced .ply if found.

    use_mvdream=None  → auto: use MVDream if installed (much better quality)
    use_mvdream=True  → force MVDream (raises if not installed)
    use_mvdream=False → force SD (skip MVDream even if installed)
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dg = PROJECT_ROOT / "submodules" / "dreamgaussian"
    main_py = dg / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(main_py)

    cfg = dg / "configs" / "text.yaml"
    cmd = [
        sys.executable,
        str(main_py),
        "--config",
        str(cfg),
        "gui=false",
        f"iters={num_steps}",
        f"outdir={output_dir}",
        f"prompt={text_prompt}",
        "save_path=object",
    ]

    # MVDream: multi-view consistent generation → much better 3D shape quality.
    # When enabled, sd_version/hf_key are ignored (MVDream has its own backbone).
    want_mvdream = use_mvdream if use_mvdream is not None else _mvdream_available()
    if want_mvdream:
        cmd.append("mvdream=true")
    else:
        if hf_key:
            cmd.append(f"hf_key={hf_key}")
        if sd_version:
            cmd.append(f"sd_version={sd_version}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(dg) + os.pathsep + env.get("PYTHONPATH", "")
    env["DG_SKIP_GEO_TEX"] = "1"
    started_ns = time.time_ns()
    stdout_log = output_dir / "dreamgaussian.stdout.log"
    stderr_log = output_dir / "dreamgaussian.stderr.log"
    try:
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(dg),
                env=env,
                timeout=86400,
                stdout=out,
                stderr=err,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        found = find_best_ply(output_dir, min_mtime_ns=started_ns)
        if found is not None:
            return found
        raise RuntimeError(f"DreamGaussian failed before producing a usable PLY: {exc}") from exc

    found = find_best_ply(output_dir, min_mtime_ns=started_ns)
    if found is not None:
        return found
    raise FileNotFoundError(f"DreamGaussian did not produce a usable PLY in {output_dir}")


def _process_image_to_rgba(
    image_path: Path,
    *,
    size: int = 256,
    border_ratio: float = 0.2,
) -> Path:
    """Run DreamGaussian's process.py to remove background and produce RGBA PNG."""
    image_path = Path(image_path).resolve()
    dg = PROJECT_ROOT / "submodules" / "dreamgaussian"
    process_py = dg / "process.py"
    if not process_py.is_file():
        raise FileNotFoundError(process_py)

    cmd = [
        sys.executable,
        str(process_py),
        str(image_path),
        "--size",
        str(int(size)),
        "--border_ratio",
        str(float(border_ratio)),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(dg) + os.pathsep + env.get("PYTHONPATH", "")
    log_dir = image_path.parent
    stdout_log = log_dir / "rembg.stdout.log"
    stderr_log = log_dir / "rembg.stderr.log"
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        subprocess.run(cmd, check=True, cwd=str(dg), env=env, timeout=900, stdout=out, stderr=err)

    rgba = image_path.with_name(image_path.stem + "_rgba.png")
    if not rgba.is_file():
        raise FileNotFoundError(f"process.py did not produce RGBA at {rgba}")
    return rgba


def generate_object_from_image(
    image_path: Path | str,
    output_dir: Path | str,
    *,
    num_steps: int = 500,
    elevation: float = 0.0,
    use_stable_zero123: bool = True,
    process_for_dg: bool = True,
    sh_degree: int | None = None,
) -> Path:
    """Run DreamGaussian image-to-3D (stage 1) on a single reference image.

    If ``process_for_dg`` is True the image is first run through DG's rembg
    pipeline to add an alpha channel and centre the subject.  The function
    returns the path to the produced stage-1 ``.ply``.
    """
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dg = PROJECT_ROOT / "submodules" / "dreamgaussian"
    main_py = dg / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(main_py)

    rgba = _process_image_to_rgba(image_path) if process_for_dg else image_path

    cfg = dg / "configs" / "image.yaml"
    cmd = [
        sys.executable,
        str(main_py),
        "--config",
        str(cfg),
        "gui=false",
        f"iters={int(num_steps)}",
        f"outdir={output_dir}",
        f"input={rgba}",
        f"elevation={float(elevation)}",
        "save_path=object",
    ]
    if use_stable_zero123:
        cmd.append("stable_zero123=true")
    if sh_degree is not None:
        cmd.append(f"sh_degree={int(sh_degree)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(dg) + os.pathsep + env.get("PYTHONPATH", "")
    env["DG_SKIP_GEO_TEX"] = "1"
    started_ns = time.time_ns()
    stdout_log = output_dir / "dreamgaussian.stdout.log"
    stderr_log = output_dir / "dreamgaussian.stderr.log"
    try:
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(dg),
                env=env,
                timeout=86400,
                stdout=out,
                stderr=err,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        found = find_best_ply(output_dir, min_mtime_ns=started_ns)
        if found is not None:
            return found
        raise RuntimeError(f"DreamGaussian image-to-3D failed before producing a usable PLY: {exc}") from exc

    found = find_best_ply(output_dir, min_mtime_ns=started_ns)
    if found is not None:
        return found
    raise FileNotFoundError(f"DreamGaussian image-to-3D did not produce a usable PLY in {output_dir}")
