"""Wrapper around official gaussian-splatting/train.py."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT


def train_scene(
    data_path: Path | str,
    output_path: Path | str,
    iterations: int = 30000,
) -> Path:
    """COLMAP-format scene directory -> trained 3DGS; returns path to point_cloud.ply."""
    data_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    train_py = PROJECT_ROOT / "submodules" / "gaussian-splatting" / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(train_py)

    cmd = [
        sys.executable,
        str(train_py),
        "-s",
        str(data_path),
        "-m",
        str(output_path),
        "--iterations",
        str(iterations),
        "--disable_viewer",
    ]
    env = os.environ.copy()
    gs_dir = str(train_py.parent)
    env["PYTHONPATH"] = gs_dir + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=gs_dir, env=env)

    ply_path = output_path / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Training failed: {ply_path} not found")
    return ply_path
