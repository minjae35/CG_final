"""Download and unpack Mip-NeRF 360-style dataset (PRD v2 task 1.2)."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_DATASET_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

# Typical indoor scene names in 360_v2 bundle
INDOOR_SCENES = frozenset({"room", "counter", "kitchen", "bonsai"})


def download_and_prepare(
    scene_name: str,
    project_root: Path,
    dataset_url: str = DEFAULT_DATASET_URL,
    raw_dir: str = "data/mipnerf360_raw",
    data_root: str = "data/mipnerf360",
) -> Path:
    """
    Download 360_v2.zip, extract, and ensure ``project_root/data_root/scene_name``
    contains ``images/`` and ``sparse/0/``.

    Returns the absolute path to the scene directory used for 3DGS ``-s``.
    """
    root = project_root.resolve()
    raw_path = root / raw_dir
    raw_path.mkdir(parents=True, exist_ok=True)

    zip_name = Path(dataset_url.split("/")[-1]).name or "360_v2.zip"
    zip_path = raw_path / zip_name

    if not zip_path.is_file():
        print(f"Downloading {dataset_url} -> {zip_path} ...")
        urllib.request.urlretrieve(dataset_url, zip_path)

    extract_root = raw_path / "extracted"
    if not extract_root.is_dir():
        extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extract_ok"
    if not marker.is_file():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        marker.write_text("ok", encoding="utf-8")

    scene_src = _find_scene_with_colmap(extract_root, scene_name)
    if scene_src is None:
        raise FileNotFoundError(
            f"Could not find COLMAP scene {scene_name!r} under {extract_root} "
            f"(need images/ and sparse/0/)."
        )

    out_scene = root / data_root / scene_name
    out_scene.parent.mkdir(parents=True, exist_ok=True)
    if out_scene.is_symlink() or out_scene.is_dir():
        if out_scene.is_symlink():
            out_scene.unlink()
        elif out_scene.is_dir():
            # Idempotent: already a real dir with data
            if (out_scene / "images").is_dir() and (out_scene / "sparse" / "0").is_dir():
                return out_scene
            shutil.rmtree(out_scene)

    shutil.copytree(scene_src, out_scene, symlinks=True)
    return out_scene


def _find_scene_with_colmap(extract_root: Path, scene_name: str) -> Path | None:
    """Locate ``scene_name`` directory that contains ``images`` and ``sparse/0``."""

    def ok(p: Path) -> bool:
        return (p / "images").is_dir() and (p / "sparse" / "0").is_dir()

    direct = extract_root / scene_name
    if ok(direct):
        return direct

    found: Path | None = None
    for p in extract_root.rglob(scene_name):
        if p.is_dir() and ok(p):
            found = p
            break
    if found is not None:
        return found

    # Some archives nest e.g. 360_v2/room
    for p in extract_root.rglob("*"):
        if p.is_dir() and p.name == scene_name and ok(p):
            return p

    return None
