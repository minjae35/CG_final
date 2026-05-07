"""Tests for Mip-NeRF 360 dataset download/layout (PRD v2)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

from data_utils.dataset_downloader import _find_scene_with_colmap, download_and_prepare


def _make_fake_360_zip(tmp: Path, scene: str = "room") -> Path:
    root = tmp / "bundle"
    (root / scene / "images").mkdir(parents=True)
    (root / scene / "sparse" / "0").mkdir(parents=True)
    (root / scene / "images" / "a.jpg").write_bytes(b"\xff\xd8")
    (root / scene / "sparse" / "0" / "cameras.bin").write_bytes(b"stub")

    zip_path = tmp / "360_v2.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                arc = p.relative_to(root)
                zf.write(p, arc)
    return zip_path


def test_find_scene_with_colmap(tmp_path: Path) -> None:
    scene = tmp_path / "room"
    (scene / "images").mkdir(parents=True)
    (scene / "sparse" / "0").mkdir(parents=True)
    assert _find_scene_with_colmap(tmp_path, "room") == scene


def test_download_and_prepare_extract_and_copy(tmp_path: Path) -> None:
    fake_zip = _make_fake_360_zip(tmp_path)
    pr = tmp_path / "proj"
    pr.mkdir()

    def fake_retrieve(url: str, dst: str) -> tuple[str, dict]:
        shutil = __import__("shutil")

        shutil.copy2(fake_zip, dst)
        return dst, {}

    with patch("data_utils.dataset_downloader.urllib.request.urlretrieve", fake_retrieve):
        out = download_and_prepare(
            "room",
            pr,
            dataset_url="http://example.com/360_v2.zip",
            raw_dir="data/mipnerf360_raw",
            data_root="data/mipnerf360",
        )

    assert out == (pr / "data" / "mipnerf360" / "room").resolve()
    assert (out / "images" / "a.jpg").is_file()
    assert (out / "sparse" / "0" / "cameras.bin").is_file()


def test_download_and_prepare_idempotent(tmp_path: Path) -> None:
    fake_zip = _make_fake_360_zip(tmp_path)
    pr = tmp_path / "proj"
    pr.mkdir()
    calls = {"n": 0}

    def fake_retrieve(url: str, dst: str) -> tuple[str, dict]:
        calls["n"] += 1
        import shutil as sh

        sh.copy2(fake_zip, dst)
        return dst, {}

    with patch("data_utils.dataset_downloader.urllib.request.urlretrieve", fake_retrieve):
        p1 = download_and_prepare(
            "room", pr, dataset_url="http://x/360_v2.zip", raw_dir="raw", data_root="data/mipnerf360"
        )
        p2 = download_and_prepare(
            "room", pr, dataset_url="http://x/360_v2.zip", raw_dir="raw", data_root="data/mipnerf360"
        )
    assert p1 == p2
    assert calls["n"] == 1
