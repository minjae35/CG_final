from __future__ import annotations

from pathlib import Path

import pytest

from generation import sds_generator
from generation.sds_generator import find_best_ply, find_latest_ply, generate_object


def test_find_latest_ply(tmp_path: Path) -> None:
    (tmp_path / "a.ply").write_text("a")
    (tmp_path / "b.ply").write_text("bb")
    p = find_latest_ply(tmp_path)
    assert p is not None and p.name == "b.ply"


def test_find_best_ply_prefers_stage1_model(tmp_path: Path) -> None:
    (tmp_path / "object.ply").write_bytes(b"x" * 5000)
    (tmp_path / "object_model.ply").write_bytes(b"y" * 5000)
    p = find_best_ply(tmp_path)
    assert p is not None and p.name == "object_model.ply"


def test_generate_object_returns_stage1_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dg = tmp_path / "submodules" / "dreamgaussian"
    (dg / "configs").mkdir(parents=True)
    (dg / "configs" / "text.yaml").write_text("", encoding="utf-8")
    (dg / "main.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "out = Path([a.split('=', 1)[1] for a in sys.argv if a.startswith('outdir=')][0])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'object_model.ply').write_bytes(b'x' * 5000)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sds_generator, "PROJECT_ROOT", tmp_path)

    p = generate_object("test object", tmp_path / "out", num_steps=1, use_mvdream=False)

    assert p.name == "object_model.ply"


def test_generate_object_raises_without_usable_ply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dg = tmp_path / "submodules" / "dreamgaussian"
    (dg / "configs").mkdir(parents=True)
    (dg / "configs" / "text.yaml").write_text("", encoding="utf-8")
    (dg / "main.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    monkeypatch.setattr(sds_generator, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError):
        generate_object("test object", tmp_path / "out", num_steps=1, use_mvdream=False)
