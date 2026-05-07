from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

from generation.object_postprocess import (
    enhance_generation_prompt,
    is_red_duck_prompt,
    wants_duck_realism,
)
from placement.surface_aware import refine_floor_contact_y


def test_rubber_duck_is_not_red_duck_prompt() -> None:
    assert not is_red_duck_prompt("a rubber duck")
    assert is_red_duck_prompt("a red rubber duck")
    assert is_red_duck_prompt("red duck")


def test_wants_duck_realism_keywords() -> None:
    assert wants_duck_realism("a rubber duck")
    assert wants_duck_realism("yellow duck toy")
    assert wants_duck_realism("a white ceramic vase")
    assert wants_duck_realism("a wooden chair")
    assert not wants_duck_realism("a teddy bear")
    assert not wants_duck_realism("a tree")


def test_enhance_yellow_rubber_duck_prompt() -> None:
    out = enhance_generation_prompt("a rubber duck")
    assert "yellow" in out.lower() or "classic" in out.lower()
    assert "red rubber duck toy" not in out.lower()


def test_refine_floor_contact_snaps_to_support(tmp_path) -> None:
    """+Y-down: support at y=2.0; frustum hint y=1.55; refine should move toward 2.0."""
    n = 400
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.04, size=n)
    z = rng.normal(0.0, 0.04, size=n)
    y = rng.normal(2.0, 0.012, size=n)
    op = np.full(n, 0.9, dtype=np.float32)
    noise = np.stack(
        [
            rng.normal(0.0, 0.08, size=80),
            rng.normal(1.45, 0.02, size=80),
            rng.normal(0.0, 0.08, size=80),
        ],
        axis=1,
    )
    x = np.concatenate([x, noise[:, 0]])
    y = np.concatenate([y, noise[:, 1]])
    z = np.concatenate([z, noise[:, 2]])
    op = np.concatenate([op, np.full(80, 0.7, dtype=np.float32)])
    verts = np.zeros(
        len(x),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("opacity", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
        ],
    )
    verts["x"] = x.astype(np.float32)
    verts["y"] = y.astype(np.float32)
    verts["z"] = z.astype(np.float32)
    verts["opacity"] = op
    ply_path = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(ply_path))

    refined = refine_floor_contact_y(ply_path, 0.0, 0.0, floor_y_hint=1.55)
    assert refined is not None
    assert refined > 1.55 + 0.2
