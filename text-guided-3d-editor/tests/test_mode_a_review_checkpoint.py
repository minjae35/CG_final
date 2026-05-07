from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline import MODE_A_REVIEW_JSON, _load_mode_a_review_checkpoint, _save_mode_a_review_checkpoint


def test_save_load_review_checkpoint_roundtrip(tmp_path: Path) -> None:
    gen = tmp_path / "sds_run"
    gen.mkdir()
    scaled = gen / "object_scaled.ply"
    obj = gen / "object_model.ply"
    scaled.write_text("ply", encoding="utf-8")
    obj.write_text("ply", encoding="utf-8")
    align = np.array([0.0, 1.0, 0.05], dtype=np.float64)
    out = _save_mode_a_review_checkpoint(
        gen,
        text="a rubber duck",
        yellow_duck=True,
        scaled=scaled,
        obj_ply=obj,
        place_x=1.0,
        place_y=2.0,
        place_z=3.0,
        floor_y=2.0,
        canonical_frame="image",
        scale_factor=0.4,
        post_rotate=(0.0, 5.0, 0.0),
        opacity_filter_logit=-0.1,
        contact_base_ao=0.2,
        image_mode=True,
        raw_object=False,
        mesh_render=True,
        mesh_skin_k=12,
        swap_red_yellow=False,
        mesh_gaussians=False,
        align_y=align,
    )
    assert out.name == MODE_A_REVIEW_JSON
    data = _load_mode_a_review_checkpoint(gen)
    assert data["text"] == "a rubber duck"
    assert data["yellow_duck"] is True
    assert data["mesh_render"] is True
    assert data["align_y"] == [0.0, 1.0, 0.05]
    z = json.loads(out.read_text(encoding="utf-8"))
    assert z["scale_factor"] == 0.4
