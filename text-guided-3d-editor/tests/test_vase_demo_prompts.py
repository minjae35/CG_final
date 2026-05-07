from __future__ import annotations

from generation.image_generator import _build_image_prompt
from generation.object_postprocess import (
    enhance_generation_prompt,
    wants_object_realism_preset,
)


def test_vase_triggers_realism_preset() -> None:
    assert wants_object_realism_preset("a white ceramic vase")
    assert wants_object_realism_preset("a blue vase")


def test_vase_image_prompt_is_solo_and_short() -> None:
    s = _build_image_prompt("a white ceramic vase")
    low = s.lower()
    assert "vase" in low
    assert "exactly one" in low
    assert "no duplicates" in low
    assert "white" in low
    assert "duck" not in low and "ball" not in low


def test_vase_enhance_prompt_is_specific() -> None:
    s = enhance_generation_prompt("a vase")
    assert "vase" in s.lower()
    assert "duck" not in s.lower() and "ball" not in s.lower()
