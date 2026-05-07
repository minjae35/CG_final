from __future__ import annotations

from generation.image_generator import _build_image_prompt


def test_rubber_duck_reference_prompt_is_short_and_single_subject() -> None:
    s = _build_image_prompt("a rubber duck")
    low = s.lower()
    assert "exactly one" in low
    assert "solo" in low or "single" in low
    assert "no second duck" in low
    assert "red rubber" not in low
    assert len(s) < 400


def test_red_duck_prompt_mentions_red() -> None:
    s = _build_image_prompt("a red rubber duck")
    assert "red" in s.lower()
