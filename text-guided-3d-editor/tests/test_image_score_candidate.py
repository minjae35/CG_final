from __future__ import annotations

from generation.image_generator import _score_candidate_info


def _info(
    *,
    h=512,
    w=512,
    kept=80_000,
    removed=0,
    cy=256.0,
    cx=256.0,
    n_components=1,
    n_peaks=1,
) -> dict:
    return {
        "h": h,
        "w": w,
        "kept_area": kept,
        "removed_area": removed,
        "centroid_yx": (cy, cx),
        "n_components": n_components,
        "n_peaks": n_peaks,
    }


def test_touching_multi_peak_blob_is_penalised() -> None:
    """4 vases lined up merge into 1 component but produce 4 horizontal peaks — must score lower."""
    single = _score_candidate_info(_info(kept=int(512 * 512 * 0.30), n_components=1, n_peaks=1))
    fake_single = _score_candidate_info(_info(kept=int(512 * 512 * 0.30), n_components=1, n_peaks=4))
    assert single > 3.0 * fake_single


def test_single_subject_outscores_multi_subject() -> None:
    """Single connected blob must beat several blobs even if combined area is similar."""
    single = _score_candidate_info(_info(kept=int(512 * 512 * 0.30), n_components=1))
    multi = _score_candidate_info(_info(kept=int(512 * 512 * 0.30), n_components=4))
    assert single > 2.0 * multi


def test_pure_outscores_split_within_kept_blob() -> None:
    pure = _score_candidate_info(_info(kept=80_000, removed=0))
    split = _score_candidate_info(_info(kept=40_000, removed=40_000))
    assert pure > split * 2.0


def test_huge_subject_is_penalised_more_than_small() -> None:
    """After downstream auto-resize a small but clean subject is recoverable; an enormous one isn't."""
    sweet = _score_candidate_info(_info(kept=int(512 * 512 * 0.30)))
    tiny = _score_candidate_info(_info(kept=int(512 * 512 * 0.03)))
    huge = _score_candidate_info(_info(kept=int(512 * 512 * 0.92)))
    assert sweet > tiny
    assert sweet > huge
    assert tiny > huge / 2.0


def test_zero_area_returns_zero() -> None:
    assert _score_candidate_info(_info(kept=0, removed=0)) == 0.0
