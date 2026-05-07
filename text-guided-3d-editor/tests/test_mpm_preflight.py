from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from physics.mpm_preflight import approximate_mpm_normalized_positions, validate_merged_object_for_mpm


def test_approximate_mpm_normalized_positions_centered() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    m = approximate_mpm_normalized_positions(pts)
    assert m.shape == (2, 3)
    assert float(m[:, 0].min()) >= -1e-6
    assert float(m[:, 0].max()) <= 2.0 + 1e-6


def test_validate_merged_object_for_mpm_ok(tmp_path: Path) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("scale_0", "f4"),
        ("opacity", "f4"),
    ]
    v = np.zeros(4, dtype=dtype)
    v["x"] = [0, 1, 2, 5]
    v["y"] = [0, 0, 0, 0]
    v["z"] = [0, 0, 0, 0]
    v["scale_0"] = -2.0
    v["opacity"] = 1.0
    ply = PlyData([PlyElement.describe(v, "vertex")])
    p = tmp_path / "m.ply"
    ply.write(str(p))
    diag = validate_merged_object_for_mpm(p, np.array([1, 2], dtype=np.int64))
    assert diag["n_object"] == 2


def test_validate_merged_object_for_mpm_rejects_nan(tmp_path: Path) -> None:
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    v = np.zeros(2, dtype=dtype)
    v["x"] = [0.0, float("nan")]
    v["y"] = [0.0, 0.0]
    v["z"] = [0.0, 0.0]
    ply = PlyData([PlyElement.describe(v, "vertex")])
    p = tmp_path / "bad.ply"
    ply.write(str(p))
    with pytest.raises(ValueError, match="non-finite"):
        validate_merged_object_for_mpm(p, np.array([1], dtype=np.int64))
