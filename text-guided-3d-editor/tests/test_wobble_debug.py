from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console

from physics.wobble_debug import log_wobble_motion_from_gauss_npy


def test_log_wobble_motion_from_gauss_npy(tmp_path: Path) -> None:
    d = tmp_path / "mesh_gauss_xyz"
    d.mkdir(parents=True)
    n = 20
    base = np.zeros((n, 3), dtype=np.float32)
    base[:, 0] = np.linspace(0, 0.1, n)
    np.save(d / "gauss_0000.npy", base)
    moved = base.copy()
    moved[:, 0] += 0.02
    np.save(d / "gauss_0005.npy", moved)
    c = Console(record=True, width=120)
    log_wobble_motion_from_gauss_npy(tmp_path, n_obj=n, console=c, sample_frames=(0, 5))
    out = c.export_text()
    assert "NR max/mean" in out
    assert "0.020000" in out or "0.02" in out


def test_log_wobble_motion_mismatched_row_counts(tmp_path: Path) -> None:
    """Older + newer gauss npy on disk: must not crash; compare common prefix."""
    d = tmp_path / "mesh_gauss_xyz"
    d.mkdir(parents=True)
    big = np.zeros((30, 3), dtype=np.float32)
    big[:, 1] = np.arange(30, dtype=np.float32) * 0.01
    small = big[:10].copy()
    small[:, 0] += 0.05
    np.save(d / "gauss_0000.npy", big)
    np.save(d / "gauss_0001.npy", small)
    c = Console(record=True, width=200)
    log_wobble_motion_from_gauss_npy(tmp_path, n_obj=100, console=c, sample_frames=(0, 1))
    out = c.export_text()
    assert "distinct_row_counts" in out
    assert "truncated to first 10 rows" in out
    assert "analysis-only" in out
