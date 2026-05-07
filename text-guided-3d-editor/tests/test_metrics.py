from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.metrics import compute_metrics_row, write_metrics_csv


def test_metrics_identical(tmp_path: Path) -> None:
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    p = tmp_path / "x.png"
    img.save(p)
    row = compute_metrics_row(p, p)
    assert "psnr" in row and "ssim" in row


def test_write_csv(tmp_path: Path) -> None:
    write_metrics_csv([{"a": 1.0, "b": 2.0}], tmp_path / "o.csv")
    assert (tmp_path / "o.csv").read_text().strip().splitlines()[0] == "a,b"
