"""PSNR, SSIM, LPIPS (PRD 5.1)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def compute_metrics_row(
    pred_path: Path | str,
    ref_path: Path | str,
    lpips_net: str = "alex",
) -> dict[str, float]:
    pred = _load_rgb(Path(pred_path))
    ref = _load_rgb(Path(ref_path))
    if pred.shape != ref.shape:
        from scipy.ndimage import zoom

        z = ref.shape[0] / pred.shape[0]
        pred = zoom(pred, (z, z, 1), order=1)

    psnr = float(peak_signal_noise_ratio(ref, pred, data_range=1.0))
    ssim = float(structural_similarity(ref, pred, channel_axis=2, data_range=1.0))
    lpips_val = 0.0
    try:
        import torch

        import lpips as lpips_mod

        loss_fn = lpips_mod.LPIPS(net=lpips_net)
        t0 = torch.from_numpy(pred).permute(2, 0, 1).float().unsqueeze(0) * 2 - 1
        t1 = torch.from_numpy(ref).permute(2, 0, 1).float().unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_val = float(loss_fn(t0, t1).item())
    except Exception:
        pass

    t0 = time.perf_counter()
    _ = pred + ref
    dt = time.perf_counter() - t0
    fps = 1.0 / (dt + 1e-9)
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val, "fps": fps}


def write_metrics_csv(rows: list[dict], out_csv: Path | str) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
