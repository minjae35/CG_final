"""Approximate visibility of Gaussian centers in a 3DGS training camera."""
from __future__ import annotations

import torch


def count_selected_visible_in_view(
    view: object,
    xyz: torch.Tensor,
    selected_idx: torch.Tensor,
    *,
    ndc_margin: float = 1.02,
) -> int:
    """Count selected Gaussians whose world centers project inside the image frustum (clip-space heuristic)."""
    if selected_idx.numel() == 0:
        return 0
    idx = selected_idx[(selected_idx >= 0) & (selected_idx < xyz.shape[0])]
    if idx.numel() == 0:
        return 0
    pts = xyz[idx]
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    homo = torch.cat([pts, ones], dim=1)
    fpt = view.full_proj_transform
    clip = homo @ fpt
    w = clip[:, 3]
    ok = w.abs() > 1e-5
    invw = 1.0 / w.clamp(min=1e-5)
    ndc_x = clip[:, 0] * invw
    ndc_y = clip[:, 1] * invw
    ndc_z = clip[:, 2] * invw
    vis = ok & (ndc_x.abs() <= ndc_margin) & (ndc_y.abs() <= ndc_margin) & (ndc_z.abs() <= ndc_margin)
    return int(vis.sum().item())
