"""Single-frame RGB preview for a chosen training camera (mode-b video alignment check)."""
from __future__ import annotations

from pathlib import Path

import torch
import torchvision

from rendering.gs_mesh_composite import build_gs_mesh_backdrop


def save_render_camera_rgb_png(
    *,
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    scene_ply: Path | str,
    camera_index: int,
    out_path: Path,
) -> None:
    """Baseline 3DGS render (no selection tint) for ``camera_index`` → ``out_path``."""
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # type: ignore

        separate_sh = True
    except ImportError:
        separate_sh = False
    from gaussian_renderer import render as gs_render  # type: ignore

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bd = build_gs_mesh_backdrop(
        colmap_scene=colmap_scene,
        model_path=model_path,
        iteration=iteration,
        base_ply=scene_ply,
        camera_index=int(camera_index),
    )
    with torch.no_grad():
        pkg = gs_render(
            bd.view,
            bd.gaussians,
            bd.pipe,
            bd.background,
            use_trained_exp=bd.dataset.train_test_exp,
            separate_sh=separate_sh,
        )
    image = pkg["render"].clamp(0.0, 1.0)
    if bd.dataset.train_test_exp:
        image = image[..., image.shape[-1] // 2 :]
    torchvision.utils.save_image(image, str(out_path))
    print(
        f"[mode-b] wrote render preview [cyan]{out_path}[/]  camera_index={int(camera_index)}  "
        f"image_name={getattr(bd.view, 'image_name', '?')}",
        flush=True,
    )
