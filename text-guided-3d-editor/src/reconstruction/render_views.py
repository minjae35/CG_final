"""Render RGB + metric depth + camera metadata from trained 3DGS (PRD v2 §1.4)."""
from __future__ import annotations

import sys
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from config import PROJECT_ROOT


def _ensure_gs_path() -> Path:
    gs = PROJECT_ROOT / "submodules" / "gaussian-splatting"
    if not gs.is_dir():
        raise FileNotFoundError(gs)
    p = str(gs.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    return gs


def _load_render_args(model_path: Path, source_path: Path, iteration: int, images: str | None = None):
    _ensure_gs_path()
    old_argv = sys.argv
    try:
        sys.argv = [
            "render.py",
            "-m",
            str(model_path),
            "-s",
            str(source_path),
            "--iteration",
            str(iteration),
            "--skip_test",
            "--quiet",
        ]
        if images:
            sys.argv.extend(["--images", images])
        from arguments import ModelParams, PipelineParams, get_combined_args  # type: ignore

        parser = ArgumentParser()
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        args = get_combined_args(parser)
        return model.extract(args), pipeline.extract(args), args
    finally:
        sys.argv = old_argv


def _resolve_source_path(source_path: Path, model_path: Path) -> tuple[Path, str | None]:
    """Use the trained 3DGS cfg_args source when the configured path is stale."""
    if (source_path / "sparse").exists() or (source_path / "transforms_train.json").exists():
        source = source_path
    else:
        source = source_path
        cfg_args = model_path / "cfg_args"
        if cfg_args.is_file():
            m = re.search(r"source_path='([^']+)'", cfg_args.read_text(encoding="utf-8"))
            if m:
                source = Path(m.group(1)).expanduser().resolve()

    images = None
    if not (source / "images").exists() and (source / "images_8").exists():
        images = "images_8"
    return source, images


def render_training_views(
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    out_dir: Path | str,
    view_stride: int = 1,
) -> Path:
    """
    For each training camera, run the 3DGS rasterizer to produce:
      - ``rgb_{i}.png`` (uint8)
      - ``depth_{i}.npy`` — camera-space metric depth *z* (meters); from inverse depth output
      - ``alpha_{i}.png`` — rough visibility from valid depth
      - ``cam_meta_{i}.npz`` — ``K`` (3,3), ``world_view_transform`` (4,4) same tensor as
        ``Camera.world_view_transform`` / rasterizer (not raw COLMAP ``R``|``t`` blocks), plus ``w2c_layout`` tag

    ``view_stride`` subsamples training cameras (1 = all).
    """
    colmap_scene = Path(colmap_scene).resolve()
    model_path = Path(model_path).resolve()
    colmap_scene, images = _resolve_source_path(colmap_scene, model_path)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("3DGS rendering requires CUDA.")

    _ensure_gs_path()
    from gaussian_renderer import render as gs_render  # type: ignore
    from scene.gaussian_model import GaussianModel  # type: ignore
    from scene import Scene  # type: ignore
    from utils.graphics_utils import fov2focal  # type: ignore

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # type: ignore

        separate_sh = True
    except ImportError:
        separate_sh = False

    dataset, pipe, _args = _load_render_args(model_path, colmap_scene, iteration, images=images)

    print(
        "[render_training_views] Loading GaussianModel + Scene (large checkpoints may take a while) ...",
        flush=True,
    )
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        views = scene.getTrainCameras(1.0)
        if view_stride > 1:
            views = views[::view_stride]

        n_views = len(views)
        print(
            f"[render_training_views] iteration={iteration} cameras={n_views} "
            f"stride={view_stride} -> {out_dir}",
            flush=True,
        )
        for idx, view in enumerate(views):
            iname = getattr(view, "image_name", str(idx))
            print(
                f"[render_training_views] {idx + 1}/{n_views}  {iname}  ({view.image_width}x{view.image_height}) …",
                flush=True,
            )
            pkg = gs_render(
                view,
                gaussians,
                pipe,
                background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=separate_sh,
            )
            image = pkg["render"].clamp(0.0, 1.0)
            if dataset.train_test_exp:
                image = image[..., image.shape[-1] // 2 :]

            inv_depth = pkg["depth"]
            if dataset.train_test_exp:
                inv_depth = inv_depth[..., inv_depth.shape[-1] // 2 :]

            inv_np = inv_depth.detach().cpu().float().numpy().squeeze()
            z = np.zeros_like(inv_np, dtype=np.float32)
            valid = inv_np > 1e-6
            z[valid] = 1.0 / inv_np[valid]

            stem = f"{idx:05d}"
            torchvision.utils.save_image(image, str(out_dir / f"rgb_{stem}.png"))

            np.save(out_dir / f"depth_{stem}.npy", z.astype(np.float32))

            alpha = np.zeros(z.shape, dtype=np.uint8)
            alpha[valid] = 255
            Image.fromarray(alpha, mode="L").save(out_dir / f"alpha_{stem}.png")

            W, H = int(view.image_width), int(view.image_height)
            fx = float(fov2focal(view.FoVx, W))
            fy = float(fov2focal(view.FoVy, H))
            cx = 0.5 * W
            cy = 0.5 * H
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            # Must match GaussianRasterizer: same ``world_view_transform`` tensor the renderer uses
            # (getWorld2View2 then .T), *not* raw COLMAP R|t blocks — see cuda_rasterizer transformPoint4x3.
            w2c = np.asarray(view.world_view_transform.detach().cpu().numpy(), dtype=np.float64)

            np.savez_compressed(
                out_dir / f"cam_meta_{stem}.npz",
                K=K,
                world_view_transform=w2c,
                w2c_layout=np.array("gs_world_view_transform_v1"),
                image_name=np.array(view.image_name),
            )

    return out_dir
