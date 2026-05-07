#!/usr/bin/env python3
"""
Overlay sparse projected Gaussian centers (from 3DGS PLY) on existing train renders.

Uses the same camera convention as 3DGS / mask_to_gaussians (world_view_transform row-homogeneous).
No generative models — OpenCV + plyfile + PyTorch Scene for exact train cameras.

Example:
  cd text-guided-3d-editor && export PYTHONPATH=src
  python scripts/make_point_overlay_slides.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

# repo root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentation.mask_to_gaussians import project_world_to_pixels  # noqa: E402


def _ensure_gs_path() -> None:
    gs = PROJECT_ROOT / "submodules" / "gaussian-splatting"
    p = str(gs.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _resolve_source_path(source_path: Path, model_path: Path) -> tuple[Path, str | None]:
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


def _load_render_args(model_path: Path, source_path: Path, iteration: int, images: str | None):
    from argparse import ArgumentParser

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


def load_xyz_from_gaussian_ply(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)


def intrinsics_from_view(view) -> np.ndarray:
    from utils.graphics_utils import fov2focal  # type: ignore

    W, H = int(view.image_width), int(view.image_height)
    fx = float(fov2focal(view.FoVx, W))
    fy = float(fov2focal(view.FoVy, H))
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def score_render_png(path: Path) -> float:
    im = cv2.imread(str(path))
    if im is None:
        return -1.0
    x = im.astype(np.float32)
    return float(x.mean() * 0.55 + x.std() * 0.35)


def pick_view_indices(renders_dir: Path, n: int) -> list[int]:
    paths = sorted(renders_dir.glob("*.png"))
    scored = [(score_render_png(p), int(p.stem)) for p in paths]
    scored = [(s, i) for s, i in scored if s >= 0]
    scored.sort(key=lambda t: t[0], reverse=True)
    if len(scored) < n:
        return [i for _, i in scored]
    # spread across high-scoring list for viewpoint diversity
    picks: list[int] = []
    L = len(scored)
    if n <= 1:
        positions = [0]
    else:
        positions = [int(round(j * (L - 1) / (n - 1))) for j in range(n)]
    for pos in positions:
        if pos < L:
            picks.append(scored[pos][1])
    # dedupe and fill
    out: list[int] = []
    for i in picks:
        if i not in out:
            out.append(i)
    k = 0
    while len(out) < n and k < L:
        _, idx = scored[k]
        if idx not in out:
            out.append(idx)
        k += 1
    return out[:n]


def draw_caption_bar(img_bgr: np.ndarray, caption_lines: list[str]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar_h = 110
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:h, :] = img_bgr
    out[h:, :] = (32, 36, 40)
    y = h + 28
    for li, line in enumerate(caption_lines):
        col = (240, 240, 240) if li == 0 else (200, 210, 220)
        fs = 0.65 if li == 0 else 0.55
        thick = 1
        cv2.putText(out, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, thick, cv2.LINE_AA)
        y += 34
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "output" / "base_scene_full")
    ap.add_argument("--ply", type=Path, default=None, help="Gaussian PLY (default: iteration_30000 under model)")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--renders-dir", type=Path, default=None)
    ap.add_argument("--max-points", type=int, default=18_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-views", type=int, default=4, help="How many overlay PNGs to write (point_overlay_view_1..N).")
    ap.add_argument(
        "--view-indices",
        type=str,
        default="",
        help="Comma-separated train indices (overrides auto-pick). Count must match --num-views.",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "output" / "presentation")
    args = ap.parse_args()

    model_path = args.model_path.resolve()
    ply_path = args.ply or (model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")
    if not ply_path.is_file():
        raise SystemExit(f"Missing PLY: {ply_path}")

    renders_dir = args.renders_dir or (model_path / "train" / f"ours_{args.iteration}" / "renders")
    if not renders_dir.is_dir():
        raise SystemExit(f"Missing renders dir: {renders_dir}")

    caption = [
        "Base 3DGS scene: output/base_scene_full/point_cloud/iteration_30000/point_cloud.ply",
    ]

    _ensure_gs_path()
    import torch
    from gaussian_renderer import GaussianModel  # type: ignore
    from scene import Scene  # type: ignore

    colmap_scene, images = _resolve_source_path(
        PROJECT_ROOT / "data" / "mipnerf360" / "room",
        model_path,
    )
    cfg = model_path / "cfg_args"
    if cfg.is_file():
        m = re.search(r"source_path='([^']+)'", cfg.read_text(encoding="utf-8"))
        if m:
            colmap_scene = Path(m.group(1)).expanduser().resolve()

    dataset, pipe, _ra = _load_render_args(model_path, colmap_scene, args.iteration, images=images)

    print("[overlay] Loading Scene + cameras (CUDA) …", flush=True)
    xyz_cpu = load_xyz_from_gaussian_ply(ply_path)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        train_views = scene.getTrainCameras(1.0)

    n_train = len(train_views)
    print(f"[overlay] Train cameras: {n_train}  PLY vertices: {len(xyz_cpu)}", flush=True)

    n_views = max(1, int(args.num_views))
    if args.view_indices.strip():
        indices = [int(x.strip()) for x in args.view_indices.split(",") if x.strip()]
        if len(indices) != n_views:
            raise SystemExit(f"--view-indices must list exactly {n_views} integers (same as --num-views)")
    else:
        indices = pick_view_indices(renders_dir, n_views)
    print(f"[overlay] Train view indices: {indices}", flush=True)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for out_i, view_idx in enumerate(indices, start=1):
        if view_idx < 0 or view_idx >= n_train:
            raise SystemExit(f"View index {view_idx} out of range [0, {n_train})")
        view = train_views[view_idx]
        W, H = int(view.image_width), int(view.image_height)
        K = intrinsics_from_view(view)
        Wm = np.asarray(view.world_view_transform.detach().cpu().numpy(), dtype=np.float64)

        png = renders_dir / f"{view_idx:05d}.png"
        if not png.is_file():
            raise SystemExit(f"Missing render {png}")
        img = cv2.imread(str(png))
        if img is None:
            raise SystemExit(f"Could not read {png}")
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        u, v, z = project_world_to_pixels(K, Wm, xyz_cpu)
        inb = (
            (z > 1e-4)
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0)
            & (u < W)
            & (v >= 0)
            & (v < H)
        )
        idx_vis = np.flatnonzero(inb)
        if idx_vis.size > args.max_points:
            idx_vis = rng.choice(idx_vis, size=args.max_points, replace=False)
        uu = u[idx_vis]
        vv = v[idx_vis]

        # green Gaussian centers (BGR)
        color = (0, 255, 0)
        for px, py in zip(uu, vv):
            xi, yi = int(round(px)), int(round(py))
            cv2.circle(img, (xi, yi), 2, color, -1, lineType=cv2.LINE_AA)

        sub = f"train view index {view_idx}  |  {len(idx_vis)} Gaussians drawn  |  render: {png.name}"
        out_img = draw_caption_bar(img, caption + [sub])
        out_path = out_dir / f"point_overlay_view_{out_i}.png"
        cv2.imwrite(str(out_path), out_img)
        print(f"[overlay] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
