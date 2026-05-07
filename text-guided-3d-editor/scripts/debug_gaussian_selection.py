#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from plyfile import PlyData

from config import PROJECT_ROOT, PipelineConfig
from segmentation.mask_to_gaussians import mask_to_gaussian_indices
from segmentation.text_to_mask import text_to_mask
from segmentation.multi_view_consensus import consensus_indices


def _load_positions_from_ply(ply_path: Path) -> np.ndarray:
    gply = PlyData.read(str(ply_path))
    v = gply["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)


def _project_points(K: np.ndarray, w2c: np.ndarray, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixel coordinates. Returns (uv, z)."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    Xc = (R @ Xw.T + t.reshape(3, 1)).T  # (N,3)
    z = Xc[:, 2]
    valid = z > 1e-6
    Xc = Xc[valid]
    z = z[valid]
    uvw = (K @ Xc.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, z


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recompute text->mask->gaussian selection and visualize by projecting selected Gaussians back to RGB views."
    )
    ap.add_argument("--text", required=True, help='Text prompt (e.g. "a desk")')
    ap.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "pipeline_config.yaml"),
        help="Pipeline config yaml path.",
    )
    ap.add_argument("--smoke", action="store_true", help="Use training_iterations_low (like pipeline --smoke).")
    ap.add_argument("--render-dir", default=None, help="Override render dir (default: output/renders_views).")
    ap.add_argument("--iteration", type=int, default=None, help="3DGS iteration (default: from --smoke + config).")
    ap.add_argument("--views", type=int, default=None, help="Number of views to use (default: config.segmentation.multi_view_count).")
    ap.add_argument("--device", default=None, help="Segmentation device: cuda or cpu (default: config).")
    ap.add_argument("--stride", type=int, default=2, help="Stride for mask pixel sampling (matches pipeline default=2).")
    ap.add_argument("--dist", type=float, default=0.2, help="3D nearest-neighbor distance threshold (matches pipeline default=0.2).")
    ap.add_argument("--max-points", type=int, default=3000, help="Max projected points per view for visualization.")
    ap.add_argument("--out-dir", default="output/selection_debug", help="Output directory for debug artifacts.")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    iters = args.iteration if args.iteration is not None else (
        cfg.scene.training_iterations_low if args.smoke else cfg.scene.training_iterations
    )

    render_dir = Path(args.render_dir) if args.render_dir else cfg.resolve("output/renders_views")
    if not render_dir.is_dir():
        raise FileNotFoundError(f"render_dir not found: {render_dir}")

    model_out = cfg.scene_model_out(args.smoke)
    ply_path = model_out / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    pos = _load_positions_from_ply(ply_path)

    text = args.text.strip()
    view_count = int(args.views) if args.views is not None else int(cfg.segmentation.multi_view_count)
    seg_device = args.device if args.device is not None else cfg.segmentation.segmentation_device

    rgbs = sorted(render_dir.glob("rgb_*.png"))[:view_count]
    if not rgbs:
        raise FileNotFoundError(f"No rgb_*.png in {render_dir}")

    sets_local: list[np.ndarray] = []
    per_view: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for rgb in rgbs:
        stem = rgb.stem.replace("rgb_", "")
        meta_p = render_dir / f"cam_meta_{stem}.npz"
        depth_p = render_dir / f"depth_{stem}.npy"
        if not meta_p.is_file() or not depth_p.is_file():
            continue

        meta = np.load(meta_p)
        K = np.asarray(meta["K"], dtype=np.float64)
        w2c = np.asarray(meta["world_view_transform"], dtype=np.float64)
        depth = np.load(depth_p)

        mask = text_to_mask(
            str(rgb),
            text,
            cfg.segmentation.box_threshold,
            cfg.segmentation.text_threshold,
            sam2_checkpoint=cfg.segmentation.sam2_checkpoint,
            sam2_config=cfg.segmentation.sam2_config,
            grounding_dino_checkpoint=cfg.segmentation.grounding_dino_checkpoint,
            segmentation_device=seg_device,
        )
        idx_local = mask_to_gaussian_indices(
            mask,
            depth,
            K,
            w2c,
            pos,
            distance_threshold=float(args.dist),
            stride=int(args.stride),
        )
        sets_local.append(idx_local)
        per_view.append((rgb, K, w2c))

    min_votes = 1 if args.smoke else int(cfg.segmentation.consensus_min_votes)
    idx = consensus_indices(sets_local, min_votes=min_votes)

    out_dir = cfg.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "selected_indices.npy", idx.astype(np.int64))

    # Basic bbox sanity.
    if len(idx):
        pts = pos[idx]
        lo = pts.min(0)
        hi = pts.max(0)
        (out_dir / "selected_bbox.txt").write_text(
            f"count={len(idx)}\n"
            f"min={lo.tolist()}\n"
            f"max={hi.tolist()}\n",
            encoding="utf-8",
        )
    else:
        (out_dir / "selected_bbox.txt").write_text("count=0\n", encoding="utf-8")

    # Visualize by projecting selected points onto each RGB.
    red = ImageColor.getrgb("#ff2d2d")
    for rgb, K, w2c in per_view:
        img = Image.open(rgb).convert("RGB")
        W, H = img.size

        if len(idx) == 0:
            img.save(out_dir / f"proj_{rgb.name}")
            continue

        uv, _z = _project_points(K, w2c, pos[idx])
        # Keep points inside image
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)]
        if uv.shape[0] > args.max_points:
            sel = np.random.default_rng(0).choice(uv.shape[0], size=int(args.max_points), replace=False)
            uv = uv[sel]

        draw = ImageDraw.Draw(img)
        for x, y in uv:
            xi, yi = int(round(float(x))), int(round(float(y)))
            r = 1
            draw.ellipse((xi - r, yi - r, xi + r, yi + r), fill=red)

        img.save(out_dir / f"proj_{rgb.name}")

    print(f"saved: {out_dir}/selected_indices.npy")
    print(f"saved: {out_dir}/selected_bbox.txt")
    print(f"saved: {out_dir}/proj_rgb_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

