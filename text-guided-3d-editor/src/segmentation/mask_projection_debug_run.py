"""Single-view mask → Gaussian projection debug (shared by mode-b-selection-debug and mode-b --debug-mask-to-gaussians-only)."""
from __future__ import annotations

import gc
import shutil
from pathlib import Path

import numpy as np
import typer
from plyfile import PlyData
from rich.console import Console

from config import PipelineConfig

from physics.mode_b_selection import save_subset_points_ply, selection_xyz_bounds
from reconstruction.render_views import render_training_views
from segmentation.mask_to_gaussians import (
    indices_project_inside_mask,
    mask_to_gaussian_indices,
    save_gaussian_projection_debug_image,
    save_subsampled_all_gaussian_projections,
    world_view_transform_apply,
)
from segmentation.mode_b_filters import filter_indices_mask_and_rendered_depth
from segmentation.debug_overlay_style import DEBUG_GREEN_BGR, apply_debug_green_mask_overlay
from segmentation.text_to_mask import text_to_mask


def run_mask_projection_debug(
    *,
    cfg: PipelineConfig,
    text: str,
    smoke: bool,
    smoke_3dgs: bool,
    force_rerender_views: bool,
    debug_mask_to_gaussians_only: bool,
    projection_stride: int,
    console: Console,
    maybe_cuda_empty,
) -> Path:
    import time

    import cv2

    t_run = time.monotonic()

    def _lap(msg: str) -> None:
        print(f"[mask-debug] {msg}  (elapsed {time.monotonic() - t_run:.1f}s)", flush=True)

    _lap("시작: PLY 로드 · 렌더 · SAM2 · 투영 순으로 진행 (중간에 수 분 걸릴 수 있음)")

    colmap_scene = cfg.resolve(cfg.scene.data_root) / cfg.scene.scene_name
    model_out = cfg.resolve(cfg.scene.model_output)
    use_smoke = smoke or smoke_3dgs
    iters = cfg.scene.training_iterations_low if use_smoke else cfg.scene.training_iterations
    ply_path = model_out / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise typer.BadParameter(f"Missing {ply_path}; run ``train`` (or ``train --smoke``) first.")

    mode_b_out = cfg.resolve(cfg.paths.sim_output) / "mode_b_jelly"
    out = mode_b_out / "debug_selection"
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    gply = PlyData.read(str(ply_path))
    v = gply["vertex"]
    pos = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    n_gaussians = int(pos.shape[0])
    _lap(f"PLY 읽기 완료  n_gaussians={n_gaussians}")

    render_dir = cfg.resolve("output/renders_views")
    seg = cfg.segmentation
    rgbs = sorted(render_dir.glob("rgb_*.png"))
    if not rgbs or force_rerender_views:
        _lap(
            "training view 렌더 시작 (3DGS Scene 로드 + 카메라별 CUDA 렌더; view_stride에 따라 카메라 수 결정)"
        )
        render_training_views(
            colmap_scene,
            model_out,
            iters,
            render_dir,
            view_stride=cfg.reconstruction.render_view_stride,
        )
        rgbs = sorted(render_dir.glob("rgb_*.png"))
        _lap(f"렌더 끝  rgb 파일 {len(rgbs)}개")
    else:
        _lap(f"기존 렌더 재사용 ({len(rgbs)} rgb). cam_meta를 새로 쓰려면 --force-rerender-views")
    if not rgbs:
        raise typer.BadParameter(f"No rgb_*.png under {render_dir}")

    rgb_path = rgbs[0]
    stem = rgb_path.stem.replace("rgb_", "")
    shutil.copy2(rgb_path, out / "01_rgb.png")

    vs = int(cfg.reconstruction.render_view_stride)
    try:
        local_render_idx = int(stem)
    except ValueError:
        local_render_idx = 0
    training_camera_index = local_render_idx * vs
    (out / "best_camera_index.txt").write_text(f"{training_camera_index}\n", encoding="utf-8")
    (out / "selection_view_meta.txt").write_text(
        f"rgb_file={rgb_path.name}\n"
        f"local_render_enumerate_index={local_render_idx}\n"
        f"render_view_stride={vs}\n"
        f"training_camera_index={training_camera_index}\n"
        "(index into Scene.getTrainCameras(1.0) full list; matches kinematic backdrop when stride aligns)\n",
        encoding="utf-8",
    )

    _lap("Grounded SAM2 + DINO (모델 로딩에 수십 초~분 단위 가능) …")
    mask = text_to_mask(
        str(rgb_path),
        text,
        seg.box_threshold,
        seg.text_threshold,
        gdino_config=seg.grounding_dino_config or None,
        gdino_checkpoint=seg.grounding_dino_checkpoint or None,
        sam2_config=seg.sam2_config or None,
        sam2_checkpoint=seg.sam2_checkpoint or None,
        device=seg.segmentation_device,
        debug_mask_dir=None,
        log_progress=True,
    )
    _lap(f"세그멘테이션 마스크 완료  true_pixels={int(mask.sum())}")
    if int(seg.mask_erode_iters) > 0 and not debug_mask_to_gaussians_only:
        from scipy.ndimage import binary_erosion

        mask = binary_erosion(mask, iterations=int(seg.mask_erode_iters))

    img_bgr = cv2.imread(str(rgb_path))
    if img_bgr is None:
        raise FileNotFoundError(rgb_path)
    ih, iw = img_bgr.shape[0], img_bgr.shape[1]
    H, W = mask.shape
    if (H, W) != (ih, iw):
        mask = (
            cv2.resize(mask.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
        )
        H, W = mask.shape
    m_u8 = (mask.astype(np.uint8) * 255).reshape(H, W)
    blend = apply_debug_green_mask_overlay(img_bgr, m_u8)
    cv2.imwrite(str(out / "02_mask_overlay.png"), blend)
    cv2.imwrite(str(out / "03_mask_binary.png"), m_u8)
    n_mask_px = int(mask.sum())
    console.print(f"[mask-debug] mask true pixels: [bold]{n_mask_px}[/]")

    meta = np.load(render_dir / f"cam_meta_{stem}.npz")
    K = np.asarray(meta["K"], dtype=np.float64)
    w2c = np.asarray(meta["world_view_transform"], dtype=np.float64)
    depth = np.load(render_dir / f"depth_{stem}.npy")

    if "w2c_layout" in meta.files:
        console.print(f"[mask-debug] cam_meta w2c_layout={meta['w2c_layout'].item()}")
    else:
        console.print(
            "[yellow][mask-debug] old cam_meta (no w2c_layout). Re-run with --force-rerender-views after upgrading.[/]"
        )

    console.print(
        f"[mask-debug] scene PLY: [cyan]{ply_path}[/]  n_vertices={n_gaussians}  "
        f"model_out=[cyan]{model_out}[/] iteration={iters}"
    )
    console.print(
        f"[mask-debug] RGB path [cyan]{rgb_path.name}[/]  image (H,W)=({ih},{iw})  "
        f"mask (H,W)=({H},{W})  depth.shape={depth.shape}"
    )
    console.print(
        f"[mask-debug] Intrinsics fx={K[0, 0]:.5f} fy={K[1, 1]:.5f} cx={K[0, 2]:.5f} cy={K[1, 2]:.5f}  "
        "pixels are u=column, v=row (OpenCV); mask indexed mask[v,u]."
    )
    console.print(
        "[mask-debug] world_view_transform matches 3DGS ``view.world_view_transform`` "
        "(row-vector: Xc_h = Xw_h @ W), same as CUDA rasterizer."
    )

    nearest_stride = 1 if debug_mask_to_gaussians_only else int(seg.mask_stride)
    stats_n: dict[str, int | float] = {}
    stats_d: dict[str, int | float] = {}

    _lap(f"마스크→Gaussian nearest (stride={nearest_stride}, kNN) …")
    idx_nearest = mask_to_gaussian_indices(
        mask,
        depth,
        K,
        w2c,
        pos,
        distance_threshold=float(seg.mask_3d_distance_threshold_m),
        stride=nearest_stride,
        use_depth_consistency=False,
        stats=stats_n,
    )
    console.print(f"[mask-debug] nearest path unique indices: [bold]{len(idx_nearest)}[/]")
    for k, v in stats_n.items():
        console.print(f"    stats[{k}]={v}")

    use_depth = bool(seg.mask_use_depth_consistency) and not debug_mask_to_gaussians_only
    if use_depth:
        idx_depth = mask_to_gaussian_indices(
            mask,
            depth,
            K,
            w2c,
            pos,
            distance_threshold=float(seg.mask_3d_distance_threshold_m),
            stride=int(seg.mask_stride),
            use_depth_consistency=True,
            depth_tolerance_abs_m=float(seg.mask_depth_tolerance_abs_m),
            depth_tolerance_rel=float(seg.mask_depth_tolerance_rel),
            knn=int(seg.mask_knn),
            pixel_tolerance_px=float(seg.mask_pixel_tolerance_px),
            min_votes=int(seg.mask_min_votes),
            stats=stats_d,
        )
    else:
        idx_depth = idx_nearest.astype(np.int64)
        stats_d.update(stats_n)
    console.print(f"[mask-debug] depth-consistency path unique indices: [bold]{len(idx_depth)}[/]")
    for k, v in stats_d.items():
        console.print(f"    stats[{k}]={v}")

    ps = max(1, int(projection_stride))
    _lap(f"마스크 안 투영만 (전체 Gaussian 스캔, subsample={ps}; 개수 많으면 수십 초) …")
    idx_proj = indices_project_inside_mask(mask, pos, K, w2c, subsample=ps)
    _lap("투영-only 인덱스 계산 완료")
    console.print(
        f"[mask-debug] projection_inside_mask only (no 3D kNN/depth), subsample={ps}: [bold]{len(idx_proj)}[/]"
    )

    console.print("[mask-debug] shell / connected-component: [dim]N/A (not run in this debug)[/]")

    if debug_mask_to_gaussians_only:
        idx_final = idx_proj.astype(np.int64)
        console.print("[mask-debug] --debug-mask-to-gaussians-only: final indices = projection-only set.")
    else:
        strict_abs = min(0.025, float(seg.mask_depth_tolerance_abs_m))
        strict_rel = min(0.022, float(seg.mask_depth_tolerance_rel))
        idx_final = filter_indices_mask_and_rendered_depth(
            pos,
            idx_depth,
            K,
            w2c,
            mask,
            depth,
            depth_tolerance_abs_m=strict_abs,
            depth_tolerance_rel=strict_rel,
        )
        if idx_final.size == 0 and idx_depth.size > 0:
            console.print(
                "[yellow]Strict mask+depth gate removed all Gaussians; using depth-consistency set only.[/]"
            )
            idx_final = idx_depth.astype(np.int64)
    console.print(f"[mask-debug] final selected count: [bold]{len(idx_final)}[/]")

    save_subset_points_ply(pos, idx_nearest, out / "04_after_mask_projection.ply", rgb=(0.25, 0.85, 0.95))
    save_subset_points_ply(pos, idx_proj, out / "04_projection_no_depth_filter.ply", rgb=(0.2, 0.95, 0.35))
    save_subset_points_ply(pos, idx_depth, out / "05_after_depth_filter.ply", rgb=(0.95, 0.85, 0.2))
    save_subset_points_ply(pos, idx_final, out / "06_final_selected_gaussians.ply", rgb=(0.95, 0.25, 0.15))
    np.save(out / "selected_indices.npy", idx_final.astype(np.int64))

    _lap("04a 전체 가우시안 투영 썸네일 그리는 중 …")
    drawn = save_subsampled_all_gaussian_projections(
        rgb_path,
        out / "04a_all_projected_gaussian_centers.png",
        pos,
        K,
        w2c,
        max_points=100_000,
        color_bgr=DEBUG_GREEN_BGR,
        radius=1,
    )
    console.print(f"[mask-debug] 04a drew ~{drawn} subsampled centres (cap 100k).")

    save_gaussian_projection_debug_image(
        str(rgb_path),
        out / "04b_mask_candidate_centers.png",
        pos,
        idx_nearest,
        K,
        w2c,
        color_bgr=DEBUG_GREEN_BGR,
        radius=3,
    )
    save_gaussian_projection_debug_image(
        str(rgb_path),
        out / "04c_depth_consistent_centers.png",
        pos,
        idx_depth,
        K,
        w2c,
        color_bgr=DEBUG_GREEN_BGR,
        radius=3,
    )

    lines = [
        "mask projection debug (first rgb_*.png in render_dir)",
        f"view_stem={stem}  rgb={rgb_path.name}",
        f"model_output={model_out}",
        f"ply_checkpoint={ply_path}",
        f"n_gaussian_vertices={n_gaussians}",
        f"iteration={iters}",
        "",
        "geometry",
        f"  rgb_hw=({ih},{iw})  mask_hw=({H},{W})  depth_hw={tuple(depth.shape)}",
        f"  K fx,fy,cx,cy = {float(K[0,0]):.6f},{float(K[1,1]):.6f},{float(K[0,2]):.6f},{float(K[1,2]):.6f}",
        "  pixel coords: u=column (x), v=row (y); depth z in camera units (see render_views).",
        "  world_view_transform: row-homogeneous Xc_h = Xw_h @ W (3DGS / CUDA same as cam_meta).",
        "",
        "kinematic / mode-b camera alignment",
        f"  best_camera_index.txt -> training_camera_index={training_camera_index}",
        f"  (= local_render_idx {local_render_idx} * render_view_stride {vs}; matches first rgb_* in render_dir)",
        "",
        f"mask_true_pixels={n_mask_px}",
        f"projection_inside_mask_subsample={ps} count={len(idx_proj)}",
        "",
        "nearest_knn_stats:",
    ]
    for k, v in stats_n.items():
        lines.append(f"  {k}={v}")
    lines.append("")
    lines.append("depth_consistency_stats:")
    for k, v in stats_d.items():
        lines.append(f"  {k}={v}")
    lines.extend(
        [
            "",
            "Gaussian counts:",
            f"  04_after_mask_projection (kNN to unprojected mask rays): {len(idx_nearest)}",
            f"  04_projection_no_depth_filter (centre inside mask, subsample={ps}): {len(idx_proj)}",
            f"  05_after_depth_filter: {len(idx_depth)}",
            f"  06_final_selected: {len(idx_final)}",
            "",
            "shell / CC: not used in this debug bundle.",
            "",
            "xyz bounds (final):",
        ]
    )
    if idx_final.size:
        lo, hi = selection_xyz_bounds(pos, idx_final)
        lines.append(f"  min={lo.tolist()}")
        lines.append(f"  max={hi.tolist()}")
    else:
        lines.append("  (empty)")

    lines.extend(
        [
            "",
            "Manual checks:",
            "  - 04_projection_no_depth_filter.ply should be visible if projection+indices are sane.",
            "  - If 04 is empty but 04a shows a cloud aligned with the room, raise mask_3d_distance_threshold_m.",
        ]
    )
    (out / "selection_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    console.print(f"[bold]mask-debug[/] wrote [cyan]{out}[/]")
    console.print(f"  summary: [cyan]{out / 'selection_summary.txt'}[/]")

    try:
        save_gaussian_projection_debug_image(
            str(rgb_path),
            out / "07_selected_red_overlay.png",
            pos,
            idx_final,
            K,
            w2c,
            color_bgr=(0, 0, 255),
            radius=3,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]07 overlay skipped: {exc}[/]")

    black = np.zeros_like(img_bgr)
    pos64 = pos.astype(np.float64)
    if idx_final.size:
        Xc = world_view_transform_apply(w2c, pos64[idx_final])
        z = Xc[:, 2]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        valid = z > 1e-4
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[valid] = fx * Xc[valid, 0] / z[valid] + cx
        v[valid] = fy * Xc[valid, 1] / z[valid] + cy
        ui = np.floor(u[valid] + 0.5).astype(np.int32)
        vi = np.floor(v[valid] + 0.5).astype(np.int32)
        for uu, vv in zip(ui.tolist(), vi.tolist()):
            if 0 <= uu < img_bgr.shape[1] and 0 <= vv < img_bgr.shape[0]:
                cv2.circle(black, (uu, vv), 3, DEBUG_GREEN_BGR, thickness=-1)
    cv2.imwrite(str(out / "08_selected_only_black.png"), black)

    maybe_cuda_empty()
    gc.collect()
    _lap(f"완료 → {out}")
    return out
