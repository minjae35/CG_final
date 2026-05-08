"""End-to-end orchestrator: Mode A / Mode B (PRD + plan)."""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import typer
from plyfile import PlyData
from rich.console import Console

# Run with: cd text-guided-3d-editor && PYTHONPATH=src python pipeline.py ...
from config import PROJECT_ROOT, PipelineConfig
from data_utils.dataset_downloader import download_and_prepare
from generation.mesh_placed_export import export_placed_dreamgaussian_mesh
from generation.object_rescaler import rescale_object_ply_to_scene
from generation.object_postprocess import (
    enhance_generation_prompt,
    is_red_duck_prompt,
    recolor_red_duck_ply,
    swap_red_yellow_duck_ply,
    wants_object_realism_preset,
)
from generation.sds_generator import generate_object, generate_object_from_image
from generation.image_generator import generate_reference_image
from generation.mesh_to_gaussian import (
    insertion_mesh_source_exists,
    mesh_to_gaussian_ply,
    resolve_insertion_mesh_path,
    stage_external_mesh_into_gen_dir,
)
from physics.config_generator import default_template_dir, generate_phys_config
from physics.mode_b_jelly import estimate_mode_b_support_contact_y
from physics.mode_b_selection import (
    filter_expanded_to_shell_around_surface,
    filter_indices_by_opacity_logit,
    largest_component_touching_seeds,
    read_ply_opacity_logits,
    save_subset_points_ply,
    selection_xyz_bounds,
)
from physics.mpm_preflight import log_phys_preflight, validate_merged_object_for_mpm
from physics.mpm_runner import run_simulation
from rendering.gs_mesh_composite import (
    composite_mesh_over_base_gs,
    run_mesh_rigid_overlay_sequence,
    write_png,
)
from placement.ply_merger import merge_ply_files
from placement.surface_aware import (
    camera_right_horizontal_world,
    estimate_floor_slopes_dy_dx_dz,
    find_camera_visible_surface_point,
    find_surface_height_from_ply,
    floor_downward_normal_from_slopes,
    refine_floor_contact_y,
    rotate_unit_vector_around_axis_deg,
)
from reconstruction.render_views import render_training_views
from rendering.kinematic_wobble_render import render_kinematic_wobble_scene_by_indices
from rendering.mode_b_debug_render import run_mode_b_selection_debug_suite
from rendering.mode_b_render_preview import save_render_camera_rgb_png
from reconstruction.train_3dgs import train_scene
from segmentation.mask_to_gaussians import (
    mask_to_gaussian_indices,
    save_gaussian_projection_debug_image,
)
from segmentation.multi_view_consensus import consensus_indices
from segmentation.object_volume import expand_indices_to_bbox_volume
from segmentation.text_to_mask import text_to_mask

app = typer.Typer(add_completion=False)
console = Console()

MODE_A_REVIEW_JSON = "mode_a_review.json"
# Canonical prompt for color/reference when --yellow-duck is set (demo may still say "red duck" in prose).
_YELLOW_RUBBER_DUCK_TEXT = "a yellow rubber duck"


def _read_training_cam_index_txt(p: Path) -> int | None:
    if not p.is_file():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip().split()[0])
    except (ValueError, IndexError):
        return None


def _save_mode_a_review_checkpoint(
    gen_dir: Path,
    *,
    text: str,
    yellow_duck: bool,
    scaled: Path,
    obj_ply: Path,
    place_x: float,
    place_y: float,
    place_z: float,
    floor_y: float,
    canonical_frame: str,
    scale_factor: float,
    post_rotate: tuple[float, float, float],
    opacity_filter_logit: float,
    contact_base_ao: float,
    image_mode: bool,
    raw_object: bool,
    mesh_render: bool,
    mesh_skin_k: int,
    swap_red_yellow: bool,
    mesh_gaussians: bool,
    align_y: np.ndarray | None,
) -> Path:
    path = gen_dir / MODE_A_REVIEW_JSON
    payload: dict = {
        "version": 1,
        "text": text,
        "yellow_duck": bool(yellow_duck),
        "scaled_path": str(Path(scaled).resolve()),
        "obj_ply": str(Path(obj_ply).resolve()),
        "place_x": float(place_x),
        "place_y": float(place_y),
        "place_z": float(place_z),
        "floor_y": float(floor_y),
        "canonical_frame": str(canonical_frame),
        "scale_factor": float(scale_factor),
        "post_rotate": [float(post_rotate[0]), float(post_rotate[1]), float(post_rotate[2])],
        "opacity_filter_logit": float(opacity_filter_logit),
        "contact_base_ao": float(contact_base_ao),
        "image_mode": bool(image_mode),
        "raw_object": bool(raw_object),
        "mesh_render": bool(mesh_render),
        "mesh_skin_k": int(mesh_skin_k),
        "swap_red_yellow": bool(swap_red_yellow),
        "mesh_gaussians": bool(mesh_gaussians),
        "align_y": None if align_y is None else [float(x) for x in np.asarray(align_y, dtype=np.float64).tolist()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_mode_a_review_checkpoint(gen_dir: Path) -> dict:
    path = gen_dir / MODE_A_REVIEW_JSON
    if not path.is_file():
        raise typer.BadParameter(
            f"Missing review checkpoint {path}. Run mode-a once with --hold-for-review first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_cuda_empty() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _log_cuda_memory_line(console: Console, label: str) -> None:
    """Best-effort VRAM summary before spawning PhysGaussian (see also nvidia-smi)."""
    try:
        import torch

        if not torch.cuda.is_available():
            console.print(f"[dim]{label}[/] CUDA 없음")
            return
        dev = torch.cuda.current_device()
        if hasattr(torch.cuda, "mem_get_info"):
            free_b, total_b = torch.cuda.mem_get_info(dev)
            console.print(
                f"[dim]{label}[/] CUDA 여유 [bold]{free_b // 1048576}[/] MiB / "
                f"전체 {total_b // 1048576} MiB  (자세히: [cyan]nvidia-smi[/])"
            )
        else:
            a = torch.cuda.memory_allocated(dev) // 1048576
            r = torch.cuda.memory_reserved(dev) // 1048576
            console.print(
                f"[dim]{label}[/] torch allocated≈{a} MiB  reserved≈{r} MiB  ([cyan]nvidia-smi[/])"
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[dim]{label}[/] GPU 메모리 조회 실패 ({exc}); [cyan]nvidia-smi[/] 권장")


@app.command()
def prepare_dataset(
    config: Path = typer.Option(PROJECT_ROOT / "configs" / "pipeline_config.yaml", "--config"),
) -> None:
    """Download and extract Mip-NeRF 360 data (PRD v2 task 1.2)."""
    cfg = PipelineConfig.load(config)
    p = download_and_prepare(
        cfg.scene.scene_name,
        PROJECT_ROOT,
        dataset_url=cfg.scene.dataset_url,
        data_root=cfg.scene.data_root,
    )
    console.print(f"Scene ready at [green]{p}[/] (images/ + sparse/0/)")


@app.command()
def train(
    config: Path = typer.Option(PROJECT_ROOT / "configs" / "pipeline_config.yaml", "--config"),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Train 3DGS to training_iterations_low; use the same flag on mode-a for that checkpoint.",
    ),
) -> None:
    cfg = PipelineConfig.load(config)
    colmap_scene = cfg.resolve(cfg.scene.data_root) / cfg.scene.scene_name
    model_out = cfg.resolve(cfg.scene.model_output)
    iters = cfg.scene.training_iterations_low if smoke else cfg.scene.training_iterations
    console.print(f"[bold]Training 3DGS[/] iters={iters}")
    ply = train_scene(colmap_scene, model_out, iterations=iters)
    console.print(f"PLY: [green]{ply}[/]")
    _maybe_cuda_empty()


@app.command()
def mode_b(
    text: str = typer.Argument(..., help="Object description"),
    config: Path = typer.Option(PROJECT_ROOT / "configs" / "pipeline_config.yaml", "--config"),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Low 3DGS iters + smaller mode-b MPM grid/shorter sim (see physics.n_grid_low, mode_b_mpm_smoke_frames).",
    ),
    smoke_3dgs: bool = typer.Option(
        False,
        "--smoke-3dgs",
        help="Low 3DGS checkpoint only; full physics grid (unlike --smoke alone).",
    ),
    reuse_selection: bool = typer.Option(
        True,
        "--reuse-selection/--no-reuse-selection",
        help="Reuse cached *surface* indices (surface_gaussian_indices.npy or legacy debug path); "
        "bbox + shell are recomputed each run. Does not reuse sim_gaussian_indices.npy as seeds.",
    ),
    debug_selection: bool = typer.Option(
        True,
        "--debug-selection/--no-debug-selection",
        help="Save bbox indices, selected-only PLY, selected_indices.npy, and optional CUDA frame-0 renders.",
    ),
    clean_debug_dir: bool = typer.Option(
        True,
        "--clean-debug-dir/--no-clean-debug-dir",
        help="With --debug-selection: delete <mode_b_out>/debug before writing new debug artifacts (keeps only the latest run).",
    ),
    shell_radius_m: float | None = typer.Option(
        None,
        "--shell-radius-m",
        help="Trim bbox fill to Gaussians within this distance (m) of surface seeds; 0 disables. "
        "Default: config physics.mode_b_shell_radius_m.",
    ),
    kinematic_wobble: bool = typer.Option(
        False,
        "--kinematic-wobble",
        help="Skip PhysGaussian MPM; apply subtle kinematic wobble to selected Gaussians only (static room).",
    ),
    wobble_track_debug: bool = typer.Option(
        False,
        "--wobble-track-debug",
        help="With --kinematic-wobble: print displacement stats on a few frames.",
    ),
    require_visual_selection: bool = typer.Option(
        True,
        "--require-visual-selection/--no-require-visual-selection",
        help="With --debug-selection: abort before wobble/MPM if too few selected Gaussians are visible in any camera.",
    ),
    debug_mask_to_gaussians_only: bool = typer.Option(
        False,
        "--debug-mask-to-gaussians-only",
        help="Run only SAM2 → mask → Gaussian projection debug (same folder as mode-b-selection-debug), then exit. "
        "No bbox / shell / CC / physics / wobble.",
    ),
    force_rerender_views: bool = typer.Option(
        False,
        "--force-rerender-views",
        help="Re-render rgb/depth/cam_meta before mask step (needed once after world_view_transform fix).",
    ),
    camera_index: int | None = typer.Option(
        None,
        "--camera-index",
        "--render-camera-index",
        help="Training camera index (Scene.getTrainCameras order) for SAM2 alignment, red overlay, wobble, MPM.",
    ),
    wobble_amp: float | None = typer.Option(
        None,
        "--wobble-amp",
        help="Kinematic wobble lateral amplitude in metres (default: physics.mode_b_kinematic_wobble_amp).",
    ),
    surface_indices_only: bool = typer.Option(
        False,
        "--surface-indices-only",
        "--no-bbox-expand",
        help="Skip bbox / shell / CC: simulate only surface (depth-consensus) indices (+ opacity filter).",
    ),
    sam2_render_camera_only: bool = typer.Option(
        False,
        "--sam2-render-camera-only",
        help="Run Grounded-SAM2 only on the rgb view matching the resolved training camera (see debug_selection/best_camera_index.txt or --camera-index).",
    ),
    taichi_memory_gb: float | None = typer.Option(
        None,
        "--taichi-memory-gb",
        help="PhysGaussian Taichi device_memory_GB (default: physics.taichi_device_memory_gb; T4 often 4–5).",
    ),
) -> None:
    cfg = PipelineConfig.load(config)
    colmap_scene = cfg.resolve(cfg.scene.data_root) / cfg.scene.scene_name
    model_out = cfg.resolve(cfg.scene.model_output)
    use_smoke_3dgs_checkpoint = smoke or smoke_3dgs
    iters = (
        cfg.scene.training_iterations_low if use_smoke_3dgs_checkpoint else cfg.scene.training_iterations
    )
    ply_path = model_out / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise typer.BadParameter(f"Missing {ply_path}; run train first")
    if debug_mask_to_gaussians_only:
        from segmentation.mask_projection_debug_run import run_mask_projection_debug

        run_mask_projection_debug(
            cfg=cfg,
            text=text,
            smoke=smoke,
            smoke_3dgs=smoke_3dgs,
            force_rerender_views=force_rerender_views,
            debug_mask_to_gaussians_only=True,
            projection_stride=1,
            console=console,
            maybe_cuda_empty=_maybe_cuda_empty,
        )
        return
    _mpm_grid = int(cfg.physics.n_grid_low if smoke else cfg.physics.n_grid)
    _mpm_frames = int(cfg.physics.mode_b_mpm_smoke_frames if smoke else cfg.physics.frame_num)
    _taichi_gb = float(taichi_memory_gb) if taichi_memory_gb is not None else float(cfg.physics.taichi_device_memory_gb)
    console.print(
        f"[bold]mode-b[/] smoke_3dgs={bool(use_smoke_3dgs_checkpoint)} iters={iters} "
        f"MPM(n_grid={_mpm_grid}, frames={_mpm_frames}) taichi_mem_GB={_taichi_gb}"
    )

    from plyfile import PlyData

    gply = PlyData.read(str(ply_path))
    v = gply["vertex"]
    pos = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)

    mode_b_out = cfg.resolve(cfg.paths.sim_output) / "mode_b_jelly"
    mode_b_out.mkdir(parents=True, exist_ok=True)
    if debug_selection and clean_debug_dir:
        shutil.rmtree(mode_b_out / "debug", ignore_errors=True)
    p_sel_cam_file = mode_b_out / "debug_selection" / "best_camera_index.txt"
    legacy_sel_cam_file = mode_b_out / "debug" / "best_camera_index.txt"
    forced_overlay_cam: int | None = (
        int(camera_index)
        if camera_index is not None
        else (
            _read_training_cam_index_txt(p_sel_cam_file)
            if _read_training_cam_index_txt(p_sel_cam_file) is not None
            else _read_training_cam_index_txt(legacy_sel_cam_file)
        )
    )
    camera_audit: list[str] = []

    surface_indices_path = mode_b_out / "surface_gaussian_indices.npy"
    idx_surface: np.ndarray | None = None
    idx_nearest_consensus: np.ndarray | None = None
    if reuse_selection:
        for cached in (
            cfg.resolve("output/3d_gaussian_selection_debug/selected_indices.npy"),
            surface_indices_path,
        ):
            if cached.is_file():
                idx_surface = np.load(cached).astype(np.int64)
                console.print(f"Reusing cached surface indices: [cyan]{cached}[/] ({len(idx_surface)} Gaussians)")
                break

    if idx_surface is None:
        render_dir = cfg.resolve("output/renders_views")
        seg = cfg.segmentation
        console.print("[mode-b] Rendering training views for segmentation (see render_views logs)...")
        render_training_views(
            colmap_scene,
            model_out,
            iters,
            render_dir,
            view_stride=cfg.reconstruction.render_view_stride,
        )

        mask_dbg = mode_b_out / "debug" / "masks"
        mask_dbg.mkdir(parents=True, exist_ok=True)
        sets_nearest: list[np.ndarray] = []
        sets_depth: list[np.ndarray] = []
        vs = int(cfg.reconstruction.render_view_stride)
        all_rgbs = sorted(render_dir.glob("rgb_*.png"))
        if sam2_render_camera_only or forced_overlay_cam is not None:
            cam_train = int(forced_overlay_cam) if forced_overlay_cam is not None else 0
            want_local = cam_train // vs
            match = render_dir / f"rgb_{want_local:05d}.png"
            if match.is_file():
                rgbs = [match]
                camera_audit.append(
                    f"Grounded-SAM2: training_camera_index={cam_train}  rgb_file={match.name}  "
                    f"(local_idx={want_local} * stride={vs})"
                )
            else:
                console.print(
                    f"[yellow][mode-b] missing {match.name} for training cam {cam_train}; "
                    f"falling back to first render[/]"
                )
                rgbs = all_rgbs[:1]
                camera_audit.append(
                    f"Grounded-SAM2: FALLBACK training_camera_index=0  rgb_file={rgbs[0].name}"
                )
        else:
            rgbs = all_rgbs[: seg.multi_view_count]
            for rgb in rgbs:
                stem0 = rgb.stem.replace("rgb_", "")
                try:
                    li = int(stem0)
                except ValueError:
                    li = 0
                camera_audit.append(
                    f"Grounded-SAM2: training_camera_index={li * vs}  rgb_file={rgb.name}"
                )
        for vi, rgb in enumerate(rgbs):
            console.print(
                f"[mode-b] Segmentation view {vi + 1}/{len(rgbs)}: [cyan]{rgb.name}[/] "
                "(Grounded SAM2 — can be slow)..."
            )
            stem = rgb.stem.replace("rgb_", "")
            meta = np.load(render_dir / f"cam_meta_{stem}.npz")
            K = np.asarray(meta["K"], dtype=np.float64)
            w2c = np.asarray(meta["world_view_transform"], dtype=np.float64)
            mask = text_to_mask(
                str(rgb),
                text,
                seg.box_threshold,
                seg.text_threshold,
                gdino_config=seg.grounding_dino_config or None,
                gdino_checkpoint=seg.grounding_dino_checkpoint or None,
                sam2_config=seg.sam2_config or None,
                sam2_checkpoint=seg.sam2_checkpoint or None,
                device=seg.segmentation_device,
                debug_mask_dir=mask_dbg,
                debug_stem=f"rgb_{stem}",
            )
            if int(seg.mask_erode_iters) > 0:
                from scipy.ndimage import binary_erosion

                mask = binary_erosion(mask, iterations=int(seg.mask_erode_iters))
            depth = np.load(render_dir / f"depth_{stem}.npy")
            idx_n = mask_to_gaussian_indices(
                mask,
                depth,
                K,
                w2c,
                pos,
                distance_threshold=float(seg.mask_3d_distance_threshold_m),
                stride=int(seg.mask_stride),
                use_depth_consistency=False,
            )
            idx_d = mask_to_gaussian_indices(
                mask,
                depth,
                K,
                w2c,
                pos,
                distance_threshold=float(seg.mask_3d_distance_threshold_m),
                stride=int(seg.mask_stride),
                use_depth_consistency=bool(seg.mask_use_depth_consistency),
                depth_tolerance_abs_m=float(seg.mask_depth_tolerance_abs_m),
                depth_tolerance_rel=float(seg.mask_depth_tolerance_rel),
                knn=int(seg.mask_knn),
                pixel_tolerance_px=float(seg.mask_pixel_tolerance_px),
                min_votes=int(seg.mask_min_votes),
            )
            sets_nearest.append(idx_n)
            sets_depth.append(idx_d)
            if debug_selection:
                save_gaussian_projection_debug_image(
                    str(rgb),
                    mask_dbg / f"{stem}_projection_nearest_centers.png",
                    pos,
                    idx_n,
                    K,
                    w2c,
                    color_bgr=(0, 255, 255),
                )
                save_gaussian_projection_debug_image(
                    str(rgb),
                    mask_dbg / f"{stem}_projection_depth_consistent.png",
                    pos,
                    idx_d,
                    K,
                    w2c,
                    color_bgr=(0, 0, 255),
                )
        _maybe_cuda_empty()
        gc.collect()

        idx_nearest_consensus = consensus_indices(sets_nearest, min_votes=seg.consensus_min_votes)
        idx_surface = consensus_indices(sets_depth, min_votes=seg.consensus_min_votes)
        if idx_surface.size == 0 and idx_nearest_consensus.size > 0:
            console.print(
                "[yellow]Depth-consistency + consensus returned empty; falling back to nearest-ray consensus.[/]"
            )
            idx_surface = idx_nearest_consensus

        if debug_selection:
            dbg_early = mode_b_out / "debug"
            dbg_early.mkdir(parents=True, exist_ok=True)
            save_subset_points_ply(pos, idx_nearest_consensus, dbg_early / "after_mask_projection.ply")
            save_subset_points_ply(pos, idx_surface, dbg_early / "after_depth_visibility_filter.ply")
            lo_n, hi_n = selection_xyz_bounds(pos, idx_nearest_consensus)
            lo_d, hi_d = selection_xyz_bounds(pos, idx_surface)
            console.print(
                f"[mode-b] After 2D→3D (nearest): {len(idx_nearest_consensus)}  "
                f"xyz min={lo_n.tolist()} max={hi_n.tolist()}"
            )
            console.print(
                f"[mode-b] After depth / pixel consistency: {len(idx_surface)}  "
                f"xyz min={lo_d.tolist()} max={hi_d.tolist()}"
            )
        else:
            console.print(
                f"[mode-b] Surface (depth-consensus) Gaussians: {len(idx_surface)} "
                "(PLY stage dumps need --debug-selection)"
            )

    np.save(surface_indices_path, idx_surface)
    lo_s, hi_s = selection_xyz_bounds(pos, idx_surface)
    console.print(
        f"Surface Gaussians: {len(idx_surface)}  xyz min={lo_s.tolist()} max={hi_s.tolist()}"
    )

    if surface_indices_only:
        idx_bbox = np.asarray(idx_surface, dtype=np.int64)
        idx_shell = idx_bbox
        idx_cc = idx_bbox
        idx = np.asarray(idx_surface, dtype=np.int64).copy()
        console.print(
            "[bold][mode-b] --surface-indices-only[/]: skipping bbox / shell / CC — "
            f"sim indices = surface set ({len(idx)} Gaussians)"
        )
    else:
        idx_bbox = expand_indices_to_bbox_volume(
            pos,
            idx_surface,
            max_indices=35000,
            margin_ratio=float(cfg.segmentation.mode_b_expand_bbox_margin_ratio),
            z_margin_ratio=float(cfg.segmentation.mode_b_expand_bbox_z_margin_ratio),
        )
        lo_b, hi_b = selection_xyz_bounds(pos, idx_bbox)
        console.print(
            f"Bbox expand: {len(idx_bbox)} Gaussians  xyz min={lo_b.tolist()} max={hi_b.tolist()}"
        )
        r_shell = cfg.physics.mode_b_shell_radius_m if shell_radius_m is None else float(shell_radius_m)
        if r_shell > 0.0:
            idx_shell = filter_expanded_to_shell_around_surface(pos, idx_surface, idx_bbox, r_shell)
            console.print(
                f"Bbox-expanded {len(idx_bbox)} → shell-filtered (r={r_shell:g} m) → {len(idx_shell)} Gaussians"
            )
        else:
            idx_shell = idx_bbox
            console.print(f"Bbox expansion only (shell filter off): {len(idx_shell)} Gaussians")
        lo_sh, hi_sh = selection_xyz_bounds(pos, idx_shell)
        console.print(
            f"After shell: {len(idx_shell)} Gaussians  xyz min={lo_sh.tolist()} max={hi_sh.tolist()}"
        )

        idx_cc = largest_component_touching_seeds(
            pos,
            idx_shell,
            idx_surface,
            float(cfg.physics.mode_b_cc_link_radius_m),
        )
        if idx_cc.size == 0:
            console.print("[yellow]Connected-component filter removed all points; using shell set.[/]")
            idx_cc = idx_shell
        else:
            console.print(
                f"Connected-component (seeds, r={cfg.physics.mode_b_cc_link_radius_m:g} m): "
                f"{len(idx_shell)} → {len(idx_cc)} Gaussians"
            )
        lo_cc, hi_cc = selection_xyz_bounds(pos, idx_cc)
        console.print(
            f"After CC: {len(idx_cc)} Gaussians  xyz min={lo_cc.tolist()} max={hi_cc.tolist()}"
        )

        idx = idx_cc
    opacity_floor = cfg.physics.mode_b_opacity_logit_min
    if opacity_floor is not None:
        logits = read_ply_opacity_logits(ply_path)
        n_op = len(idx)
        idx = filter_indices_by_opacity_logit(logits, idx, float(opacity_floor))
        console.print(f"Opacity logit ≥ {opacity_floor}: {n_op} → {len(idx)} Gaussians")

    lo_i, hi_i = selection_xyz_bounds(pos, idx)
    console.print(
        f"Final sim set: {len(idx)} Gaussians  xyz min={lo_i.tolist()} max={hi_i.tolist()}"
    )

    sim_indices_path = mode_b_out / "sim_gaussian_indices.npy"
    selected_indices_path = mode_b_out / "selected_indices.npy"
    np.save(sim_indices_path, idx)
    np.save(selected_indices_path, idx)
    console.print(
        f"[mode-b] saved [cyan]{sim_indices_path}[/] + [cyan]{selected_indices_path}[/]  "
        f"n_indices={len(idx)} (wobble/MPM use this set)",
    )

    dbg = mode_b_out / "debug"
    best_cam_idx = 0
    dbg_info: dict | None = None
    selection_debug_used_cuda = False
    if debug_selection:
        dbg.mkdir(parents=True, exist_ok=True)
        if surface_indices_only:
            save_subset_points_ply(pos, idx_surface, dbg / "surface_indices_only_input.ply")
        else:
            np.save(dbg / "bbox_expanded_indices.npy", np.asarray(idx_bbox, dtype=np.int64))
            save_subset_points_ply(pos, idx_bbox, dbg / "raw_selected_gaussians.ply")
            save_subset_points_ply(pos, idx_shell, dbg / "after_shell_filter.ply")
            save_subset_points_ply(pos, idx_cc, dbg / "after_connected_component_filter.ply")
        save_subset_points_ply(pos, idx, dbg / "final_selected_gaussians_only_points.ply")
        console.print(
            f"Stage PLYs: [cyan]{dbg}[/] "
            + (
                "(surface_indices_only_input + final)"
                if surface_indices_only
                else "(raw_selected_gaussians / after_shell / after_CC / final + bbox_expanded_indices.npy)"
            )
        )

    if debug_selection:
        try:
            import torch

            if torch.cuda.is_available():
                console.print(
                    "[yellow]Loading full 3DGS Scene once for visual debug + visibility table.[/] "
                    "Slow — use [bold]--no-debug-selection[/] to skip."
                )
                dbg_info = run_mode_b_selection_debug_suite(
                    colmap_scene=colmap_scene,
                    model_path=model_out,
                    iteration=iters,
                    scene_ply=ply_path,
                    selected_indices=idx,
                    dbg_dir=dbg,
                    forced_overlay_camera_index=forced_overlay_cam,
                    package_dir=mode_b_out,
                )
                selection_debug_used_cuda = True
                best_cam_idx = int(dbg_info["best_camera_index"])
                ov_cam = int(dbg_info.get("overlay_camera_used", best_cam_idx))
                camera_audit.append(
                    f"selected_red_overlay (frame00_selected_red_dc_primary.png): training_camera_index={ov_cam}"
                )
                vis0 = int(dbg_info["visible_view0"])
                visb = int(dbg_info["visible_best"])
                ns = dbg_info["counts_sample"]
                if not isinstance(ns, dict):
                    ns = {}
                console.print(
                    f"Selected total = {len(idx)}; visible (center in frustum) "
                    f"view0={vis0}, view50={int(ns.get(50, 0))}, view100={int(ns.get(100, 0))}, "
                    f"view150={int(ns.get(150, 0))}, best_cam={best_cam_idx} → {visb}"
                )
                if require_visual_selection:
                    mn = int(cfg.physics.mode_b_min_visible_selected)
                    if visb < mn:
                        raise typer.BadParameter(
                            f"Selection visual check failed: best camera only sees {visb} selected "
                            f"Gaussian centers in frustum (minimum {mn}). Inspect {dbg}, tune "
                            f"segmentation/shell/CC, or use --no-require-visual-selection."
                        )
                console.print(
                    f"Debug PNGs: primary overlay [cyan]{mode_b_out / 'selected_red_overlay_training_cam.png'}[/], "
                    f"baseline [cyan]{mode_b_out / 'render_camera_rgb.png'}[/], "
                    f"also [cyan]{dbg / 'frame00_selected_red_dc_view0.png'}[/], "
                    f"best_vis [cyan]{dbg / f'frame00_selected_red_dc_best_c{best_cam_idx}.png'}[/]"
                )
            else:
                console.print("[yellow]Debug CUDA frames skipped (no CUDA).[/]")
        except typer.BadParameter:
            raise
        except Exception as exc:  # noqa: BLE001 — diagnostics only
            console.print(f"[yellow]Debug render failed: {exc}[/]")

    if selection_debug_used_cuda:
        gc.collect()
        _maybe_cuda_empty()
        console.print("[dim]3DGS selection debug 후 gc + empty_cache[/]")

    render_camera_index = int(best_cam_idx)
    cam_src = "visibility suite (multi-cam best)"
    if forced_overlay_cam is not None:
        render_camera_index = int(forced_overlay_cam)
        cam_src = (
            "--camera-index/--render-camera-index"
            if camera_index is not None
            else str(p_sel_cam_file)
        )
    else:
        # Persist the chosen camera so future runs can stay view-consistent without
        # passing --camera-index explicitly.
        p_sel_cam_file.parent.mkdir(parents=True, exist_ok=True)
        p_sel_cam_file.write_text(f"{render_camera_index}\n", encoding="utf-8")
    console.print(
        f"[mode-b] final video / MPM camera_index=[bold]{render_camera_index}[/]  "
        f"(getTrainCameras order)  resolved_from={cam_src}"
    )
    camera_audit.append(
        f"kinematic_or_MPM_video: training_camera_index={render_camera_index}  (same as PhysGaussian default_camera_index)"
    )
    if not debug_selection:
        camera_audit.append("CUDA overlay suite skipped: use --debug-selection for red overlay PNGs")
    (mode_b_out / "camera_usage.txt").write_text(
        "\n".join(camera_audit)
        + f"\nrender_camera_rgb.png = baseline at training_camera_index={render_camera_index}\n",
        encoding="utf-8",
    )
    console.print(f"[mode-b] wrote camera audit [cyan]{mode_b_out / 'camera_usage.txt'}[/]")

    preview_png = mode_b_out / "render_camera_rgb.png"
    if not preview_png.is_file():
        try:
            import torch

            if torch.cuda.is_available():
                save_render_camera_rgb_png(
                    colmap_scene=colmap_scene,
                    model_path=model_out,
                    iteration=iters,
                    scene_ply=ply_path,
                    camera_index=render_camera_index,
                    out_path=preview_png,
                )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]render_camera_rgb preview skipped: {exc}[/]")

    w_amp_kinematic = float(cfg.physics.mode_b_kinematic_wobble_amp)
    if wobble_amp is not None:
        w_amp_kinematic = float(wobble_amp)
        console.print(f"[mode-b] kinematic wobble amp override: [bold]{w_amp_kinematic}[/] m")

    console.print(
        "[dim]Static background:[/] PhysGaussian `unselected_*` tensors are never updated in the "
        "simulation loop; each frame concatenates [simulated subset | static rest] "
        "(see submodules/PhysGaussian/gs_simulation.py). With [bold]--kinematic-wobble[/], only "
        "selected indices receive xyz offsets; the rest match the idle copy."
    )

    if kinematic_wobble:
        fnum = cfg.physics.frame_num_test if smoke else cfg.physics.frame_num
        playback = cfg.physics.compile_video_playback_sec
        console.print(
            f"[mode-b kinematic] loading indices from in-memory selection n={len(idx)}  "
            f"(same as [cyan]{selected_indices_path}[/] just written)",
        )
        vid = render_kinematic_wobble_scene_by_indices(
            colmap_scene=colmap_scene,
            model_path=model_out,
            iteration=iters,
            scene_ply=ply_path,
            object_indices=idx,
            sim_run=mode_b_out,
            frame_num=int(fnum),
            frame_dt=float(cfg.physics.frame_dt),
            playback_seconds=float(playback if playback is not None else 10.0),
            camera_index=render_camera_index,
            wobble_amp=w_amp_kinematic,
            wobble_frequency=float(cfg.physics.mode_b_kinematic_wobble_freq_hz),
            wobble_height_weight=float(cfg.physics.mode_b_kinematic_wobble_height_gamma),
            wobble_bottom_pin=float(cfg.physics.mode_b_kinematic_wobble_bottom_pin),
            track_debug=wobble_track_debug,
            indices_source=str(selected_indices_path.resolve()),
        )
        console.print(f"Video (kinematic): [green]{vid}[/]")
        return

    sim_cfg = mode_b_out / "phys_config.json"
    sim_n_grid = int(cfg.physics.n_grid_low if smoke else cfg.physics.n_grid)
    sim_frame_num = int(cfg.physics.mode_b_mpm_smoke_frames if smoke else cfg.physics.frame_num)
    # Sustained wobble drives MPM physics time → match requested motion horizon to video/sim length.
    if cfg.physics.mode_b_phys_sustained_wobble and not smoke:
        motion_s = cfg.physics.mode_b_phys_wobble_motion_seconds
        if motion_s is None:
            motion_s = float(cfg.physics.compile_video_playback_sec)
        tgt = int(math.ceil(float(motion_s) / float(cfg.physics.frame_dt)) + 2)
        sim_frame_num = max(sim_frame_num, tgt)
    gc.collect()
    _log_cuda_memory_line(console, "[mode-b] PhysGaussian 직전 GPU")
    _maybe_cuda_empty()
    # Mode-b jelly: support plane from the *selection* (+Y-down → high-Y percentile = contact).
    # PhysGaussian pins COM each frame (see gs_simulation + phys JSON) so the desk
    # stays in place while MPM adds local deformation / shear wobble BCs.
    support_y = estimate_mode_b_support_contact_y(
        pos,
        idx,
        contact_percentile=float(cfg.physics.mode_b_support_contact_percentile),
    )
    sim_margin = (
        float(cfg.physics.mode_b_sim_area_margin)
        if cfg.physics.mode_b_sim_area_margin is not None
        else float(cfg.physics.sim_area_margin)
    )
    mat_b: dict[str, float] = {}
    if cfg.physics.mode_b_jelly_E is not None:
        mat_b["E"] = float(cfg.physics.mode_b_jelly_E)
    if cfg.physics.mode_b_jelly_grid_v_damping_scale is not None:
        mat_b["grid_v_damping_scale"] = float(cfg.physics.mode_b_jelly_grid_v_damping_scale)
    console.print(
        f"[mode-b] jelly in-place: support_contact_y={support_y:.4f} "
        f"(pctl={cfg.physics.mode_b_support_contact_percentile})  "
        f"selected bbox min={lo_i.tolist()} max={hi_i.tolist()}  "
        f"world_down=+scene_Y  mpm_world_up=(0,-1,0)  "
        f"pin_com={cfg.physics.mode_b_pin_initial_com}  "
        f"pin_vertical_only={cfg.physics.mode_b_pin_com_vertical_only}  "
        f"shear_wobble={cfg.physics.mode_b_mpm_in_place_shear_wobble}  "
        f"shear_full_vol={cfg.physics.mode_b_shear_wobble_full_volume}  "
        f"sust_phys={cfg.physics.mode_b_phys_sustained_wobble}  "
        f"sust_hz={cfg.physics.mode_b_phys_wobble_frequency_hz}  "
        f"vx_peak={cfg.physics.mode_b_phys_wobble_velocity_peak_x or cfg.physics.mode_b_phys_wobble_velocity_peak}  "
        f"vy_peak={cfg.physics.mode_b_phys_wobble_velocity_peak_y}  "
        f"vz_peak={cfg.physics.mode_b_phys_wobble_velocity_peak_z}  "
        f"d12=({cfg.physics.mode_b_phys_wobble_velocity_peak_diag1},"
        f"{cfg.physics.mode_b_phys_wobble_velocity_peak_diag2})  "
        f"twist_peak={cfg.physics.mode_b_phys_wobble_velocity_peak_twist}  "
        f"frames={sim_frame_num}  "
        f"sim_margin={sim_margin}  "
        f"E={mat_b.get('E', 'preset')}  "
        f"shear_v={cfg.physics.mode_b_shear_wobble_velocity}  "
        f"shear_t={cfg.physics.mode_b_shear_wobble_end_time}s  "
        f"disp_retention={cfg.physics.mode_b_mpm_displacement_retention}  "
        f"kabsch_strip={cfg.physics.mode_b_mpm_kabsch_rigid_strip}  "
        f"kabsch_amp={cfg.physics.mode_b_mpm_kabsch_elastic_amp}  "
        f"anchor_feet_y_pctl={cfg.physics.mode_b_mpm_anchor_feet_y_percentile}  "
        f"shear_symmetric_lr={cfg.physics.mode_b_mpm_shear_symmetric_lr_split}  "
        f"tilt_diag={cfg.physics.mode_b_mpm_tilt_diagnostics}  "
        f"freeze_cov_render={cfg.physics.mode_b_render_freeze_gaussian_cov}"
    )
    generate_phys_config(
        idx,
        pos,
        "jelly",
        sim_cfg,
        template_path=default_template_dir() / "jelly.json",
        n_grid=sim_n_grid,
        frame_num=sim_frame_num,
        frame_dt=cfg.physics.frame_dt,
        substep_dt=cfg.physics.substep_dt,
        gravity=float(cfg.physics.gravity)
        * float(cfg.physics.gravity_scale)
        * float(cfg.physics.mode_b_jelly_gravity_mult),
        camera_index=render_camera_index,
        simulate_indices_npy=sim_indices_path,
        subtract_rigid_drift=False,
        floor_y=support_y,
        world_up=(0.0, -1.0, 0.0),
        floor_collider=True,
        floor_friction=float(cfg.physics.floor_friction),
        in_place_wobble=bool(cfg.physics.mode_b_mpm_in_place_shear_wobble),
        wobble_velocity=float(cfg.physics.mode_b_shear_wobble_velocity),
        wobble_end_time=float(cfg.physics.mode_b_shear_wobble_end_time),
        shear_wobble_full_volume=bool(cfg.physics.mode_b_shear_wobble_full_volume),
        pin_initial_com_mpm=bool(cfg.physics.mode_b_pin_initial_com),
        pin_com_vertical_only=bool(cfg.physics.mode_b_pin_com_vertical_only),
        pin_zero_mean_velocity_gs=bool(cfg.physics.mode_b_pin_zero_mean_velocity_gs),
        mode_b_mpm_displacement_retention=(
            float(cfg.physics.mode_b_mpm_displacement_retention)
            if cfg.physics.mode_b_mpm_displacement_retention is not None
            else None
        ),
        mode_b_render_freeze_gaussian_cov=bool(cfg.physics.mode_b_render_freeze_gaussian_cov),
        mode_b_mpm_kabsch_rigid_strip=bool(cfg.physics.mode_b_mpm_kabsch_rigid_strip),
        mode_b_mpm_kabsch_elastic_amp=float(cfg.physics.mode_b_mpm_kabsch_elastic_amp),
        mode_b_mpm_anchor_feet_y_percentile=(
            float(cfg.physics.mode_b_mpm_anchor_feet_y_percentile)
            if cfg.physics.mode_b_mpm_anchor_feet_y_percentile is not None
            else None
        ),
        shear_symmetric_lr_split=bool(cfg.physics.mode_b_mpm_shear_symmetric_lr_split),
        phys_sustained_wobble=bool(cfg.physics.mode_b_phys_sustained_wobble),
        phys_wobble_frequency_hz=float(cfg.physics.mode_b_phys_wobble_frequency_hz),
        phys_wobble_velocity_peak=float(cfg.physics.mode_b_phys_wobble_velocity_peak),
        phys_wobble_velocity_peak_x=(
            float(cfg.physics.mode_b_phys_wobble_velocity_peak_x)
            if cfg.physics.mode_b_phys_wobble_velocity_peak_x is not None
            else None
        ),
        phys_wobble_velocity_peak_y=float(cfg.physics.mode_b_phys_wobble_velocity_peak_y),
        phys_wobble_phase_y_rad=float(cfg.physics.mode_b_phys_wobble_phase_y),
        phys_wobble_band_amp_x=cfg.physics.mode_b_phys_wobble_band_amp_x,
        phys_wobble_band_amp_y=cfg.physics.mode_b_phys_wobble_band_amp_y,
        phys_wobble_band_vy_polarity=cfg.physics.mode_b_phys_wobble_band_vy_polarity,
        phys_wobble_band_phase_y_offset_rad=cfg.physics.mode_b_phys_wobble_band_phase_y_offset_rad,
        phys_wobble_velocity_peak_z=float(cfg.physics.mode_b_phys_wobble_velocity_peak_z),
        phys_wobble_velocity_peak_diag1=float(cfg.physics.mode_b_phys_wobble_velocity_peak_diag1),
        phys_wobble_velocity_peak_diag2=float(cfg.physics.mode_b_phys_wobble_velocity_peak_diag2),
        phys_wobble_velocity_peak_twist=float(cfg.physics.mode_b_phys_wobble_velocity_peak_twist),
        phys_wobble_phase_z_rad=float(cfg.physics.mode_b_phys_wobble_phase_z),
        phys_wobble_phase_diag1_rad=float(cfg.physics.mode_b_phys_wobble_phase_diag1_rad),
        phys_wobble_phase_diag2_rad=float(cfg.physics.mode_b_phys_wobble_phase_diag2_rad),
        phys_wobble_phase_twist_rad=float(cfg.physics.mode_b_phys_wobble_phase_twist_rad),
        phys_wobble_bundle_quad_phase_rad=cfg.physics.mode_b_phys_wobble_bundle_quad_phase_rad,
        phys_wobble_bundle_band_phase_rad=cfg.physics.mode_b_phys_wobble_bundle_band_phase_rad,
        phys_wobble_band_amp_z=cfg.physics.mode_b_phys_wobble_band_amp_z,
        phys_wobble_band_amp_diag1=cfg.physics.mode_b_phys_wobble_band_amp_diag1,
        phys_wobble_band_amp_diag2=cfg.physics.mode_b_phys_wobble_band_amp_diag2,
        phys_wobble_band_amp_twist=cfg.physics.mode_b_phys_wobble_band_amp_twist,
        phys_wobble_force_scale=float(cfg.physics.mode_b_phys_wobble_force_scale),
        phys_wobble_decay_lambda_per_s=float(cfg.physics.mode_b_phys_wobble_decay_lambda_per_s),
        phys_wobble_duration_s=(
            float(cfg.physics.mode_b_phys_wobble_duration_s)
            if cfg.physics.mode_b_phys_wobble_duration_s is not None
            else None
        ),
        phys_wobble_ramp_time_s=float(cfg.physics.mode_b_phys_wobble_ramp_time_s),
        mode_b_mpm_tilt_diagnostics=bool(cfg.physics.mode_b_mpm_tilt_diagnostics),
        mode_b_mpm_tilt_top_y_percentile=float(cfg.physics.mode_b_mpm_tilt_top_y_percentile),
        material_overrides=mat_b if mat_b else None,
        enable_internal_particle_fill=cfg.physics.enable_mpm_particle_filling,
        sim_area_margin=sim_margin,
        particle_filling={
            "n_grid": min(50, sim_n_grid),
            "density_threshold": 3.0,
            "search_threshold": 0.8,
            "max_partciels_per_cell": 4,
            "smooth": True,
            "visualize": False,
        },
    )
    vid = run_simulation(
        str(ply_path),
        str(sim_cfg),
        str(mode_b_out),
        camera_scene_path=str(model_out),
        playback_seconds=cfg.physics.compile_video_playback_sec,
        taichi_device_memory_gb=_taichi_gb,
    )
    dbg_js = mode_b_out / "frames" / "mode_b_mpm_debug.json"
    if dbg_js.is_file():
        console.print(f"[mode-b] MPM COM / drift log: [cyan]{dbg_js}[/]")
    tilt_js = mode_b_out / "frames" / "mode_b_mpm_tilt_series.json"
    if tilt_js.is_file():
        console.print(f"[mode-b] MPM tilt time-series + drift summary: [cyan]{tilt_js}[/]")
    console.print(f"Video: [green]{vid}[/]")


@app.command("mode-b-selection-debug")
def mode_b_selection_debug(
    text: str = typer.Argument(..., help="Segmentation prompt, e.g. \"wooden table in the foreground.\""),
    config: Path = typer.Option(PROJECT_ROOT / "configs" / "pipeline_config.yaml", "--config"),
    smoke: bool = typer.Option(False, "--smoke", help="Use low-iter 3DGS PLY (same as mode-b --smoke)."),
    smoke_3dgs: bool = typer.Option(
        False,
        "--smoke-3dgs",
        help="Low 3DGS checkpoint only (combined with --smoke semantics).",
    ),
    force_rerender_views: bool = typer.Option(
        False,
        "--force-rerender-views",
        help="Run render_training_views even if rgb_*.png already exist (needed once after cam_meta fix).",
    ),
    debug_mask_to_gaussians_only: bool = typer.Option(
        False,
        "--debug-mask-to-gaussians-only",
        help="Skip mask erode, depth consistency, and strict 2D gate; final = projection inside mask only.",
    ),
    projection_stride: int = typer.Option(
        1,
        "--projection-stride",
        min=1,
        help="Subsample Gaussians when testing projection-inside-mask (1 = every Gaussian).",
    ),
) -> None:
    """Clean folder: SAM2 → mask → Gaussian indices (single training view). No MPM / wobble."""
    from segmentation.mask_projection_debug_run import run_mask_projection_debug

    cfg = PipelineConfig.load(config)
    run_mask_projection_debug(
        cfg=cfg,
        text=text,
        smoke=smoke,
        smoke_3dgs=smoke_3dgs,
        force_rerender_views=force_rerender_views,
        debug_mask_to_gaussians_only=debug_mask_to_gaussians_only,
        projection_stride=projection_stride,
        console=console,
        maybe_cuda_empty=_maybe_cuda_empty,
    )


@app.command()
def mode_a(
    text: str = typer.Argument(..., help="Object to generate"),
    config: Path = typer.Option(PROJECT_ROOT / "configs" / "pipeline_config.yaml", "--config"),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Match ``train --smoke`` (low-iter 3DGS PLY) and lower SDS unless --full-sds.",
    ),
    smoke_3dgs: bool = typer.Option(
        False,
        "--smoke-3dgs",
        help="Low 3DGS checkpoint only; full SDS / physics settings (unlike --smoke alone).",
    ),
    full_sds: bool = typer.Option(
        False,
        "--full-sds",
        help="With --smoke: still use full generation.sds_steps for DreamGaussian (3DGS stays on low iters).",
    ),
    no_physics: bool = typer.Option(
        False,
        "--no-physics",
        help="Skip MPM config + simulation after merge (placement / merged PLY only).",
    ),
    material: str = typer.Option(
        "jelly",
        "--material",
        help="Physics preset: jelly (soft), rubber (stiffer toy / less spread), foam, sand, metal, rigid",
    ),
    mvdream: bool = typer.Option(
        True,
        "--mvdream/--no-mvdream",
        help="Use MVDream multi-view backbone (default on); --no-mvdream forces SD 1.5",
    ),
    reuse_object: bool = typer.Option(
        False,
        "--reuse-object",
        help="Reuse an existing generated object in output/generated_objects/sds_run when present.",
    ),
    asset_path: Path | None = typer.Option(
        None,
        "--asset-path",
        help="External triangle mesh (e.g. Sketchfab .glb). Staged as sds_run/object_mesh.glb "
        "(or .obj), sampled to object_mesh_gaussians.ply, then rescale/merge/physics — skips SD + DG.",
    ),
    object_gaussians: Path | None = typer.Option(
        None,
        "--object-gaussians",
        help="Pre-built Gaussian object PLY (e.g. object_mesh_gaussians.ply). Copied into sds_run; "
        "skips SD, DreamGaussian, and mesh sampling. Use --asset-path .glb if you need --mesh-render overlay.",
    ),
    no_recolor: bool = typer.Option(
        False,
        "--no-recolor",
        help="Disable the red-duck recolor pass; keep DreamGaussian's raw SH colors.",
    ),
    raw_object: bool = typer.Option(
        False,
        "--raw-object",
        help="Diagnostic: skip recolor + opacity/spatial filtering + reorient. "
             "Renders DreamGaussian's raw stage-1 PLY as-is (apart from uniform scaling).",
    ),
    image_mode: bool = typer.Option(
        False,
        "--image-mode",
        help="Use DreamGaussian image-to-3D: generate one SD reference image from "
             "the prompt, run rembg, then drive DG with stable-zero123. Much more "
             "accurate shapes than text-only SDS for objects like rubber ducks.",
    ),
    image_steps: int = typer.Option(
        500,
        "--image-steps",
        help="Stage-1 iterations for DreamGaussian image-to-3D (default 500).",
    ),
    image_seed: int = typer.Option(
        123,
        "--image-seed",
        help="Base seed for SD reference candidates (uses [seed, seed+1, ...]).",
    ),
    image_candidates: int = typer.Option(
        4,
        "--image-candidates",
        help="How many SD candidates to generate; the best (single subject + centred + sized) is chosen.",
    ),
    reuse_reference: bool = typer.Option(
        False,
        "--reuse-reference",
        help="Reuse the existing reference.png for --image-mode instead of regenerating with SD.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Render a single static frame (frame_num=1) so placement / orientation "
             "can be confirmed before committing to a full physics simulation.",
    ),
    swap_red_yellow: bool = typer.Option(
        False,
        "--swap-red-yellow",
        help="Swap red/pink duck body colours with yellow beak/wing colours.",
    ),
    yellow_duck: bool = typer.Option(
        False,
        "--yellow-duck",
        help="Keep classic yellow (orange beak) rubber duck colours: use yellow wording for SD / text SDS "
        "and never apply red-duck SH recolor—even if the typed prompt says 'red' for the write-up.",
    ),
    object_scale_factor: float | None = typer.Option(
        None,
        "--object-scale-factor",
        help="Override generation.object_scale_factor for this run.",
    ),
    in_place_wobble: bool = typer.Option(
        False,
        "--in-place-wobble",
        help="Shape-preserving jelly wobble: default is kinematic sinusoidal deformation on object Gaussians "
        "(no MPM collapse). Use --wobble-use-mpm for PhysGaussian impulse+wobble instead.",
    ),
    wobble_use_mpm: bool = typer.Option(
        False,
        "--wobble-use-mpm",
        help="With --in-place-wobble: drive wobble via PhysGaussian MPM (contact-prone). Default is kinematic.",
    ),
    wobble_amp: float = typer.Option(
        0.012,
        "--wobble-amp",
        help="Kinematic wobble: lateral amplitude in world metres (mean-free displacement).",
    ),
    wobble_frequency: float = typer.Option(
        1.15,
        "--wobble-frequency",
        help="Kinematic wobble: cycles per second (sin 2πft on lateral motion).",
    ),
    wobble_height_weight: float = typer.Option(
        1.75,
        "--wobble-height-weight",
        help="Kinematic wobble: exponent on height weight (larger = more motion near head, less near feet).",
    ),
    wobble_bottom_pin: float = typer.Option(
        0.24,
        "--wobble-bottom-pin",
        help="Kinematic wobble: fraction of object height (+Y-down bbox) to pin at the feet band.",
    ),
    no_pin_com_drift: bool = typer.Option(
        False,
        "--no-pin-com-drift",
        help="With --in-place-wobble and --wobble-use-mpm: disable PhysGaussian subtract_rigid_drift.",
    ),
    wobble_debug: bool = typer.Option(
        False,
        "--wobble-debug",
        help="With --in-place-wobble: save per-frame object Gaussian xyz and print motion stats.",
    ),
    opacity_filter_logit: float = typer.Option(
        -0.1,
        "--opacity-filter-logit",
        help="Filter low-opacity object Gaussians. Higher values remove more fuzzy splat noise.",
    ),
    post_rotate_x: float = typer.Option(0.0, "--post-rotate-x", help="Manual object correction rotation in degrees."),
    post_rotate_y: float = typer.Option(0.0, "--post-rotate-y", help="Manual object correction rotation in degrees."),
    post_rotate_z: float = typer.Option(0.0, "--post-rotate-z", help="Manual object correction rotation in degrees."),
    mesh_gaussians: bool = typer.Option(
        False,
        "--mesh-gaussians",
        help="Sample object_mesh.glb (preferred) or object_mesh.obj into clean Gaussian splats.",
    ),
    canonical_frame_override: str | None = typer.Option(
        None,
        "--canonical-frame",
        help="Override object canonical frame for rescaling: text, image, or mesh (OBJ import; no DG tilt preset).",
    ),
    auto_floor_tilt: bool = typer.Option(
        True,
        "--auto-floor-tilt/--no-auto-floor-tilt",
        help="Tilt the object so its base matches a local linear fit of the rug (+Y-down scenes).",
    ),
    rug_tilt: bool = typer.Option(
        False,
        "--rug-tilt",
        help="For image-mode ducks: match local rug plane slope (default off — ducks stand upright on level floor).",
    ),
    contact_base_ao: float = typer.Option(
        0.28,
        "--contact-base-ao",
        help="Darken SH DC near the feet (0=off, typ. 0.2–0.4) for subtle contact shading.",
    ),
    camera_base_pitch_deg: float = typer.Option(
        -5.5,
        "--camera-base-pitch-deg",
        help="With --auto-floor-tilt: pitch foot-normal (deg) about horizontal camera-right to match rug perspective; 0 disables.",
    ),
    mesh_render: bool = typer.Option(
        False,
        "--mesh-render",
        help="Export DreamGaussian object_mesh.obj into scene space; after sim, composite sharp mesh over base 3DGS "
        "(KNN skinning from saved Gaussian trajectories when available, else centroid rigid), then re-encode mp4. "
        "With --no-physics, one composite frame only.",
    ),
    mesh_skin_k: int = typer.Option(
        12,
        "--mesh-skin-k",
        help="Number of nearest Gaussians per mesh vertex for --mesh-render jelly transfer (3–32).",
    ),
    no_auto_realism: bool = typer.Option(
        False,
        "--no-auto-realism",
        help="Disable duck/rubber-duck defaults: auto image-mode, mesh-render, and floor Y refinement.",
    ),
    hold_for_review: bool = typer.Option(
        False,
        "--hold-for-review",
        help="Stop after generate + rescale; writes mode_a_review.json. Inspect reference.png / "
        "object_scaled.ply, then rerun the same prompt with --continue-after-review.",
    ),
    continue_after_review: bool = typer.Option(
        False,
        "--continue-after-review",
        help="Resume from mode_a_review.json (same prompt): merge, physics, video.",
    ),
) -> None:
    cfg = PipelineConfig.load(config)
    model_out = cfg.resolve(cfg.scene.model_output)
    colmap_scene = cfg.resolve(cfg.scene.data_root) / cfg.scene.scene_name
    use_smoke_3dgs_checkpoint = smoke or smoke_3dgs
    iters = (
        cfg.scene.training_iterations_low if use_smoke_3dgs_checkpoint else cfg.scene.training_iterations
    )
    base_ply = model_out / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not base_ply.is_file():
        raise typer.BadParameter(f"Missing {base_ply}; run train first")

    gen_dir = cfg.resolve(cfg.paths.generated_objects) / "sds_run"
    gen_dir.mkdir(parents=True, exist_ok=True)
    use_prebuilt_object_gaussians = False
    if asset_path is not None and object_gaussians is not None:
        raise typer.BadParameter("Use only one of --asset-path (mesh) or --object-gaussians (.ply).")
    if hold_for_review and continue_after_review:
        raise typer.BadParameter("Use only one of --hold-for-review / --continue-after-review.")
    from_review = bool(continue_after_review)
    review_data: dict | None = None
    if object_gaussians is not None and from_review:
        raise typer.BadParameter("--object-gaussians cannot be used with --continue-after-review.")
    if object_gaussians is not None and image_mode:
        raise typer.BadParameter("--object-gaussians skips image-to-3D; remove --image-mode.")
    if object_gaussians is not None and not from_review:
        src_g = Path(object_gaussians).expanduser().resolve()
        if not src_g.is_file():
            raise typer.BadParameter(f"--object-gaussians not a file: {src_g}")
        if src_g.suffix.lower() != ".ply":
            raise typer.BadParameter("--object-gaussians must be a .ply (Gaussian object export).")
        dest_g = gen_dir / "object_mesh_gaussians.ply"
        dest_resolved = dest_g.resolve()
        if src_g != dest_resolved:
            shutil.copy2(src_g, dest_g)
        use_prebuilt_object_gaussians = True
        image_mode = False
        if src_g == dest_resolved:
            console.print(
                f"[bold]Pre-built Gaussian object[/] for insertion: [cyan]{dest_resolved}[/] "
                f"(already at insertion path; no copy)"
            )
        else:
            console.print(
                f"[bold]Pre-built Gaussian object[/] for insertion: [cyan]{dest_resolved}[/] "
                f"(copied from {src_g})"
            )
    if asset_path is not None and from_review:
        raise typer.BadParameter("--asset-path cannot be used with --continue-after-review.")
    if asset_path is not None and image_mode:
        raise typer.BadParameter("--asset-path already skips image-to-3D; remove --image-mode.")
    if asset_path is not None and not from_review:
        try:
            staged = stage_external_mesh_into_gen_dir(gen_dir, asset_path)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise typer.BadParameter(f"--asset-path: could not load mesh: {exc}") from exc
        mesh_gaussians = True
        image_mode = False
        console.print(
            f"[bold]External mesh asset[/] staged as [cyan]{staged.resolve()}[/] "
            f"(source {Path(asset_path).expanduser().resolve()}) → will sample to object_mesh_gaussians.ply"
        )

    if from_review:
        review_data = _load_mode_a_review_checkpoint(gen_dir)
        if review_data.get("text") != text:
            raise typer.BadParameter(
                f"Review checkpoint expects the same prompt ({review_data.get('text')!r}); "
                f"got {text!r}. Delete {gen_dir / MODE_A_REVIEW_JSON} to reset."
            )
        yellow_duck_eff = bool(review_data.get("yellow_duck", False))
    else:
        yellow_duck_eff = bool(yellow_duck)

    if not from_review:
        auto_duck_realism = (
            not no_auto_realism
            and not raw_object
            and not mesh_gaussians
            and not use_prebuilt_object_gaussians
            and wants_object_realism_preset(text)
        )
        if auto_duck_realism and not image_mode:
            image_mode = True
            no_recolor = True
            console.print(
                "[cyan]Duck realism preset:[/] enabling [bold]--image-mode[/] for stable shape "
                "(pass [bold]--no-auto-realism[/] to force text-only SDS)."
            )
    else:
        auto_duck_realism = False
    use_full_steps = (not smoke) or full_sds
    steps = cfg.generation.sds_steps if use_full_steps else cfg.generation.sds_steps_low
    use_mv = bool(mvdream)
    backbone = "MVDream" if use_mv else "SD"
    console.print(
        f"[bold]mode-a[/] backbone={backbone} smoke_3dgs={bool(use_smoke_3dgs_checkpoint)} iters={iters} "
        f"sds_steps={steps} material={material} no_physics={bool(no_physics)} "
        f"hold_for_review={bool(hold_for_review)} continue_after_review={bool(continue_after_review)} "
        f"realism_preset={bool(auto_duck_realism)} "
        f"mesh_render={bool(mesh_render)} mesh_skin_k={int(mesh_skin_k) if mesh_render else '-'} "
        f"in_place_wobble={bool(in_place_wobble)} wobble_mpm={bool(in_place_wobble and wobble_use_mpm)} "
        f"wobble_kinematic={bool(in_place_wobble and not wobble_use_mpm)}"
    )
    if yellow_duck_eff:
        console.print(
            "[cyan]Yellow duck mode:[/] colours follow [bold]yellow[/] rubber duck "
            "(red SH recolor off; SD reference wording uses yellow duck)."
        )
    m = cfg.generation.stable_diffusion_model.strip()
    sd_ver = "1.5" if ("v1-5" in m or "1.5" in m) else "2.1"
    colour_source = _YELLOW_RUBBER_DUCK_TEXT if yellow_duck_eff else text
    generation_prompt = enhance_generation_prompt(colour_source)
    if generation_prompt != colour_source:
        console.print(f"Generation prompt: {generation_prompt}")
    if from_review:
        rd = review_data
        assert rd is not None
        scaled = Path(rd["scaled_path"])
        obj_ply = Path(rd["obj_ply"])
        if not scaled.is_file() or not obj_ply.is_file():
            raise typer.BadParameter(
                "Review checkpoint paths are invalid or files were deleted. "
                f"scaled={scaled} obj_ply={obj_ply}"
            )
        place_x = float(rd["place_x"])
        place_y = float(rd["place_y"])
        place_z = float(rd["place_z"])
        floor_y = float(rd["floor_y"])
        al = rd.get("align_y")
        align_y = None if al is None else np.asarray(al, dtype=np.float64)
        canonical_frame = str(rd["canonical_frame"])
        scale_factor = float(rd["scale_factor"])
        pr = rd["post_rotate"]
        post_rotate_x, post_rotate_y, post_rotate_z = float(pr[0]), float(pr[1]), float(pr[2])
        opacity_filter_logit = float(rd["opacity_filter_logit"])
        contact_base_ao = float(rd["contact_base_ao"])
        image_mode = bool(rd["image_mode"])
        raw_object = bool(rd["raw_object"])
        mesh_render = bool(rd["mesh_render"])
        mesh_skin_k = int(rd.get("mesh_skin_k", 12))
        swap_red_yellow = bool(rd.get("swap_red_yellow", False))
        mesh_gaussians = bool(rd.get("mesh_gaussians", False))
        red_duck = False
        n_scaled = len(PlyData.read(str(scaled))["vertex"])
        console.print(
            f"[green]Review resume:[/] using checkpoint [cyan]{gen_dir / MODE_A_REVIEW_JSON}[/] "
            f"({n_scaled} object Gaussians)."
        )
    else:
        # ``--raw-object`` is a stronger version of ``--no-recolor`` that also
        # disables every other post-processing step on the rescaler.  Image-mode
        # always implies ``--no-recolor`` because the SD reference already supplies
        # the desired colours / shape — we don't want to repaint with hard-coded
        # red SH values afterwards.
        if raw_object:
            no_recolor = True
        if image_mode:
            no_recolor = True
        red_duck = is_red_duck_prompt(text) and not no_recolor and not yellow_duck_eff
        # When --no-recolor is set we fall back to the raw DreamGaussian PLY so the
        # SH colours / fine geometry survive into the simulation.
        if no_recolor:
            reuse_candidate = gen_dir / "object_model.ply"
        else:
            reuse_candidate = gen_dir / ("object_red_duck.ply" if red_duck else "object_model.ply")
        if asset_path is None and not use_prebuilt_object_gaussians:
            if reuse_object and reuse_candidate.is_file() and reuse_candidate.stat().st_size > 1_000_000:
                obj_ply = reuse_candidate
                console.print(f"Reusing generated object: [cyan]{obj_ply}[/]")
            elif image_mode:
                # Stable-Diffusion → rembg → DreamGaussian image-to-3D (stable-zero123)
                ref_dir = gen_dir / "reference"
                ref_dir.mkdir(parents=True, exist_ok=True)
                ref_png = ref_dir / "reference.png"
                if reuse_reference and ref_png.is_file():
                    console.print(f"[bold]image-mode:[/] reusing existing reference image: {ref_png}")
                else:
                    console.print(
                        f"[bold]image-mode:[/] generating SD candidates "
                        f"(base_seed={image_seed}, candidates={image_candidates}) → {ref_png}"
                    )
                    generate_reference_image(
                        colour_source,
                        ref_png,
                        seed=image_seed,
                        num_candidates=int(image_candidates),
                    )
                    _maybe_cuda_empty()
                    gc.collect()
                # Do not auto-raise image iters for ducks: e.g. 750 iters + DG's prune/opacity schedule
                # can collapse the splat set to 0 points → empty object_model.ply.  Use --image-steps manually.
                console.print(
                    f"[bold]image-mode:[/] running DreamGaussian image-to-3D ({int(image_steps)} iters, stable-zero123)"
                )
                # NOTE: Do not override sh_degree for image.yaml — sh_degree=1 makes DG export 0 Gaussians
                # (empty PLY) with the current DreamGaussian image pipeline.
                obj_ply = generate_object_from_image(
                    ref_png,
                    gen_dir,
                    num_steps=int(image_steps),
                    elevation=0.0,
                    use_stable_zero123=True,
                )
            else:
                obj_ply = generate_object(
                    generation_prompt, gen_dir, num_steps=steps,
                    hf_key=m if "/" in m else None,
                    sd_version=sd_ver,
                    use_mvdream=use_mv,
                )
                if red_duck:
                    obj_ply = recolor_red_duck_ply(obj_ply, gen_dir / "object_red_duck.ply")
            if swap_red_yellow:
                swapped = gen_dir / "object_red_yellow_swapped.ply"
                obj_ply = swap_red_yellow_duck_ply(obj_ply, swapped)
                console.print(f"Swapped red/yellow duck colours: [cyan]{obj_ply}[/]")
        _maybe_cuda_empty()
        gc.collect()
    
        if mesh_gaussians and not use_prebuilt_object_gaussians:
            try:
                mesh_path = resolve_insertion_mesh_path(gen_dir)
            except FileNotFoundError as exc:
                raise typer.BadParameter(
                    f"Missing {gen_dir / 'object_mesh.glb'} and {gen_dir / 'object_mesh.obj'}; "
                    "place an imported mesh, or run image-mode once without --reuse-object "
                    "so DreamGaussian exports stage-2 mesh."
                ) from exc
            mesh_ply = gen_dir / "object_mesh_gaussians.ply"
            console.print(f"[cyan]Mesh source (triangle asset → Gaussians):[/] {mesh_path.resolve()}")
            obj_ply = mesh_to_gaussian_ply(
                mesh_path,
                mesh_ply,
                num_points=30000,
                swap_red_yellow=swap_red_yellow,
                stylize_yellow_duck=swap_red_yellow,
                log=lambda s: console.print(s),
            )
            console.print(f"Converted mesh to Gaussian object PLY: [cyan]{Path(obj_ply).resolve()}[/]")
        elif use_prebuilt_object_gaussians:
            obj_ply = gen_dir / "object_mesh_gaussians.ply"
            if not obj_ply.is_file():
                raise typer.BadParameter(f"Expected staged Gaussian PLY at {obj_ply}")
            console.print(
                f"[bold]Insertion object PLY[/] (pre-built Gaussians, no mesh sampling): "
                f"[cyan]{obj_ply.resolve()}[/]"
            )
            if swap_red_yellow:
                swapped = gen_dir / "object_red_yellow_swapped.ply"
                obj_ply = swap_red_yellow_duck_ply(obj_ply, swapped)
                console.print(f"Swapped red/yellow duck colours: [cyan]{obj_ply}[/]")
    
        if auto_duck_realism and not mesh_render and insertion_mesh_source_exists(gen_dir):
            mesh_render = True
            console.print(
                "[cyan]Realism preset:[/] enabling [bold]--mesh-render[/] "
                "(sharp mesh + base 3DGS after sim; needs [italic]object_mesh.glb[/] or [italic]object_mesh.obj[/])."
            )
        elif auto_duck_realism and not mesh_render:
            console.print(
                "[yellow]Realism preset:[/] [italic]object_mesh.glb[/] / [italic]object_mesh.obj[/] missing — "
                "run once without [bold]--reuse-object[/] (or complete image-mode DG) "
                "so mesh-render can sharpen the final frames."
            )
    
        # Pick a real floor point that is actually visible in camera 0 (before
        # rescaling so we can match the object's base tilt to the local rug plane).
        place_x, place_y, place_z = find_camera_visible_surface_point(
            base_ply,
            model_out / "cameras.json",
            camera_index=0,
            target_u_frac=0.50,
            target_v_frac=0.78,
            opacity_threshold=0.0,
            z_quantile=50.0,
        )
        y_hint = place_y
        refined_y = refine_floor_contact_y(
            base_ply, place_x, place_z, floor_y_hint=y_hint
        )
        if refined_y is not None:
            new_y = max(place_y, refined_y)
            if abs(new_y - place_y) > 1e-4:
                console.print(
                    f"[cyan]Floor contact Y[/] (+Y-down): {place_y:.4f} → {new_y:.4f} "
                    f"(XZ-disk rug support, hint was frustum median)"
                )
            place_y = new_y
        floor_y = place_y  # In this scene +y is DOWN; floor is the largest y value.
        align_y = None
        duck_image = bool(image_mode) and "duck" in text.lower()
        duck_stand_upright = bool(not raw_object and duck_image and not rug_tilt)
        # Sketchfab / mesh-sampled / pre-built Gaussians: do NOT rotate the asset to
        # match a locally tilted rug plane (that reads as a leaning duck).  World
        # +ŷ is floor-down; use --rug-tilt if you explicitly want rug-normal alignment.
        external_asset_level_floor = bool(
            not raw_object
            and (mesh_gaussians or use_prebuilt_object_gaussians)
            and not rug_tilt
        )
        if not raw_object:
            if duck_stand_upright or external_asset_level_floor:
                # Rug slope fits often pick ∂y/∂z ≪ 0 (camera looks along -z), which reads as a
                # "lying on the rug" rotation for small toys.  Keep world-down for a natural stand.
                align_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                if external_asset_level_floor and not duck_stand_upright:
                    console.print(
                        "[cyan]Duck placement:[/] upright on level world floor "
                        "(external mesh / pre-built Gaussians — [bold]no rug-plane tilt[/]; "
                        "use [bold]--rug-tilt[/] to align feet to local rug normal)."
                    )
                else:
                    console.print(
                        "[cyan]Duck placement:[/] upright on level floor ([bold]--rug-tilt[/] to match rug plane)."
                    )
            elif auto_floor_tilt:
                a_sl, b_sl = estimate_floor_slopes_dy_dx_dz(
                    base_ply, place_x, place_z, floor_y_ref=floor_y
                )
                align_y = floor_downward_normal_from_slopes(a_sl, b_sl)
                tilt0 = float(np.linalg.norm(align_y - np.array([0.0, 1.0, 0.0], dtype=np.float64)))
                if abs(camera_base_pitch_deg) > 1e-6:
                    u_h = camera_right_horizontal_world(model_out / "cameras.json", 0)
                    align_y = rotate_unit_vector_around_axis_deg(align_y, u_h, camera_base_pitch_deg)
                tilt1 = float(np.linalg.norm(align_y - np.array([0.0, 1.0, 0.0], dtype=np.float64)))
                console.print(
                    f"Floor local fit: ∂y/∂x={a_sl:.4f}, ∂y/∂z={b_sl:.4f}  "
                    f"(geom |n-ŷ|={tilt0:.4f}  after cam pitch {camera_base_pitch_deg:+.1f}°: |n-ŷ|={tilt1:.4f})"
                )
            elif abs(camera_base_pitch_deg) > 1e-6:
                align_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                u_h = camera_right_horizontal_world(model_out / "cameras.json", 0)
                align_y = rotate_unit_vector_around_axis_deg(align_y, u_h, camera_base_pitch_deg)
                console.print(
                    f"Foot normal: camera-right pitch only ({camera_base_pitch_deg:+.1f}°, "
                    f"|n-ŷ|={float(np.linalg.norm(align_y - np.array([0.0, 1.0, 0.0]))):.4f})"
                )
    
        scaled = gen_dir / "object_scaled.ply"
        canonical_frame = "image" if image_mode else "text"
        if mesh_gaussians or use_prebuilt_object_gaussians:
            # Mesh / Sketchfab frame or pre-built mesh_gaussians PLY — not DG text/image tilt.
            canonical_frame = "mesh"
        if canonical_frame_override is not None:
            canonical_frame = canonical_frame_override
        scale_factor = object_scale_factor if object_scale_factor is not None else cfg.generation.object_scale_factor
        ao_strength = float(np.clip(contact_base_ao, 0.0, 0.95)) if not raw_object else 0.0
        if raw_object:
            rescale_object_ply_to_scene(
                obj_ply,
                base_ply,
                scale_factor,
                scaled,
                opacity_filter_logit=-1e9,
                spatial_outlier_sigma=None,
                reorient_to_world_y_down=False,
            )
            console.print("[yellow]raw-object mode: rescaler filters/reorient disabled.[/]")
        else:
            rescale_object_ply_to_scene(
                obj_ply,
                base_ply,
                scale_factor,
                scaled,
                opacity_filter_logit=opacity_filter_logit,
                canonical_frame=canonical_frame,
                post_rotation_deg=(post_rotate_x, post_rotate_y, post_rotate_z),
                align_y_axis_to=align_y,
                contact_base_occlusion=ao_strength,
            )
            console.print(f"Rescaler canonical_frame={canonical_frame}")
        n_scaled = len(PlyData.read(str(scaled))["vertex"])
        console.print(f"Object Gaussians after opacity filter: {n_scaled}")
        console.print(
            f"[green]Rescale input object PLY:[/] [cyan]{Path(obj_ply).resolve()}[/] "
            f"(this is what gets merged + simulated as the duck)"
        )

    # Count Gaussians in base scene and generated object so we can isolate
    # object indices later for physics (simulating the whole scene is wrong).
    base_pdata = PlyData.read(str(base_ply))
    n_base = len(base_pdata["vertex"])
    scaled_pdata = PlyData.read(str(scaled))
    n_obj = len(scaled_pdata["vertex"])
    y_obj = np.asarray(scaled_pdata["vertex"]["y"], dtype=np.float64)
    del base_pdata
    del scaled_pdata

    # +Y is DOWN: the object's contact with the floor is the maximum world y of
    # its Gaussians.  ``rescale_object_ply_to_scene`` (with reorient) sets that
    # foot plane to y=0 in object-local space, so merge translation y should be
    # ``floor_y`` (not ``floor_y - epsilon``), which used to float the duck.  For
    # ``--raw-object`` (no reorient), align using the current bbox max y.
    foot_local_y = float(np.max(y_obj))
    eps = float(cfg.physics.placement_floor_contact_y_eps)
    y_place = float(floor_y) - foot_local_y + eps
    if (not raw_object) and foot_local_y > 0.02:
        console.print(
            f"[yellow]Placement note:[/] scaled object max(y)={foot_local_y:.4f} "
            f"(expected ≈0 after reorient); merge y uses floor_y − max(y)+eps."
        )
    console.print(
        f"Placement target: ({place_x:.3f}, {floor_y:.3f}, {place_z:.3f})  "
        f"[camera-image (u=0.50, v=0.78) → rug]  "
        f"merge_y={y_place:.4f} (floor contact: max_y_local={foot_local_y:.4f} + eps={eps:g})"
    )
    if hold_for_review:
        if from_review:
            raise typer.BadParameter("--hold-for-review cannot be used with --continue-after-review.")
        ck = _save_mode_a_review_checkpoint(
            gen_dir,
            text=text,
            yellow_duck=yellow_duck_eff,
            scaled=scaled,
            obj_ply=obj_ply,
            place_x=place_x,
            place_y=place_y,
            place_z=place_z,
            floor_y=floor_y,
            canonical_frame=canonical_frame,
            scale_factor=scale_factor,
            post_rotate=(post_rotate_x, post_rotate_y, post_rotate_z),
            opacity_filter_logit=opacity_filter_logit,
            contact_base_ao=contact_base_ao,
            image_mode=image_mode,
            raw_object=raw_object,
            mesh_render=mesh_render,
            mesh_skin_k=mesh_skin_k,
            swap_red_yellow=swap_red_yellow,
            mesh_gaussians=mesh_gaussians,
            align_y=align_y,
        )
        ref_png = gen_dir / "reference" / "reference.png"
        extra = (
            f"  • SD reference image: [cyan]{ref_png.resolve()}[/]\n" if ref_png.is_file() else ""
        )
        console.print(
            "[bold yellow]Stopped for your review[/] ([bold]--hold-for-review[/]). "
            "Inspect the duck shape, then continue only if satisfied:\n"
            f"  • scaled object PLY: [cyan]{scaled.resolve()}[/]\n"
            f"{extra}"
            f"  • checkpoint: [cyan]{ck.resolve()}[/]\n\n"
            "Next step — same prompt + [bold]--continue-after-review[/] "
            "→ merge into scene, physics, video.\n"
            "To reject: remove the generated PLYs / reference under [cyan]sds_run[/] and rerun without hold."
        )
        return

    merged = cfg.resolve(cfg.paths.merged_output) / "mode_a_merged.ply"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merge_ply_files(
        str(base_ply), str(scaled),
        np.array([place_x, y_place, place_z], dtype=np.float64),
        str(merged),
    )

    place_vec = np.array([place_x, y_place, place_z], dtype=np.float64)
    mesh_verts_w: np.ndarray | None = None
    mesh_faces_np: np.ndarray | None = None
    if mesh_render:
        if raw_object:
            raise typer.BadParameter("--mesh-render cannot be used with --raw-object.")
        if use_prebuilt_object_gaussians and not insertion_mesh_source_exists(gen_dir):
            raise typer.BadParameter(
                "--mesh-render needs object_mesh.glb or object_mesh.obj under sds_run for the sharp overlay. "
                "Omit --mesh-render when using only --object-gaussians, or also pass --asset-path to your .glb."
            )
        try:
            overlay_mesh_path = resolve_insertion_mesh_path(gen_dir)
        except FileNotFoundError as exc:
            raise typer.BadParameter(
                f"Missing {gen_dir / 'object_mesh.glb'} and {gen_dir / 'object_mesh.obj'}. "
                "Use --asset-path for your Sketchfab mesh, or disable --mesh-render."
            ) from exc
        placed_mesh_path = gen_dir / "placed_object_mesh.obj"
        mesh_verts_w, mesh_faces_np = export_placed_dreamgaussian_mesh(
            overlay_mesh_path,
            placed_mesh_path,
            object_ply_path=obj_ply,
            base_ply_path=base_ply,
            scale_factor=scale_factor,
            canonical_frame=canonical_frame,
            post_rotation_deg=(post_rotate_x, post_rotate_y, post_rotate_z),
            align_y_axis_to=align_y,
            merge_translation=place_vec,
        )
        console.print(
            f"[bold]Mesh overlay[/] (same transform as Gaussians): source [cyan]{overlay_mesh_path.resolve()}[/] "
            f"→ placed [cyan]{placed_mesh_path.resolve()}[/]"
        )

    if no_physics:
        console.print(f"Merged: [cyan]{merged}[/]  (--no-physics: skipped simulation)")
        if mesh_render and mesh_verts_w is not None and mesh_faces_np is not None:
            sim_run = cfg.resolve(cfg.paths.sim_output) / "mode_a_run"
            (sim_run / "frames").mkdir(parents=True, exist_ok=True)
            rgb = composite_mesh_over_base_gs(
                colmap_scene=colmap_scene,
                model_path=model_out,
                iteration=iters,
                base_ply=base_ply,
                mesh_vertices=mesh_verts_w,
                mesh_faces=mesh_faces_np,
                camera_index=0,
            )
            write_png(sim_run / "frames" / "0000.png", rgb)
            console.print(
                f"[green]Mesh composite[/] (base 3DGS + DG mesh): [cyan]{sim_run / 'frames' / '0000.png'}[/]"
            )
        return

    frame_num_override = 1 if preview else cfg.physics.frame_num
    if preview:
        console.print("[yellow]preview mode:[/] ONE output frame (kinematic or MPM).")
    sim_run = cfg.resolve(cfg.paths.sim_output) / "mode_a_run"
    track_debug = bool(wobble_debug and in_place_wobble)
    save_gauss_for_mesh = bool(mesh_render and in_place_wobble and not wobble_use_mpm)

    if in_place_wobble and not wobble_use_mpm:
        from rendering.kinematic_wobble_render import render_kinematic_wobble_sequence

        pb = cfg.physics.compile_video_playback_sec
        if pb is None:
            pb = 10.0
        console.print(
            "[bold cyan]Kinematic in-place wobble[/] (no MPM): "
            f"amp={wobble_amp} m  f={wobble_frequency} Hz  height_exp={wobble_height_weight}  "
            f"bottom_pin={wobble_bottom_pin}"
        )
        vid = render_kinematic_wobble_sequence(
            colmap_scene=colmap_scene,
            model_path=model_out,
            iteration=iters,
            merged_ply=merged,
            n_base=n_base,
            n_obj=n_obj,
            sim_run=sim_run,
            frame_num=frame_num_override,
            frame_dt=cfg.physics.frame_dt,
            playback_seconds=float(pb),
            camera_index=0,
            wobble_amp=float(wobble_amp),
            wobble_frequency=float(wobble_frequency),
            wobble_height_weight=float(wobble_height_weight),
            wobble_bottom_pin=float(wobble_bottom_pin),
            track_debug=track_debug or save_gauss_for_mesh,
        )
        console.print(f"Merged: [cyan]{merged}[/]  Video: [green]{vid}[/]  (kinematic wobble)")
        if wobble_debug:
            from physics.wobble_debug import log_wobble_motion_from_gauss_npy

            log_wobble_motion_from_gauss_npy(sim_run, n_obj=n_obj, console=console)
        if mesh_render and mesh_verts_w is not None and mesh_faces_np is not None:
            vid_mesh = run_mesh_rigid_overlay_sequence(
                sim_run=sim_run,
                colmap_scene=colmap_scene,
                model_path=model_out,
                iteration=iters,
                base_ply=base_ply,
                mesh_vertices_world_rest=mesh_verts_w,
                mesh_faces=mesh_faces_np,
                frame_dt=cfg.physics.frame_dt,
                playback_seconds=float(pb),
                camera_index=0,
                mesh_skin_k=max(3, min(32, int(mesh_skin_k))),
            )
            console.print(
                f"[green]Mesh skin overlay[/] (KNN from kinematic Gaussians) + video: [cyan]{vid_mesh}[/]"
            )
        return

    # Simulate ONLY the generated object Gaussians (indices [n_base, n_base+n_obj)).
    # Simulating the full merged PLY applies physics to the background scene, which
    # destroys the render.
    sim_cfg_dir = cfg.resolve(cfg.paths.sim_output)
    sim_cfg_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = sim_cfg_dir / "phys_mode_a.json"
    sim_idx_path = sim_cfg_dir / "mode_a_obj_indices.npy"
    gply = PlyData.read(str(merged))
    v = gply["vertex"]
    pos = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    obj_idx = np.arange(n_base, n_base + n_obj, dtype=np.int64)
    np.save(sim_idx_path, obj_idx)
    console.print(f"Physics: simulating {len(obj_idx)} object Gaussians / {len(pos)} total")

    diag = validate_merged_object_for_mpm(merged, obj_idx)

    tpl_file = default_template_dir() / f"{material}.json"
    # --smoke only swaps the 3DGS checkpoint; physics always runs at full
    # resolution / length so the duck wobble is properly visible.  Selecting
    # via simulate_indices_npy guarantees only the duck Gaussians move while
    # the rug / table / TV stand stay rigid.  ``floor_collider=True`` adds a
    # slip surface at ``floor_y`` so the duck actually lands on the rug and
    # the jelly material gets to deform on impact (instead of falling forever).
    #
    # MPM ``subtract_rigid_drift`` applies only with ``--in-place-wobble --wobble-use-mpm``.
    subtract_drift = bool(in_place_wobble and wobble_use_mpm and not no_pin_com_drift)
    if in_place_wobble and wobble_use_mpm and not no_pin_com_drift:
        console.print(
            "[cyan]MPM in-place wobble:[/] COM pinning [bold]on[/] (subtract_rigid_drift) — "
            "shear velocity slabs on upper vs mid body."
        )
    mat_overrides: dict[str, float] = {}
    if material == "jelly":
        if cfg.physics.mode_a_jelly_E is not None:
            mat_overrides["E"] = float(cfg.physics.mode_a_jelly_E)
        if cfg.physics.mode_a_jelly_grid_v_damping_scale is not None:
            mat_overrides["grid_v_damping_scale"] = float(
                cfg.physics.mode_a_jelly_grid_v_damping_scale
            )
    _sim_json, sim_area_list = generate_phys_config(
        obj_idx,
        pos,
        material,
        sim_cfg,
        template_path=tpl_file if tpl_file.is_file() else None,
        n_grid=cfg.physics.n_grid,
        frame_num=frame_num_override,
        frame_dt=cfg.physics.frame_dt,
        substep_dt=cfg.physics.substep_dt,
        gravity=float(cfg.physics.gravity) * float(cfg.physics.gravity_scale),
        floor_y=floor_y,
        simulate_indices_npy=sim_idx_path,
        subtract_rigid_drift=subtract_drift,
        world_up=(0.0, -1.0, 0.0),
        floor_collider=True,
        floor_friction=cfg.physics.floor_friction,
        in_place_wobble=bool(in_place_wobble and wobble_use_mpm),
        wobble_velocity=cfg.physics.wobble_velocity,
        wobble_end_time=cfg.physics.wobble_end_time,
        sim_area_margin=cfg.physics.sim_area_margin,
        enable_internal_particle_fill=cfg.physics.enable_mpm_particle_filling,
        material_overrides=mat_overrides if mat_overrides else None,
    )
    log_phys_preflight(
        console,
        diag,
        sim_area_list,
        n_grid=cfg.physics.n_grid,
        substep_dt=cfg.physics.substep_dt,
        frame_dt=cfg.physics.frame_dt,
        material=material,
        floor_friction=cfg.physics.floor_friction,
        gravity_y_cfg=float(cfg.physics.gravity) * float(cfg.physics.gravity_scale),
    )
    if in_place_wobble and wobble_use_mpm:
        sj = json.loads(sim_cfg.read_text(encoding="utf-8"))
        console.print(
            f"[cyan]MPM in-place wobble[/]  subtract_rigid_drift={sj.get('subtract_rigid_drift', False)}  "
            f"wobble_velocity={cfg.physics.wobble_velocity}  wobble_end_time={cfg.physics.wobble_end_time}s  "
            f"frames={frame_num_override}  frame_dt={cfg.physics.frame_dt}"
        )
        for bc in sj.get("boundary_conditions", []):
            if bc.get("type") == "enforce_particle_translation":
                console.print(
                    f"  [cyan]wobble BC[/] velocity={bc.get('velocity')}  "
                    f"time [{bc.get('start_time')}, {bc.get('end_time')})  "
                    f"point={bc.get('point')}  size={bc.get('size')}"
                )
        console.print(
            "[dim]Render path:[/] PhysGaussian updates merged Gaussian means each frame from MPM; "
            "static mesh overlay is [bold]not[/] used unless [bold]--mesh-render[/]."
        )
    if wobble_debug and not in_place_wobble:
        console.print("[yellow]--wobble-debug ignored[/] (requires --in-place-wobble).")
    vid = run_simulation(
        str(merged),
        str(sim_cfg),
        str(sim_run),
        camera_scene_path=str(model_out),
        playback_seconds=cfg.physics.compile_video_playback_sec,
        save_obj_centroid=mesh_render or track_debug,
        save_obj_gauss_xyz=mesh_render or track_debug,
        save_obj_tracks_for_debug=track_debug,
        taichi_device_memory_gb=float(cfg.physics.taichi_device_memory_gb),
    )
    console.print(f"Merged: [cyan]{merged}[/]  Video: [green]{vid}[/]")
    if track_debug:
        from physics.wobble_debug import log_wobble_motion_from_gauss_npy

        log_wobble_motion_from_gauss_npy(sim_run, n_obj=n_obj, console=console)
    if mesh_render and mesh_verts_w is not None and mesh_faces_np is not None:
        vid_mesh = run_mesh_rigid_overlay_sequence(
            sim_run=sim_run,
            colmap_scene=colmap_scene,
            model_path=model_out,
            iteration=iters,
            base_ply=base_ply,
            mesh_vertices_world_rest=mesh_verts_w,
            mesh_faces=mesh_faces_np,
            frame_dt=cfg.physics.frame_dt,
            playback_seconds=cfg.physics.compile_video_playback_sec,
            camera_index=0,
            mesh_skin_k=max(3, min(32, int(mesh_skin_k))),
        )
        console.print(
            f"[green]Mesh rigid overlay[/] (centroid-tracked DG mesh on base 3DGS) + re-encoded video: "
            f"[cyan]{vid_mesh}[/]"
        )


def _write_dg_standalone_preview_topdown(ply_path: Path, out_png: Path, *, size: int = 512) -> None:
    """Lightweight XZ orthographic dump of Gaussian centres (debug; not a full GS render)."""
    from PIL import Image

    ply = PlyData.read(str(ply_path))
    vx = ply["vertex"]
    x = np.asarray(vx["x"], dtype=np.float64)
    z = np.asarray(vx["z"], dtype=np.float64)
    y = np.asarray(vx["y"], dtype=np.float64)
    if x.size == 0:
        raise ValueError(f"empty PLY: {ply_path}")
    xm, xM = float(x.min()), float(x.max())
    zm, zM = float(z.min()), float(z.max())
    ym, yM = float(y.min()), float(y.max())
    pad = 1e-6
    xr = (x - xm) / (xM - xm + pad)
    zr = (z - zm) / (zM - zm + pad)
    yn = (y - ym) / (yM - ym + pad)
    H = W = int(size)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ii = np.clip((xr * (W - 1)).round().astype(np.int64), 0, W - 1)
    jj = np.clip((zr * (H - 1)).round().astype(np.int64), 0, H - 1)
    r = (40 + 215 * yn).astype(np.uint8)
    g = (180 + 60 * (1.0 - yn)).astype(np.uint8)
    b = (20 + 80 * yn).astype(np.uint8)
    img[jj, ii] = np.stack([r, g, b], axis=1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(str(out_png))


@app.command("dg-asset")
def dg_asset(
    text: str = typer.Option(
        _YELLOW_RUBBER_DUCK_TEXT,
        "--text",
        help="Prompt for DreamGaussian text-to-3D (ignored when --image is set).",
    ),
    image: Path | None = typer.Option(
        None,
        "--image",
        exists=True,
        readable=True,
        help="Run image-to-3D from this PNG (rembg + stable-zero123) instead of text SDS.",
    ),
    out: Path = typer.Option(
        PROJECT_ROOT / "output" / "dg_standalone" / "last",
        "--out",
        help="Output directory (PLY, logs, preview_topdown.png, meta.json).",
    ),
    iters: int = typer.Option(500, "--iters", help="DreamGaussian stage-1 iterations."),
    mvdream: bool = typer.Option(
        True,
        "--mvdream/--no-mvdream",
        help="Text-to-3D only: use MVDream when installed (recommended).",
    ),
) -> None:
    """Run DreamGaussian alone (no 3DGS scene, merge, or physics) to validate asset quality."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[bold]dg-asset[/] out={out} iters={iters} "
        f"mode={'image' if image is not None else 'text'} mvdream={bool(image is None and mvdream)}"
    )
    if image is not None:
        ply = generate_object_from_image(
            image.resolve(),
            out,
            num_steps=int(iters),
            elevation=0.0,
            use_stable_zero123=True,
        )
    else:
        prompt = enhance_generation_prompt(text)
        if prompt != text:
            console.print(f"[dim]enhanced prompt:[/] {prompt}")
        ply = generate_object(
            prompt,
            out,
            num_steps=int(iters),
            use_mvdream=mvdream,
        )
    nv = len(PlyData.read(str(ply))["vertex"])
    meta = {
        "ply": str(ply.resolve()),
        "vertex_count": int(nv),
        "text": text,
        "image": str(image.resolve()) if image is not None else None,
        "iters": int(iters),
        "mvdream": bool(mvdream) if image is None else None,
    }
    prev = out / "preview_topdown.png"
    _write_dg_standalone_preview_topdown(Path(ply), prev)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    console.print(f"[green]Done:[/] {ply}  ({nv} vertices)")
    console.print(f"[green]Preview (XZ top-down):[/] {prev}")


@app.command("regen-mesh-assets")
def regen_mesh_assets(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs" / "pipeline_config.yaml",
        "--config",
    ),
    num_points: int = typer.Option(30000, "--num-points", help="Surface samples for mesh_to_gaussian_ply."),
    copy_object_model: bool = typer.Option(
        True,
        "--copy-object-model/--no-copy-object-model",
        help="Copy object_mesh_gaussians.ply → object_model.ply for --reuse-object size gate.",
    ),
    no_axis_rotation: bool = typer.Option(
        False,
        "--no-axis-rotation",
        help="Pass through to mesh_to_gaussian_ply (raw OBJ frame; not for SuperSplat default).",
    ),
) -> None:
    """Rebuild ``object_mesh_gaussians.ply`` (and optionally ``object_model.ply``) from mesh import.

    Uses ``object_mesh.glb`` if present, else ``object_mesh.obj``.
    Re-run ``mode-a`` with ``--reuse-object --mesh-gaussians`` (and remove stale
    ``object_scaled.ply`` / ``mode_a_review.json`` if you need a fresh scaled object).
    """
    cfg = PipelineConfig.load(config)
    gen_dir = cfg.resolve(cfg.paths.generated_objects) / "sds_run"
    gen_dir.mkdir(parents=True, exist_ok=True)
    try:
        mesh = resolve_insertion_mesh_path(gen_dir)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"Missing {gen_dir / 'object_mesh.glb'} and {gen_dir / 'object_mesh.obj'}."
        ) from exc
    console.print(f"[cyan]regen-mesh-assets source:[/] {mesh}")
    out_g = gen_dir / "object_mesh_gaussians.ply"
    mesh_to_gaussian_ply(
        mesh,
        out_g,
        num_points=int(num_points),
        apply_export_axis_rotation=not bool(no_axis_rotation),
        log=lambda s: console.print(s),
    )
    console.print(f"[green]Wrote[/] {out_g}")
    if copy_object_model:
        dest = gen_dir / "object_model.ply"
        shutil.copyfile(out_g, dest)
        console.print(f"[green]Copied →[/] {dest}  (--reuse-object gate)")
    console.print(
        "[yellow]For a new object_scaled.ply:[/] remove "
        f"{gen_dir / 'object_scaled.ply'} and {gen_dir / MODE_A_REVIEW_JSON} if present, "
        "then run [bold]mode-a[/] with the same flags as before "
        "([bold]--reuse-object --mesh-gaussians[/], etc.)."
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
