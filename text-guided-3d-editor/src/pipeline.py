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
from config import PROJECT_ROOT, PipelineConfig, get_sim_run_dir
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


def _mode_b_cleanup_minimal_artifacts(mode_b_out: Path) -> None:
    """Delete bulky mode-b intermediates; keep phys_config, indices, and main mp4s."""
    video_dir = mode_b_out / "video"
    keep_mp4 = {"output.mp4", "final_chair_only_collapse.mp4", "final_cleanup_fullframe.mp4"}
    if video_dir.is_dir():
        for p in list(video_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".mp4" and p.name not in keep_mp4:
                try:
                    p.unlink()
                except OSError:
                    pass
    for name in (
        "debug",
        "debug_selection",
        "frames",
        "masks_objmask",
        # Keep matte compositing layers (foreground RGB, alpha, composite frames).
        "frames_final_chair_only",
        "frames_composited_objmask",
    ):
        shutil.rmtree(mode_b_out / name, ignore_errors=True)
    for p in mode_b_out.glob("frames_composited_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for name in (
        "render_camera_rgb.png",
        "selected_red_overlay_training_cam.png",
        "selection_trace.json",
        "camera_usage.txt",
        "surface_gaussian_indices.npy",
    ):
        q = mode_b_out / name
        if q.is_file():
            try:
                q.unlink()
            except OSError:
                pass


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
            console.print(f"[dim]{label}[/] CUDA unavailable")
            return
        dev = torch.cuda.current_device()
        if hasattr(torch.cuda, "mem_get_info"):
            free_b, total_b = torch.cuda.mem_get_info(dev)
            console.print(
                f"[dim]{label}[/] CUDA free [bold]{free_b // 1048576}[/] MiB / "
                f"total {total_b // 1048576} MiB  (details: [cyan]nvidia-smi[/])"
            )
        else:
            a = torch.cuda.memory_allocated(dev) // 1048576
            r = torch.cuda.memory_reserved(dev) // 1048576
            console.print(
                f"[dim]{label}[/] torch allocated≈{a} MiB  reserved≈{r} MiB  ([cyan]nvidia-smi[/])"
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[dim]{label}[/] GPU memory query failed ({exc}); try [cyan]nvidia-smi[/]")


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
    out_rel = (
        str(cfg.scene.model_output_smoke or cfg.scene.model_output)
        if smoke
        else str(cfg.scene.model_output_full or cfg.scene.model_output)
    )
    model_out = cfg.resolve(out_rel)
    iters = cfg.scene.training_iterations_low if smoke else cfg.scene.training_iterations
    console.print(f"[bold]Training 3DGS[/] iters={iters}")
    ply = train_scene(colmap_scene, model_out, iterations=iters)
    console.print(f"PLY: [green]{ply}[/]")
    _maybe_cuda_empty()


@app.command()
def mode_b(
    text: str | None = typer.Argument(None, help="Object description (ignored when --preset is set)"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Named mode-b profile from config mode_b_presets (e.g. desk_jelly, footrest_sand).",
    ),
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
    cc_bypass: bool = typer.Option(
        False,
        "--cc-bypass",
        help="DIAGNOSTIC: skip CC filtering entirely (final selection = shell set, before completion).",
    ),
    cc_keep_topk: int | None = typer.Option(
        None,
        "--cc-keep-topk",
        help="DIAGNOSTIC: keep top-K mask-plausible connected components (overrides thresholds); "
        "requires cc strategy mask_components.",
    ),
    stop_after_selection: bool = typer.Option(
        False,
        "--stop-after-selection",
        help="Exit after mask→3D selection (+ bbox/shell/CC + completion) and write debug artifacts. "
        "Skips kinematic wobble / PhysGaussian MPM.",
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
    allow_cpu_fallback: bool = typer.Option(
        False,
        "--allow-cpu-fallback",
        help="If GPU Grounded-SAM2 / GroundingDINO CUDA ops fail, retry once on CPU (slow). "
        "If not set (default), segmentation is GPU-only and failures abort loudly.",
    ),
) -> None:
    cfg = PipelineConfig.load(config)
    mode_b_preset = None
    if preset is None:
        raise typer.BadParameter(
            "mode-b requires --preset (e.g. --preset footrest_sand). "
            "This avoids accidentally writing to the legacy mode_b_jelly folder."
        )
    if preset not in cfg.mode_b_presets:
        raise typer.BadParameter(
            f"Unknown --preset {preset!r}. Available: {sorted(cfg.mode_b_presets.keys())}"
        )
    mode_b_preset = cfg.mode_b_presets[preset]
    text = mode_b_preset.selection_prompt

    # Always log the final resolved prompt/preset early (helps reproduce runs).
    console.print(
        f"[mode-b debug] preset={preset}  "
        f"segmentation_prompt={text!r}  "
        f"reuse_selection={bool(reuse_selection)}  "
        f"debug_selection={bool(debug_selection)}  "
        f"sam2_render_camera_only={bool(sam2_render_camera_only)}"
    )
    # Propagate behavior to segmentation helpers (GPU-only by default).
    try:
        cfg.segmentation.allow_cpu_fallback = bool(allow_cpu_fallback)
    except Exception:
        pass

    resolved_out_subdir = str(mode_b_preset.output_subdir)
    mode_b_out = cfg.resolve(cfg.paths.sim_output) / resolved_out_subdir
    mode_b_out.mkdir(parents=True, exist_ok=True)

    # Per-preset physics config overrides (do not mutate cfg.physics globally).
    phys = cfg.physics
    if mode_b_preset is not None and mode_b_preset.physics_overrides:
        phys = type(cfg.physics).model_validate(
            {**cfg.physics.model_dump(), **dict(mode_b_preset.physics_overrides)}
        )
    material_name = str(mode_b_preset.physics_type) if mode_b_preset is not None else "jelly"

    console.print(
        f"[mode-b preset] preset={preset or 'none'}  "
        f"prompt={text!r}  "
        f"physics_type={material_name}  "
        f"out={mode_b_out}"
    )
    def _resolve_base_scene(*, use_smoke_ckpt: bool) -> tuple[Path, int]:
        """
        Resolve base scene output dir + 3DGS iteration.

        Rules (main repo only, no submodule edits):
        - If config provides `scene.model_output_full/smoke`, select by `use_smoke_ckpt`.
        - Otherwise fall back to legacy `scene.model_output`.
        - Iteration is chosen to match the selected base output:
          - smoke output → `training_iterations_low`
          - full output  → `training_iterations`
        """
        # Pick output dir.
        out_rel: str
        if use_smoke_ckpt:
            out_rel = str(cfg.scene.model_output_smoke or cfg.scene.model_output)
        else:
            out_rel = str(cfg.scene.model_output_full or cfg.scene.model_output)
        model_out = cfg.resolve(out_rel)

        # Pick iteration (match the chosen output dir).
        smoke_out = cfg.scene.model_output_smoke
        is_smoke_out = bool(smoke_out) and Path(out_rel).as_posix() == Path(str(smoke_out)).as_posix()
        iters = int(cfg.scene.training_iterations_low if (use_smoke_ckpt or is_smoke_out) else cfg.scene.training_iterations)
        return model_out, iters

    def _list_existing_iterations(scene_out: Path) -> list[int]:
        pc_dir = scene_out / "point_cloud"
        if not pc_dir.is_dir():
            return []
        out: list[int] = []
        for p in pc_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if not name.startswith("iteration_"):
                continue
            try:
                out.append(int(name.split("_", 1)[1]))
            except Exception:
                continue
        return sorted(set(out))

    colmap_scene = cfg.resolve(cfg.scene.data_root) / cfg.scene.scene_name
    preset_smoke = bool(getattr(mode_b_preset, "use_smoke_3dgs", False))
    # CLI flags override preset defaults.
    use_smoke_3dgs_checkpoint = bool(smoke or smoke_3dgs or (preset_smoke and not (smoke or smoke_3dgs)))
    model_out, iters = _resolve_base_scene(use_smoke_ckpt=use_smoke_3dgs_checkpoint)
    ply_path = model_out / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    console.print(
        f"[mode-b] base_scene={model_out}  iters={iters}  ply={ply_path}  "
        f"(smoke_flag={bool(smoke or smoke_3dgs)} preset_smoke={preset_smoke})"
    )
    if not ply_path.is_file():
        found = _list_existing_iterations(model_out)
        raise typer.BadParameter(
            "Missing base-scene checkpoint.\n"
            f" - preset: {preset or 'none'}\n"
            f" - base scene: {model_out}\n"
            f" - expected iteration: {iters}\n"
            f" - missing path: {ply_path}\n"
            f" - found iterations: {found if found else '[] (no point_cloud/iteration_*/ found)'}\n"
            "Fix by training the base scene (train / train --smoke) or updating scene.model_output_full/smoke + iterations."
        )
    if debug_mask_to_gaussians_only:
        from segmentation.mask_projection_debug_run import run_mask_projection_debug

        run_mask_projection_debug(
            cfg=cfg,
            text=text,
            mode_b_out=mode_b_out,
            smoke=smoke,
            smoke_3dgs=smoke_3dgs,
            force_rerender_views=force_rerender_views,
            debug_mask_to_gaussians_only=True,
            projection_stride=1,
            console=console,
            maybe_cuda_empty=_maybe_cuda_empty,
        )
        return
    _mpm_grid = int(phys.n_grid_low if smoke else phys.n_grid)
    _mpm_frames = int(phys.mode_b_mpm_smoke_frames if smoke else phys.frame_num)
    _taichi_gb = float(taichi_memory_gb) if taichi_memory_gb is not None else float(phys.taichi_device_memory_gb)
    console.print(
        f"[bold]mode-b[/] smoke_3dgs={bool(use_smoke_3dgs_checkpoint)} iters={iters} "
        f"MPM(n_grid={_mpm_grid}, frames={_mpm_frames}) taichi_mem_GB={_taichi_gb}"
    )

    from plyfile import PlyData

    gply = PlyData.read(str(ply_path))
    v = gply["vertex"]
    pos = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
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
    reused_surface_cache: Path | None = None
    # Cache for selection completion: training_camera_index -> (mask, depth, K, w2c).
    # Populated when we run Grounded-SAM2; can be lazily populated later for a single camera.
    mask_view_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if reuse_selection:
        # IMPORTANT: per-preset selection must not reuse the legacy global cache
        # (it usually contains the desk selection). Keep desk_jelly tied to that
        # cache even if its output folder is renamed.
        cached_candidates: list[Path] = [surface_indices_path]
        global_cache = cfg.resolve("output/3d_gaussian_selection_debug/selected_indices.npy")
        if preset == "desk_jelly" or (resolved_out_subdir == "mode_b_jelly" and preset is None):
            cached_candidates.insert(0, global_cache)
        for cached in cached_candidates:
            if cached.is_file():
                idx_surface = np.load(cached).astype(np.int64)
                reused_surface_cache = cached
                console.print(f"Reusing cached surface indices: [cyan]{cached}[/] ({len(idx_surface)} Gaussians)")
                break

    if idx_surface is not None:
        console.print(
            f"[mode-b selection] preset={preset or 'none'}  prompt={text!r}  "
            f"reuse_selection={bool(reuse_selection)}  reused_surface_cache={str(reused_surface_cache) if reused_surface_cache else None}"
        )

    if idx_surface is None:
        console.print(
            f"[mode-b selection] preset={preset or 'none'}  prompt={text!r}  "
            f"reuse_selection={bool(reuse_selection)}  cached_surface_exists={surface_indices_path.is_file()}"
        )
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
        # IMPORTANT: forced_overlay_cam is used for later debug/video camera selection, but should
        # NOT implicitly reduce Grounded-SAM2 to a single view. Otherwise consensus_min_votes (>=2)
        # can zero out the selection when multi_view_count=1.
        if sam2_render_camera_only:
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
            mask_dbg_info: dict = {}
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
                allow_cpu_fallback=bool(getattr(seg, "allow_cpu_fallback", False)),
                debug_mask_dir=mask_dbg,
                debug_stem=f"rgb_{stem}",
                debug_info=mask_dbg_info,
            )
            h_m, w_m = mask.shape
            true_px = int(mask.sum())
            bbox = mask_dbg_info.get("mask_bbox_xyxy")
            area_ratio = mask_dbg_info.get("mask_area_ratio")
            is_fallback_rect = bool(mask_dbg_info.get("is_fallback_rect", False))
            console.print(
                f"[mode-b mask] stem={stem}  hw={h_m}x{w_m}  true_px={true_px}  "
                f"area_ratio={float(area_ratio) if area_ratio is not None else None}  "
                f"bbox_xyxy={bbox}  fallback_rect={is_fallback_rect}"
            )
            if is_fallback_rect:
                console.print(
                    "[yellow][mode-b] Rejecting fallback rectangular mask (segmentation failed).[/]"
                )
                continue
            if int(seg.mask_erode_iters) > 0:
                from scipy.ndimage import binary_erosion

                mask = binary_erosion(mask, iterations=int(seg.mask_erode_iters))
            depth = np.load(render_dir / f"depth_{stem}.npy")
            # Persist for completion pass (use training cam index = local_idx * stride).
            try:
                cam_train_idx = int(stem) * int(vs)
            except Exception:
                cam_train_idx = 0
            mask_view_cache[int(cam_train_idx)] = (mask.astype(bool), depth.astype(np.float32), K, w2c)
            stats_n: dict[str, int | float] = {}
            stats_d: dict[str, int | float] = {}
            idx_n = mask_to_gaussian_indices(
                mask,
                depth,
                K,
                w2c,
                pos,
                distance_threshold=float(seg.mask_3d_distance_threshold_m),
                stride=int(seg.mask_stride),
                use_depth_consistency=False,
                stats=stats_n,
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
                stats=stats_d,
            )
            from segmentation.mask_to_gaussians import projection_stage_counts

            ss = max(1, int(np.ceil(pos.shape[0] / 200_000)))
            proj_counts = projection_stage_counts(mask, pos, K, w2c, subsample=ss)
            console.print(
                f"[mode-b proj] tested={proj_counts['tested']} (subsample={proj_counts['subsample']})  "
                f"in_bounds={proj_counts['projected_in_bounds']}  inside_mask={proj_counts['projected_inside_mask']}  "
                f"nearest_unique={int(stats_n.get('nearest_after_knn_unique', 0))}  "
                f"depth_unique_after_votes={int(stats_d.get('depth_unique_after_vote_min_votes', 0))}"
            )
            if idx_n.size == 0 and int(stats_n.get("mask_true_pixels", 0)) > 0:
                # If kNN-on-unprojected points returns empty (often due to depth=0 or too-tight distance),
                # fall back to projection-only (no depth/knn) to avoid a hard 0-selection.
                from segmentation.mask_to_gaussians import indices_project_inside_mask

                idx_p = indices_project_inside_mask(mask, pos, K, w2c, subsample=1)
                console.print(
                    f"[mode-b] mask→3D nearest yielded 0 (mask_true={int(stats_n.get('mask_true_pixels', 0))}, "
                    f"valid_z={int(stats_n.get('nearest_samples_valid_z', 0))}); "
                    f"projection-only fallback={len(idx_p)}"
                )
                idx_n = idx_p
                if idx_d.size == 0:
                    idx_d = idx_p
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

        if not sets_nearest or not sets_depth:
            raise typer.BadParameter(
                "No valid segmentation views (all masks were fallback rectangles or rejected). "
                "Try --sam2-render-camera-only/--render-camera-index or increase multi_view_count."
            )

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
        r_shell = phys.mode_b_shell_radius_m if shell_radius_m is None else float(shell_radius_m)
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

        if bool(cc_bypass):
            idx_cc = idx_shell
            console.print(f"[yellow]CC bypass enabled: using shell set ({len(idx_cc)} Gaussians).[/]")
        else:
            cc_strategy = str(getattr(phys, "mode_b_cc_keep_strategy", "single")).strip().lower()
        if cc_strategy == "mask_components":
            # Keep multiple components that genuinely overlap the target SAM mask in the render camera.
            # This is robust when the object is fragmented (seat/backrest split) and the seed-touching
            # "largest component" would drop the backrest.
            # NOTE: the final render_camera_index is chosen later (visibility suite). For CC filtering
            # we use a stable provisional training camera index: CLI --camera-index if provided,
            # otherwise 0. If needed, lazily compute SAM2 mask for that view.
            provisional_cam = int(camera_index) if camera_index is not None else 0
            view = mask_view_cache.get(int(provisional_cam)) if "mask_view_cache" in locals() else None
            if view is None:
                try:
                    render_dir = cfg.resolve("output/renders_views")
                    seg = cfg.segmentation
                    vs = int(cfg.reconstruction.render_view_stride)
                    stem = f"{int(provisional_cam) // int(vs):05d}"
                    rgb = render_dir / f"rgb_{stem}.png"
                    meta_p = render_dir / f"cam_meta_{stem}.npz"
                    depth_p = render_dir / f"depth_{stem}.npy"
                    if rgb.exists() and meta_p.exists() and depth_p.exists():
                        meta = np.load(meta_p)
                        K = np.asarray(meta["K"], dtype=np.float64)
                        w2c = np.asarray(meta["world_view_transform"], dtype=np.float64)
                        depth = np.load(depth_p).astype(np.float32)
                        mask_dbg = mode_b_out / "debug" / "masks"
                        mask_dbg.mkdir(parents=True, exist_ok=True)
                        mask_dbg_info: dict = {}
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
                            allow_cpu_fallback=bool(getattr(seg, "allow_cpu_fallback", False)),
                            debug_mask_dir=mask_dbg,
                            debug_stem=f"{rgb.stem}_ccmask",
                            debug_info=mask_dbg_info,
                        )
                        mask_view_cache[int(provisional_cam)] = (
                            mask.astype(bool),
                            depth,
                            K,
                            w2c,
                        )
                        view = mask_view_cache.get(int(provisional_cam))
                        console.print(
                            f"[mode-b] CC(mask_components): lazily computed mask for provisional_cam={provisional_cam} "
                            f"(rgb={rgb.name})"
                        )
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[yellow][mode-b] CC(mask_components): lazy mask compute failed: {exc}[/]")
            if view is None:
                idx_cc = largest_component_touching_seeds(
                    pos,
                    idx_shell,
                    idx_surface,
                    float(phys.mode_b_cc_link_radius_m),
                )
                console.print(
                    "[yellow]CC strategy=mask_components but no (mask,depth,K,w2c) cached yet; "
                    "falling back to single-component CC.[/]"
                )
            else:
                from physics.mode_b_selection import filter_components_by_mask_projection

                mask_c, depth_c, K_c, w2c_c = view
                idx_cc, cc_dbg = filter_components_by_mask_projection(
                    positions=pos,
                    candidate_indices=idx_shell,
                    mask=mask_c,
                    depth_map=depth_c,
                    K=K_c,
                    world_view_transform=w2c_c,
                    link_radius_m=float(phys.mode_b_cc_link_radius_m),
                    depth_tolerance_abs_m=float(seg.mask_depth_tolerance_abs_m),
                    depth_tolerance_rel=float(seg.mask_depth_tolerance_rel),
                    max_components=int(cc_keep_topk) if cc_keep_topk is not None else int(getattr(phys, "mode_b_cc_max_components", 4)),
                    score_min=0.0 if cc_keep_topk is not None else float(getattr(phys, "mode_b_cc_mask_score_min", 0.02)),
                    neighbor_radius_m=float(getattr(phys, "mode_b_selection_completion_neighbor_radius_m", 0.0))
                    if getattr(phys, "mode_b_selection_completion_enabled", False)
                    else None,
                    seed_indices=idx_surface,
                )
                (mode_b_out / "debug").mkdir(parents=True, exist_ok=True)
                (mode_b_out / "debug" / "cc_mask_components_debug.json").write_text(
                    json.dumps(cc_dbg, indent=2),
                    encoding="utf-8",
                )
                console.print(
                    f"CC(mask_components): {len(idx_shell)} → {len(idx_cc)} "
                    f"(wrote {mode_b_out / 'debug' / 'cc_mask_components_debug.json'})"
                )
        else:
            idx_cc = largest_component_touching_seeds(
                pos,
                idx_shell,
                idx_surface,
                float(phys.mode_b_cc_link_radius_m),
            )
            console.print(
                f"Connected-component (single, seeds, r={phys.mode_b_cc_link_radius_m:g} m): "
                f"{len(idx_shell)} → {len(idx_cc)} Gaussians"
            )

        if idx_cc.size == 0:
            console.print("[yellow]Connected-component filter removed all points; using shell set.[/]")
            idx_cc = idx_shell
        lo_cc, hi_cc = selection_xyz_bounds(pos, idx_cc)
        console.print(
            f"After CC: {len(idx_cc)} Gaussians  xyz min={lo_cc.tolist()} max={hi_cc.tolist()}"
        )

        idx = idx_cc
    opacity_floor = phys.mode_b_opacity_logit_min
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
    console.print(
        f"[mode-b debug] final_selection_count={len(idx)}  "
        f"surface_count={len(idx_surface) if idx_surface is not None else 'None'}  "
        f"out_dir={mode_b_out}"
    )

    dbg = mode_b_out / "debug"
    best_cam_idx = 0
    dbg_info: dict | None = None
    selection_debug_used_cuda = False

    # Always write compact selection stats (requested for rigorous tracing).
    def _sel_stats(stage: str, indices: np.ndarray) -> dict:
        ii = np.asarray(indices, dtype=np.int64).ravel()
        ii = ii[(ii >= 0) & (ii < pos.shape[0])]
        if ii.size == 0:
            return {"stage": stage, "count": 0}
        pts = pos[ii]
        y = pts[:, 1].astype(np.float64)
        qs = [0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0]
        qv = np.percentile(y, qs).tolist()
        hist_edges = np.linspace(float(np.min(y)), float(np.max(y)), num=11)
        hist_counts, _ = np.histogram(y, bins=hist_edges)
        lo, hi = pts.min(axis=0).astype(np.float64), pts.max(axis=0).astype(np.float64)
        cen = pts.mean(axis=0).astype(np.float64)
        return {
            "stage": stage,
            "count": int(ii.size),
            "bbox_min_xyz": lo.tolist(),
            "bbox_max_xyz": hi.tolist(),
            "centroid_xyz": cen.tolist(),
            "y_percentiles": {"q": qs, "v": qv},
            "y_hist": {"edges": hist_edges.tolist(), "counts": hist_counts.astype(int).tolist()},
        }

    sel_trace = {
        "surface": _sel_stats("surface", idx_surface if idx_surface is not None else np.asarray([], dtype=np.int64)),
        "bbox": _sel_stats("bbox", idx_bbox if "idx_bbox" in locals() else idx),
        "shell": _sel_stats("shell", idx_shell if "idx_shell" in locals() else idx),
        "cc": _sel_stats("cc", idx_cc if "idx_cc" in locals() else idx),
        "final": _sel_stats("final", idx),
    }
    (mode_b_out / "selection_trace.json").write_text(json.dumps(sel_trace, indent=2), encoding="utf-8")

    # For debug overlays we need to distinguish pre- vs post-completion sets.
    idx_pre_completion = np.asarray(idx, dtype=np.int64).copy()
    added_by_completion = np.asarray([], dtype=np.int64)
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
                    mn = int(phys.mode_b_min_visible_selected)
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
        console.print("[dim]After 3DGS selection debug: gc + empty_cache[/]")

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

    # ---- Selection completion pass (fix: visible parts dominated by unselected gaussians) ----
    # Uses chosen render_camera_index and the corresponding SAM mask view (if present).
    if bool(phys.mode_b_selection_completion_enabled):
        view = mask_view_cache.get(int(render_camera_index))
        if view is None:
            # If we reused cached surface indices, we may not have run SAM2 in this invocation.
            # Lazily run SAM2 for a few rendered views (multi-view), using the already-rendered views folder.
            try:
                render_dir = cfg.resolve("output/renders_views")
                seg = cfg.segmentation
                vs = int(cfg.reconstruction.render_view_stride)
                all_rgbs = sorted(render_dir.glob("rgb_*.png"))
                rgbs = all_rgbs[: int(seg.multi_view_count)]
                if not rgbs:
                    raise FileNotFoundError(f"no rgb_*.png under {render_dir}")
                mask_dbg = mode_b_out / "debug" / "masks"
                mask_dbg.mkdir(parents=True, exist_ok=True)
                for rgb in rgbs:
                    stem = rgb.stem.replace("rgb_", "")
                    meta = np.load(render_dir / f"cam_meta_{stem}.npz")
                    K = np.asarray(meta["K"], dtype=np.float64)
                    w2c = np.asarray(meta["world_view_transform"], dtype=np.float64)
                    mask_dbg_info: dict = {}
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
                        allow_cpu_fallback=bool(getattr(seg, "allow_cpu_fallback", False)),
                        debug_mask_dir=mask_dbg,
                        debug_stem=f"{rgb.stem}_completion",
                        debug_info=mask_dbg_info,
                    )
                    depth = np.load(render_dir / f"depth_{stem}.npy")
                    cam_train_idx = int(stem) * int(vs)
                    mask_view_cache[int(cam_train_idx)] = (
                        mask.astype(bool),
                        depth.astype(np.float32),
                        K,
                        w2c,
                    )
                view = mask_view_cache.get(int(render_camera_index))
                console.print(
                    f"[mode-b] selection completion: lazily computed SAM2 masks for {len(rgbs)} views; "
                    f"camera={render_camera_index} has_mask={view is not None}"
                )
            except Exception as exc:  # noqa: BLE001
                view = None
                console.print(
                    f"[yellow][mode-b] selection completion enabled but failed to compute mask lazily: {exc}[/]"
                )

        if view is None:
            console.print(
                f"[yellow][mode-b] selection completion enabled but no mask/depth view for camera {render_camera_index}; "
                "skipping.[/]"
            )
        else:
            from segmentation.selection_completion import complete_selection_by_mask_projection

            mask_c, depth_c, K_c, w2c_c = view
            idx_before = np.asarray(idx, dtype=np.int64)
            lo_b, hi_b = selection_xyz_bounds(pos, idx_before)
            idx2, added = complete_selection_by_mask_projection(
                positions=pos,
                selected_indices=idx_before,
                mask=mask_c,
                depth_map=depth_c,
                K=K_c,
                world_view_transform=w2c_c,
                selected_aabb_lo=lo_b,
                selected_aabb_hi=hi_b,
                aabb_expand_ratio=float(phys.mode_b_selection_completion_aabb_expand_ratio),
                mask_dilate_px=int(getattr(phys, "mode_b_selection_completion_mask_dilate_px", 0)),
                neighbor_radius_m=float(getattr(phys, "mode_b_selection_completion_neighbor_radius_m", 0.0)),
                depth_tolerance_abs_m=float(seg.mask_depth_tolerance_abs_m),
                depth_tolerance_rel=float(seg.mask_depth_tolerance_rel),
                max_add=int(phys.mode_b_selection_completion_max_add),
            )
            if added.size:
                idx = idx2
                added_by_completion = np.asarray(added, dtype=np.int64)
                np.save(sim_indices_path, idx)
                np.save(selected_indices_path, idx)
                console.print(
                    f"[mode-b] selection completion: original={len(idx_before)}  "
                    f"expanded={len(idx)}  added={len(added)}  "
                    f"camera={render_camera_index} aabb_expand={phys.mode_b_selection_completion_aabb_expand_ratio}"
                )
                if debug_selection:
                    dbg.mkdir(parents=True, exist_ok=True)
                    save_subset_points_ply(pos, added, dbg / "selection_completion_added_only.ply", rgb=(0.1, 0.95, 0.1))
                    save_subset_points_ply(pos, idx, dbg / "selection_completion_expanded_selected.ply", rgb=(1.0, 0.2, 0.05))
                    import json as _json

                    (dbg / "selection_completion_stats.json").write_text(
                        _json.dumps(
                            {
                                "camera_index": int(render_camera_index),
                                "original_selected_n": int(len(idx_before)),
                                "expanded_selected_n": int(len(idx)),
                                "added_n": int(len(added)),
                                "aabb_expand_ratio": float(phys.mode_b_selection_completion_aabb_expand_ratio),
                                "max_add": int(phys.mode_b_selection_completion_max_add),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            else:
                console.print("[mode-b] selection completion: no additional gaussians added.")

    # ---- Post-completion prune (remove background/window contamination) ----
    if bool(getattr(phys, "mode_b_selection_prune_enabled", False)):
        view = mask_view_cache.get(int(render_camera_index))
        if view is None:
            console.print("[yellow][mode-b] prune enabled but no cached mask/depth view; skipping prune.[/]")
        else:
            from segmentation.mask_to_gaussians import project_world_to_pixels

            mask_c, depth_c, K_c, w2c_c = view
            idx_before_prune = np.asarray(idx, dtype=np.int64)
            prune_diag: dict[str, int | float] = {
                "before_prune": int(len(idx_before_prune)),
                "rejected_by_neighbor_distance": 0,
                "rejected_by_mask": 0,
                "rejected_by_upper_v_gate": 0,
                "rejected_by_depth_seed_local": 0,
                "retained_final": 0,
            }
            # Neighbor gate around surface seeds (tightens to chair volume)
            r3 = float(getattr(phys, "mode_b_selection_prune_neighbor_radius_m", 0.0))
            idx_prune = idx_before_prune
            if r3 > 0.0 and idx_surface is not None and idx_surface.size:
                try:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(pos[np.asarray(idx_surface, dtype=np.int64)])
                    d, _ = tree.query(pos[idx_prune], k=1, workers=-1)
                    keep_nb = np.asarray(d <= r3)
                    prune_diag["rejected_by_neighbor_distance"] = int((~keep_nb).sum())
                    idx_prune = idx_prune[keep_nb]
                except Exception:
                    pass

            # Strict mask check (no dilation): retained candidates must land inside the chair mask.
            mask_gate = np.asarray(mask_c, dtype=bool)
            r = int(getattr(phys, "mode_b_selection_prune_mask_dilate_px", 0))

            depth_obj = np.asarray(depth_c, dtype=np.float64).copy()
            depth_obj[~np.asarray(mask_c, dtype=bool)] = np.inf
            depth_ref = depth_obj
            if r > 0:
                try:
                    from scipy.ndimage import minimum_filter

                    depth_ref = minimum_filter(depth_obj, size=(2 * r + 1, 2 * r + 1), mode="nearest")
                except Exception:
                    depth_ref = depth_obj

            Xw = pos[idx_prune]
            u, v, z = project_world_to_pixels(
                np.asarray(K_c, dtype=np.float64),
                np.asarray(w2c_c, dtype=np.float64),
                Xw,
            )
            H, Wm = mask_gate.shape
            valid = z > 1e-6
            ui = np.floor(u + 0.5).astype(np.int32)
            vi = np.floor(v + 0.5).astype(np.int32)
            inb = valid & (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < H)

            keep = np.zeros_like(inb, dtype=bool)
            if np.any(inb):
                # Strict mask check
                inside = np.zeros_like(inb, dtype=bool)
                inside[inb] = mask_gate[vi[inb], ui[inb]]
                prune_diag["rejected_by_mask"] = int(np.sum(inb & (~inside)))
                sel = np.where(inside)[0]

                # Seed-local depth gate:
                # build a per-pixel depth map from projected SURFACE SEEDS (min depth in a local window).
                seed_depth = np.full((H, Wm), np.inf, dtype=np.float64)
                seed_v_min = None
                try:
                    if idx_surface is not None and idx_surface.size:
                        Xs = pos[np.asarray(idx_surface, dtype=np.int64)]
                        us, vs, zs = project_world_to_pixels(
                            np.asarray(K_c, dtype=np.float64),
                            np.asarray(w2c_c, dtype=np.float64),
                            Xs,
                        )
                        val_s = zs > 1e-6
                        uis = np.floor(us[val_s] + 0.5).astype(np.int32)
                        vis = np.floor(vs[val_s] + 0.5).astype(np.int32)
                        zss = zs[val_s].astype(np.float64)
                        if vis.size:
                            seed_v_min = float(np.percentile(vis.astype(np.float64), 1.0))
                        for uu, vv, zz in zip(uis.tolist(), vis.tolist(), zss.tolist()):
                            if 0 <= uu < Wm and 0 <= vv < H:
                                if zz < seed_depth[vv, uu]:
                                    seed_depth[vv, uu] = zz
                        # local neighborhood min to allow "nearby seed depth"
                        from scipy.ndimage import minimum_filter

                        seed_depth = minimum_filter(seed_depth, size=7, mode="nearest")
                except Exception:
                    pass

                if sel.size:
                    abs_raw = getattr(phys, "mode_b_selection_prune_depth_abs_m", None)
                    rel_raw = getattr(phys, "mode_b_selection_prune_depth_rel", None)
                    abs_m = float(seg.mask_depth_tolerance_abs_m if abs_raw is None else abs_raw)
                    rel = float(seg.mask_depth_tolerance_rel if rel_raw is None else rel_raw)

                    # Reference depth: take the MIN of (mask-anchored rendered depth) and (seed-local depth).
                    zr_mask = depth_ref[vi[sel], ui[sel]].astype(np.float64)
                    zr_seed = seed_depth[vi[sel], ui[sel]].astype(np.float64)
                    zr = np.minimum(zr_mask, zr_seed)
                    # Upper-v gate: reject pixels far above the surface-seed vertical extent.
                    if seed_v_min is not None:
                        margin_px = 8.0
                        ok_v = vi[sel].astype(np.float64) >= (float(seed_v_min) - margin_px)
                        rejected_v = int(np.sum(~ok_v))
                        prune_diag["rejected_by_upper_v_gate"] = rejected_v
                        sel = sel[ok_v]
                        zr = zr[ok_v]
                    okz = np.isfinite(zr) & (zr > 0.0)
                    sel2 = sel[okz]
                    if sel2.size:
                        zr2 = zr[okz]
                        z2 = z[sel2]
                        tol = np.maximum(abs_m, rel * zr2)
                        # Reject behind-chair points relative to LOCAL seed depth
                        depth_ok = z2 <= (zr2 + tol)
                        keep_idx = sel2[depth_ok]
                        keep[keep_idx] = True
                        prune_diag["rejected_by_depth_seed_local"] = int(sel2.size - keep_idx.size)

            idx_after_prune = np.sort(np.unique(idx_prune[keep].astype(np.int64)))
            removed = np.setdiff1d(idx_before_prune, idx_after_prune, assume_unique=False).astype(np.int64)
            idx = idx_after_prune
            np.save(sim_indices_path, idx)
            np.save(selected_indices_path, idx)
            dbg.mkdir(parents=True, exist_ok=True)
            np.save(dbg / "pruned_removed_global_indices.npy", removed)
            prune_diag["retained_final"] = int(len(idx))
            (dbg / "prune_rejection_reasons.json").write_text(json.dumps(prune_diag, indent=2), encoding="utf-8")
            console.print(
                f"[mode-b] prune: before={len(idx_before_prune)} after={len(idx)} removed={len(removed)} "
                f"(neighbor_r={r3} dilate_px={r})"
            )
            # Cam0 overlays for prune results (exact sets).
            try:
                render_dir = cfg.resolve("output/renders_views")
                rgb0 = render_dir / "rgb_00000.png"
                meta0 = render_dir / "cam_meta_00000.npz"
                if rgb0.is_file() and meta0.is_file():
                    meta = np.load(meta0)
                    K0 = np.asarray(meta["K"], dtype=np.float64)
                    w2c0 = np.asarray(meta["world_view_transform"], dtype=np.float64)
                    save_gaussian_projection_debug_image(
                        rgb0,
                        dbg / "cam0_overlay_post_prune_final_set_orange.png",
                        pos,
                        np.asarray(idx, dtype=np.int64),
                        K0,
                        w2c0,
                        color_bgr=(0, 165, 255),  # orange
                        radius=2,
                    )
                    if removed.size:
                        save_gaussian_projection_debug_image(
                            rgb0,
                            dbg / "cam0_overlay_pruned_removed_magenta.png",
                            pos,
                            np.asarray(removed, dtype=np.int64),
                            K0,
                            w2c0,
                            color_bgr=(255, 0, 255),  # magenta
                            radius=2,
                        )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow][mode-b] prune cam0 overlays skipped: {exc}[/]")

    # ---- Cam0 projection overlays for actual sets used by PhysGaussian/MPM ----
    # Colors (BGR):
    # - pre-completion sim set: red
    # - added-by-completion: magenta
    # - post-completion full set: orange
    try:
        render_dir = cfg.resolve("output/renders_views")
        stem0 = "00000"
        rgb0 = render_dir / f"rgb_{stem0}.png"
        meta0 = render_dir / f"cam_meta_{stem0}.npz"
        if rgb0.is_file() and meta0.is_file():
            meta = np.load(meta0)
            K0 = np.asarray(meta["K"], dtype=np.float64)
            w2c0 = np.asarray(meta["world_view_transform"], dtype=np.float64)
            dbg_proj = mode_b_out / "debug"
            dbg_proj.mkdir(parents=True, exist_ok=True)

            save_gaussian_projection_debug_image(
                rgb0,
                dbg_proj / "cam0_overlay_pre_completion_sim_set_red.png",
                pos,
                np.asarray(idx_pre_completion, dtype=np.int64),
                K0,
                w2c0,
                color_bgr=(0, 0, 255),
                radius=2,
            )
            if added_by_completion.size:
                save_gaussian_projection_debug_image(
                    rgb0,
                    dbg_proj / "cam0_overlay_added_by_completion_magenta.png",
                    pos,
                    np.asarray(added_by_completion, dtype=np.int64),
                    K0,
                    w2c0,
                    color_bgr=(255, 0, 255),
                    radius=2,
                )
            save_gaussian_projection_debug_image(
                rgb0,
                dbg_proj / "cam0_overlay_post_completion_full_set_orange.png",
                pos,
                np.asarray(idx, dtype=np.int64),
                K0,
                w2c0,
                color_bgr=(0, 165, 255),  # orange
                radius=2,
            )
    except Exception as exc:  # noqa: BLE001 — diagnostics only
        console.print(f"[yellow][mode-b] cam0 completion-set overlays skipped: {exc}[/]")

    if bool(stop_after_selection):
        console.print(
            f"[mode-b] stop-after-selection: wrote [cyan]{mode_b_out / 'selection_trace.json'}[/] "
            f"(pre_completion={len(idx_pre_completion)} post_completion={len(idx)} added={len(added_by_completion)})"
        )
        return

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

    w_amp_kinematic = float(phys.mode_b_kinematic_wobble_amp)
    if wobble_amp is not None:
        w_amp_kinematic = float(wobble_amp)
        console.print(f"[mode-b] kinematic wobble amp override: [bold]{w_amp_kinematic}[/] m")

    console.print(
        "[dim]Static background:[/] PhysGaussian `unselected_*` tensors are never updated in the "
        "simulation loop; each frame concatenates [simulated subset | static rest] "
        "(see submodules/PhysGaussian/gs_simulation.py). With [bold]--kinematic-wobble[/], only "
        "selected indices receive xyz offsets; the rest match the idle copy."
    )

    stable_jelly_kinematic_override = preset in {"desk_jelly", "armchair_jelly"}
    if kinematic_wobble or stable_jelly_kinematic_override:
        fnum = phys.frame_num_test if smoke else phys.frame_num
        playback = phys.compile_video_playback_sec
        wobble_freq_kinematic = float(phys.mode_b_kinematic_wobble_freq_hz)
        wobble_height_kinematic = float(phys.mode_b_kinematic_wobble_height_gamma)
        wobble_bottom_pin_kinematic = float(phys.mode_b_kinematic_wobble_bottom_pin)
        if preset == "desk_jelly" and not kinematic_wobble:
            w_amp_kinematic = 0.065
            wobble_freq_kinematic = 1.35
            wobble_height_kinematic = 2.3
            wobble_bottom_pin_kinematic = 0.38
            console.print(
                "[mode-b kinematic] desk_jelly preset override: using stable in-place wobble "
                "and skipping PhysGaussian/MPM."
            )
            console.print(
                "[mode-b kinematic] desk_jelly wobble params: "
                f"amp={w_amp_kinematic}m freq={wobble_freq_kinematic}Hz "
                f"height_weight={wobble_height_kinematic} bottom_pin={wobble_bottom_pin_kinematic}"
            )
        elif preset == "armchair_jelly" and not kinematic_wobble:
            w_amp_kinematic = 0.14
            wobble_freq_kinematic = 1.35
            wobble_height_kinematic = 1.35
            wobble_bottom_pin_kinematic = 0.25
            console.print(
                "[preset=armchair_jelly] using isolated armchair_jelly kinematic wobble override",
                markup=False,
            )
            console.print(
                "[preset=armchair_jelly] selected_count="
                f"{len(idx)} run_simulation_called=False",
                markup=False,
            )
            console.print(
                "[mode-b kinematic] armchair_jelly wobble params: "
                f"amp={w_amp_kinematic}m freq={wobble_freq_kinematic}Hz "
                f"height_weight={wobble_height_kinematic} bottom_pin={wobble_bottom_pin_kinematic}"
            )
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
            frame_dt=float(phys.frame_dt),
            playback_seconds=float(playback if playback is not None else 10.0),
            camera_index=render_camera_index,
            wobble_amp=w_amp_kinematic,
            wobble_frequency=wobble_freq_kinematic,
            wobble_height_weight=wobble_height_kinematic,
            wobble_bottom_pin=wobble_bottom_pin_kinematic,
            track_debug=wobble_track_debug,
            indices_source=str(selected_indices_path.resolve()),
        )
        console.print(f"Video (kinematic): [green]{vid}[/]")
        return

    if len(idx) == 0:
        raise typer.BadParameter(
            "Empty selection (0 Gaussians). Segmentation likely failed or mask→3D projection returned nothing. "
            "In your log, Grounded-SAM-2 fell back due to missing files. "
            "Fix submodules/checkpoints, then rerun (use --no-reuse-selection to avoid stale caches)."
        )

    sim_cfg = mode_b_out / "phys_config.json"
    sim_n_grid = int(phys.n_grid_low if smoke else phys.n_grid)
    sim_frame_num = int(phys.mode_b_mpm_smoke_frames if smoke else phys.frame_num)
    # Sustained wobble drives MPM physics time → match requested motion horizon to video/sim length.
    if phys.mode_b_phys_sustained_wobble and not smoke:
        motion_s = phys.mode_b_phys_wobble_motion_seconds
        if motion_s is None:
            motion_s = float(phys.compile_video_playback_sec)
        tgt = int(math.ceil(float(motion_s) / float(phys.frame_dt)) + 2)
        sim_frame_num = max(sim_frame_num, tgt)
    gc.collect()
    _log_cuda_memory_line(console, "[mode-b] GPU before PhysGaussian")
    _maybe_cuda_empty()
    # Mode-b jelly: support plane from the *selection* (+Y-down → high-Y percentile = contact).
    # PhysGaussian pins COM each frame (see gs_simulation + phys JSON) so the desk
    # stays in place while MPM adds local deformation / shear wobble BCs.
    support_y = estimate_mode_b_support_contact_y(
        pos,
        idx,
        contact_percentile=float(phys.mode_b_support_contact_percentile),
    )
    sim_margin = (
        float(phys.mode_b_sim_area_margin)
        if phys.mode_b_sim_area_margin is not None
        else float(phys.sim_area_margin)
    )
    mat_b: dict[str, float] = {}
    if material_name == "jelly":
        if phys.mode_b_jelly_E is not None:
            mat_b["E"] = float(phys.mode_b_jelly_E)
        if phys.mode_b_jelly_grid_v_damping_scale is not None:
            mat_b["grid_v_damping_scale"] = float(phys.mode_b_jelly_grid_v_damping_scale)
    console.print(
        f"[mode-b] jelly in-place: support_contact_y={support_y:.4f} "
        f"(pctl={phys.mode_b_support_contact_percentile})  "
        f"selected bbox min={lo_i.tolist()} max={hi_i.tolist()}  "
        f"world_down=+scene_Y  mpm_world_up=(0,-1,0)  "
        f"pin_com={phys.mode_b_pin_initial_com}  "
        f"pin_vertical_only={phys.mode_b_pin_com_vertical_only}  "
        f"shear_wobble={phys.mode_b_mpm_in_place_shear_wobble}  "
        f"shear_full_vol={phys.mode_b_shear_wobble_full_volume}  "
        f"sust_phys={phys.mode_b_phys_sustained_wobble}  "
        f"sust_hz={phys.mode_b_phys_wobble_frequency_hz}  "
        f"vx_peak={phys.mode_b_phys_wobble_velocity_peak_x or phys.mode_b_phys_wobble_velocity_peak}  "
        f"vy_peak={phys.mode_b_phys_wobble_velocity_peak_y}  "
        f"vz_peak={phys.mode_b_phys_wobble_velocity_peak_z}  "
        f"d12=({phys.mode_b_phys_wobble_velocity_peak_diag1},"
        f"{phys.mode_b_phys_wobble_velocity_peak_diag2})  "
        f"twist_peak={phys.mode_b_phys_wobble_velocity_peak_twist}  "
        f"frames={sim_frame_num}  "
        f"sim_margin={sim_margin}  "
        f"E={mat_b.get('E', 'preset')}  "
        f"shear_v={phys.mode_b_shear_wobble_velocity}  "
        f"shear_t={phys.mode_b_shear_wobble_end_time}s  "
        f"disp_retention={phys.mode_b_mpm_displacement_retention}  "
        f"kabsch_strip={phys.mode_b_mpm_kabsch_rigid_strip}  "
        f"kabsch_amp={phys.mode_b_mpm_kabsch_elastic_amp}  "
        f"anchor_feet_y_pctl={phys.mode_b_mpm_anchor_feet_y_percentile}  "
        f"shear_symmetric_lr={phys.mode_b_mpm_shear_symmetric_lr_split}  "
        f"tilt_diag={phys.mode_b_mpm_tilt_diagnostics}  "
        f"freeze_cov_render={phys.mode_b_render_freeze_gaussian_cov}"
    )
    generate_phys_config(
        idx,
        pos,
        material_name,
        sim_cfg,
        template_path=default_template_dir() / "jelly.json",
        n_grid=sim_n_grid,
        frame_num=sim_frame_num,
        frame_dt=phys.frame_dt,
        substep_dt=phys.substep_dt,
        gravity=float(phys.gravity)
        * float(phys.gravity_scale)
        * float(phys.mode_b_jelly_gravity_mult if material_name == "jelly" else 1.0),
        camera_index=render_camera_index,
        simulate_indices_npy=sim_indices_path,
        subtract_rigid_drift=False,
        floor_y=support_y,
        world_up=(0.0, -1.0, 0.0),
        floor_collider=True,
        floor_friction=float(phys.floor_friction),
        in_place_wobble=bool(phys.mode_b_mpm_in_place_shear_wobble),
        wobble_velocity=float(phys.mode_b_shear_wobble_velocity),
        wobble_end_time=float(phys.mode_b_shear_wobble_end_time),
        shear_wobble_full_volume=bool(phys.mode_b_shear_wobble_full_volume),
        pin_initial_com_mpm=bool(phys.mode_b_pin_initial_com),
        pin_com_vertical_only=bool(phys.mode_b_pin_com_vertical_only),
        pin_zero_mean_velocity_gs=bool(phys.mode_b_pin_zero_mean_velocity_gs),
        mode_b_mpm_displacement_retention=(
            float(phys.mode_b_mpm_displacement_retention)
            if phys.mode_b_mpm_displacement_retention is not None
            else None
        ),
        mode_b_render_freeze_gaussian_cov=bool(phys.mode_b_render_freeze_gaussian_cov),
        mode_b_render_cov_override=str(getattr(phys, "mode_b_render_cov_override", "none")),
        mode_b_render_cov_override_scope=str(getattr(phys, "mode_b_render_cov_override_scope", "selected")),
        mode_b_render_tiny_splats_var=float(getattr(phys, "mode_b_render_tiny_splats_var", 1.0e-6)),
        mode_b_render_cov_diag_min=float(getattr(phys, "mode_b_render_cov_diag_min", 1.0e-6)),
        mode_b_render_cov_diag_max=float(getattr(phys, "mode_b_render_cov_diag_max", 5.0e-3)),
        mode_b_disp_propagation_enabled=bool(getattr(phys, "mode_b_disp_propagation_enabled", False)),
        mode_b_disp_propagation_k=int(getattr(phys, "mode_b_disp_propagation_k", 8)),
        mode_b_disp_low_percentile=float(getattr(phys, "mode_b_disp_low_percentile", 10.0)),
        mode_b_disp_min_neighbor_moved_m=float(getattr(phys, "mode_b_disp_min_neighbor_moved_m", 0.05)),
        mode_b_disp_propagation_alpha=float(getattr(phys, "mode_b_disp_propagation_alpha", 1.0)),
        mode_b_save_selected_means3d_per_frame=bool(getattr(phys, "mode_b_save_selected_means3d_per_frame", False)),
        mode_b_debug_dump_selected_only_renders=bool(getattr(phys, "mode_b_debug_dump_selected_only_renders", False)),
        mode_b_extra_render_dump_variants=str(getattr(phys, "mode_b_extra_render_dump_variants", "all")),
        mode_b_sand_render_override=bool(phys.mode_b_sand_render_override)
        if material_name == "sand"
        else False,
        mode_b_sand_opacity_scale=float(phys.mode_b_sand_opacity_scale),
        mode_b_sand_opacity_min=float(phys.mode_b_sand_opacity_min),
        mode_b_sand_opacity_max=float(phys.mode_b_sand_opacity_max),
        mode_b_sand_cov_scale=float(phys.mode_b_sand_cov_scale),
        mode_b_sand_color_override_rgb=phys.mode_b_sand_color_override_rgb,
        mode_b_sand_motion_correction=bool(phys.mode_b_sand_motion_correction)
        if material_name == "sand"
        else False,
        mode_b_sand_motion_alpha=float(phys.mode_b_sand_motion_alpha),
        mode_b_sand_global_dy_percentile=float(phys.mode_b_sand_global_dy_percentile),
        mode_b_sand_min_dy_as_global_frac=(
            float(phys.mode_b_sand_min_dy_as_global_frac)
            if phys.mode_b_sand_min_dy_as_global_frac is not None
            else None
        ),
        mode_b_sand_extreme_all_selected_down=bool(phys.mode_b_sand_extreme_all_selected_down)
        if material_name == "sand"
        else False,
        mode_b_sand_extreme_total_down_m=float(phys.mode_b_sand_extreme_total_down_m),
        mode_b_render_gaussian_subset=str(phys.mode_b_render_gaussian_subset),
        mode_b_mpm_kabsch_rigid_strip=bool(phys.mode_b_mpm_kabsch_rigid_strip),
        mode_b_mpm_kabsch_elastic_amp=float(phys.mode_b_mpm_kabsch_elastic_amp),
        mode_b_mpm_anchor_feet_y_percentile=(
            float(phys.mode_b_mpm_anchor_feet_y_percentile)
            if phys.mode_b_mpm_anchor_feet_y_percentile is not None
            else None
        ),
        shear_symmetric_lr_split=bool(phys.mode_b_mpm_shear_symmetric_lr_split),
        phys_sustained_wobble=bool(phys.mode_b_phys_sustained_wobble),
        phys_wobble_frequency_hz=float(phys.mode_b_phys_wobble_frequency_hz),
        phys_wobble_velocity_peak=float(phys.mode_b_phys_wobble_velocity_peak),
        phys_wobble_velocity_peak_x=(
            float(phys.mode_b_phys_wobble_velocity_peak_x)
            if phys.mode_b_phys_wobble_velocity_peak_x is not None
            else None
        ),
        phys_wobble_velocity_peak_y=float(phys.mode_b_phys_wobble_velocity_peak_y),
        phys_wobble_phase_y_rad=float(phys.mode_b_phys_wobble_phase_y),
        phys_wobble_band_amp_x=phys.mode_b_phys_wobble_band_amp_x,
        phys_wobble_band_amp_y=phys.mode_b_phys_wobble_band_amp_y,
        phys_wobble_band_vy_polarity=phys.mode_b_phys_wobble_band_vy_polarity,
        phys_wobble_band_phase_y_offset_rad=phys.mode_b_phys_wobble_band_phase_y_offset_rad,
        phys_wobble_velocity_peak_z=float(phys.mode_b_phys_wobble_velocity_peak_z),
        phys_wobble_velocity_peak_diag1=float(phys.mode_b_phys_wobble_velocity_peak_diag1),
        phys_wobble_velocity_peak_diag2=float(phys.mode_b_phys_wobble_velocity_peak_diag2),
        phys_wobble_velocity_peak_twist=float(phys.mode_b_phys_wobble_velocity_peak_twist),
        phys_wobble_phase_z_rad=float(phys.mode_b_phys_wobble_phase_z),
        phys_wobble_phase_diag1_rad=float(phys.mode_b_phys_wobble_phase_diag1_rad),
        phys_wobble_phase_diag2_rad=float(phys.mode_b_phys_wobble_phase_diag2_rad),
        phys_wobble_phase_twist_rad=float(phys.mode_b_phys_wobble_phase_twist_rad),
        phys_wobble_bundle_quad_phase_rad=phys.mode_b_phys_wobble_bundle_quad_phase_rad,
        phys_wobble_bundle_band_phase_rad=phys.mode_b_phys_wobble_bundle_band_phase_rad,
        phys_wobble_band_amp_z=phys.mode_b_phys_wobble_band_amp_z,
        phys_wobble_band_amp_diag1=phys.mode_b_phys_wobble_band_amp_diag1,
        phys_wobble_band_amp_diag2=phys.mode_b_phys_wobble_band_amp_diag2,
        phys_wobble_band_amp_twist=phys.mode_b_phys_wobble_band_amp_twist,
        phys_wobble_force_scale=float(phys.mode_b_phys_wobble_force_scale),
        phys_wobble_decay_lambda_per_s=float(phys.mode_b_phys_wobble_decay_lambda_per_s),
        phys_wobble_duration_s=(
            float(phys.mode_b_phys_wobble_duration_s)
            if phys.mode_b_phys_wobble_duration_s is not None
            else None
        ),
        phys_wobble_ramp_time_s=float(phys.mode_b_phys_wobble_ramp_time_s),
        mode_b_mpm_tilt_diagnostics=bool(phys.mode_b_mpm_tilt_diagnostics),
        mode_b_mpm_tilt_top_y_percentile=float(phys.mode_b_mpm_tilt_top_y_percentile),
        material_overrides=mat_b if mat_b else None,
        enable_internal_particle_fill=phys.enable_mpm_particle_filling,
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
    if preset == "desk_jelly":
        # Preserve the original desk_jelly PhysGaussian bbox/sim_area selection
        # while allowing only its output folder to be renamed.
        sj = json.loads(sim_cfg.read_text(encoding="utf-8"))
        sj["disable_exact_simulate_indices_for_preset"] = "desk_jelly"
        sim_cfg.write_text(json.dumps(sj, indent=2), encoding="utf-8")
    vid = run_simulation(
        str(ply_path),
        str(sim_cfg),
        str(mode_b_out),
        camera_scene_path=str(model_out),
        playback_seconds=phys.compile_video_playback_sec,
        taichi_device_memory_gb=_taichi_gb,
    )
    # Post-sim diagnostics: project moved/unmoved selected Gaussians onto cam0.
    # moved_selected_local_indices.npy contains indices local to the *post-completion* selected set `idx`.
    if not bool(getattr(phys, "mode_b_minimal_output", False)):
        try:
            dbg_proj = mode_b_out / "debug"
            moved_local_p = dbg_proj / "moved_selected_local_indices.npy"
            means0_p = dbg_proj / "selected_means3d_first.npy"
            meansT_p = dbg_proj / "selected_means3d_last.npy"
            if moved_local_p.is_file():
                moved_local = np.load(moved_local_p).astype(np.int64).ravel()
                sel_global = np.asarray(idx, dtype=np.int64).ravel()
                moved_local = moved_local[(moved_local >= 0) & (moved_local < sel_global.shape[0])]
                moved_global = np.unique(sel_global[moved_local]).astype(np.int64)
                moved_mask = np.zeros(sel_global.shape[0], dtype=bool)
                moved_mask[moved_local] = True
                unmoved_global = np.unique(sel_global[~moved_mask]).astype(np.int64)

                render_dir = cfg.resolve("output/renders_views")
                stem0 = "00000"
                rgb0 = render_dir / f"rgb_{stem0}.png"
                meta0 = render_dir / f"cam_meta_{stem0}.npz"
                if rgb0.is_file() and meta0.is_file():
                    meta = np.load(meta0)
                    K0 = np.asarray(meta["K"], dtype=np.float64)
                    w2c0 = np.asarray(meta["world_view_transform"], dtype=np.float64)
                    save_gaussian_projection_debug_image(
                        rgb0,
                        dbg_proj / "cam0_overlay_moved_selected_green.png",
                        pos,
                        moved_global,
                        K0,
                        w2c0,
                        color_bgr=(0, 255, 0),
                        radius=2,
                    )
                    save_gaussian_projection_debug_image(
                        rgb0,
                        dbg_proj / "cam0_overlay_unmoved_selected_blue.png",
                        pos,
                        unmoved_global,
                        K0,
                        w2c0,
                        color_bgr=(255, 0, 0),
                        radius=2,
                    )

                    # Displacement magnitude heatmap overlays (selected set only).
                    if means0_p.is_file() and meansT_p.is_file():
                        import cv2
                        from segmentation.mask_to_gaussians import project_world_to_pixels

                        X0 = np.load(means0_p).astype(np.float64)  # (Nsel,3)
                        XT = np.load(meansT_p).astype(np.float64)  # (Nsel,3)
                        nsel = int(min(sel_global.shape[0], X0.shape[0], XT.shape[0]))
                        X0 = X0[:nsel]
                        XT = XT[:nsel]
                        disp = np.linalg.norm(XT - X0, axis=1)  # meters

                        qs = [0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0]
                        qv = np.percentile(disp, qs).tolist() if disp.size else []
                        hist_edges = np.linspace(
                            float(np.min(disp)) if disp.size else 0.0,
                            float(np.max(disp)) if disp.size else 1.0,
                            num=31,
                        )
                        hist_counts, _ = (
                            np.histogram(disp, bins=hist_edges)
                            if disp.size
                            else (np.zeros(30, dtype=int), hist_edges)
                        )
                        (dbg_proj / "displacement_magnitude_summary.json").write_text(
                            json.dumps(
                                {
                                    "selected_n": int(nsel),
                                    "moved_n_threshold_1e-4": int(moved_local.size),
                                    "unmoved_n_threshold_1e-4": int(nsel - moved_local.size),
                                    "threshold_m": 1e-4,
                                    "disp_percentiles_m": {"q": qs, "v": qv},
                                    "disp_hist_m": {
                                        "edges": hist_edges.tolist(),
                                        "counts": hist_counts.astype(int).tolist(),
                                    },
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )

                        def _draw_heatmap(base_bgr, proj_xyz: np.ndarray, out_name: str) -> None:
                            if base_bgr is None or disp.size == 0:
                                return
                            H, W = base_bgr.shape[:2]
                            u, v, z = project_world_to_pixels(K0, w2c0, proj_xyz)
                            valid = z > 1e-6
                            ui = np.floor(u[valid] + 0.5).astype(np.int32)
                            vi = np.floor(v[valid] + 0.5).astype(np.int32)
                            disp_v = disp[valid]
                            if disp_v.size == 0:
                                return
                            cap = float(np.percentile(disp_v, 99.0))
                            cap = max(cap, 1e-6)
                            t = np.clip(disp_v / cap, 0.0, 1.0)
                            vals = (t * 255.0).astype(np.uint8)
                            colors = cv2.applyColorMap(vals.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
                            for (uu, vv, col) in zip(ui.tolist(), vi.tolist(), colors.tolist()):
                                if 0 <= uu < W and 0 <= vv < H:
                                    cv2.circle(base_bgr, (uu, vv), 2, tuple(int(c) for c in col), thickness=-1)
                            cv2.imwrite(str(dbg_proj / out_name), base_bgr)

                        base_final = cv2.imread(str(rgb0))
                        if base_final is not None:
                            _draw_heatmap(
                                base_final,
                                XT,
                                "cam0_overlay_displacement_magnitude_heatmap_projected_at_final_turbo.png",
                            )
                            base_init = cv2.imread(str(rgb0))
                            _draw_heatmap(
                                base_init,
                                X0,
                                "cam0_overlay_displacement_magnitude_on_initial_positions_turbo.png",
                            )
        except Exception as exc:  # noqa: BLE001 — diagnostics only
            console.print(f"[yellow][mode-b] moved/unmoved cam0 overlays skipped: {exc}[/]")
    dbg_js = mode_b_out / "frames" / "mode_b_mpm_debug.json"
    if dbg_js.is_file() and not bool(getattr(phys, "mode_b_minimal_output", False)):
        console.print(f"[mode-b] MPM COM / drift log: [cyan]{dbg_js}[/]")
    tilt_js = mode_b_out / "frames" / "mode_b_mpm_tilt_series.json"
    if tilt_js.is_file() and not bool(getattr(phys, "mode_b_minimal_output", False)):
        console.print(f"[mode-b] MPM tilt time-series + drift summary: [cyan]{tilt_js}[/]")
    console.print(f"Video: [green]{vid}[/]")

    # Background compositing (fills black holes behind collapsed chair).
    try:
        if bool(getattr(phys, "mode_b_background_composite_enabled", False)):
            from rendering.background_composite import (
                compile_video_ffmpeg,
                composite_frames_over_plate,
                composite_frames_over_plate_objectmask,
                composite_frames_over_plate_objectmask_hard,
                composite_selected_only_matte_over_plate,
                fullframe_plate_cleanup_encode_mp4,
                make_inpainted_plate,
            )

            dbg = mode_b_out / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            render_dir = cfg.resolve("output/renders_views")
            cam0_rgb = render_dir / "rgb_00000.png"
            # Prefer cached SAM mask from CC/completion stage if present.
            default_mask = dbg / "masks" / "rgb_00000_ccmask_mask_binary.png"
            mask_path = Path(getattr(phys, "mode_b_background_mask_path", "")).expanduser() if getattr(phys, "mode_b_background_mask_path", None) else default_mask
            if not mask_path.is_file():
                # Fallback: build a mask by projecting selected centers at t=0.
                # This makes bg compositing usable even when debug_selection is disabled.
                try:
                    import numpy as _np
                    import cv2

                    meta = _np.load(cfg.resolve("output/renders_views") / "cam_meta_00000.npz")
                    K = _np.asarray(meta["K"], dtype=_np.float64)
                    W = _np.asarray(meta["world_view_transform"], dtype=_np.float64)
                    sel0 = dbg / "selected_means3d_first.npy"
                    if sel0.is_file():
                        Xi = _np.load(sel0).astype(_np.float64)
                        img = cv2.imread(str(cam0_rgb))
                        if img is None:
                            raise FileNotFoundError(cam0_rgb)
                        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        from segmentation.mask_to_gaussians import project_world_to_pixels

                        u, v, z = project_world_to_pixels(K, W, Xi)
                        valid = z > 1e-6
                        ui = _np.floor(u[valid] + 0.5).astype(_np.int32)
                        vi = _np.floor(v[valid] + 0.5).astype(_np.int32)
                        pr = int(getattr(phys, "mode_b_background_objectmask_point_radius_px", 2))
                        for uu, vv in zip(ui.tolist(), vi.tolist()):
                            if 0 <= uu < mask.shape[1] and 0 <= vv < mask.shape[0]:
                                cv2.circle(mask, (uu, vv), max(1, pr), 255, thickness=-1)
                        dpx = int(getattr(phys, "mode_b_background_objectmask_dilate_px", 6))
                        if dpx > 0:
                            yy, xx = _np.ogrid[-dpx : dpx + 1, -dpx : dpx + 1]
                            kernel = ((xx * xx + yy * yy) <= (dpx * dpx)).astype(_np.uint8)
                            mask = cv2.dilate(mask, kernel, iterations=1)
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        mask_path = dbg / "masks" / "rgb_00000_auto_mask_from_selected_centers.png"
                        cv2.imwrite(str(mask_path), mask)
                except Exception:
                    pass
            plate = make_inpainted_plate(
                cam0_rgb_path=cam0_rgb,
                chair_mask_path=mask_path,
                out_plate_path=dbg / "cam0_inpainted_background_plate.png",
                inpaint_radius_px=int(getattr(phys, "mode_b_background_inpaint_radius_px", 5)),
            )
            frames_dir = mode_b_out / "frames"
            fps = int(round(1.0 / float(phys.frame_dt)))
            use_objmask = bool(getattr(phys, "mode_b_background_objectmask_composite", True))
            if bool(getattr(phys, "mode_b_fullframe_plate_cleanup", False)):
                out_mp4 = mode_b_out / "video" / "final_cleanup_fullframe.mp4"
                n = fullframe_plate_cleanup_encode_mp4(
                    frames_dir=frames_dir,
                    plate_path=plate,
                    original_chair_mask_path=mask_path,
                    out_mp4=out_mp4,
                    fps=fps,
                    allowed_region_dilate_px=int(
                        getattr(phys, "mode_b_background_allowed_region_dilate_px", 8)
                    ),
                    allowed_region_down_extend_px=int(
                        getattr(phys, "mode_b_background_allowed_region_down_extend_px", 220)
                    ),
                    allowed_region_up_exclude_px=int(
                        getattr(phys, "mode_b_background_allowed_region_up_exclude_px", 40)
                    ),
                    dark_thresh_u8=int(getattr(phys, "mode_b_fullframe_cleanup_dark_thresh_u8", 42)),
                    protect_erode_px=int(getattr(phys, "mode_b_fullframe_cleanup_protect_erode_px", 5)),
                    near_chair_dilate_px=int(
                        getattr(phys, "mode_b_fullframe_cleanup_near_chair_dilate_px", 64)
                    ),
                    feather_px=int(getattr(phys, "mode_b_fullframe_cleanup_feather_px", 5)),
                )
                console.print(
                    f"[mode-b] full-frame plate cleanup → [green]{out_mp4}[/] ({n} frames)"
                )
            elif bool(getattr(phys, "mode_b_background_selected_only_matte", False)):
                fg_dir = dbg / "extra_renders" / "selected_only_normal"
                mask_dir = dbg / "extra_renders" / "selected_only_normal_mask"
                if not fg_dir.is_dir() or not any(fg_dir.glob("*.png")):
                    raise FileNotFoundError(
                        f"missing {fg_dir} PNGs; enable physics.mode_b_debug_dump_selected_only_renders "
                        "and re-run PhysGaussian (selected-only normal pass)."
                    )
                if not mask_dir.is_dir() or not any(mask_dir.glob("*.png")):
                    raise FileNotFoundError(
                        f"missing {mask_dir} PNGs; enable physics.mode_b_debug_dump_selected_only_renders "
                        "and re-run PhysGaussian."
                    )
                out_frames = mode_b_out / "frames_final_selected_only_plate"
                n = composite_selected_only_matte_over_plate(
                    plate_path=plate,
                    chair_rgb_dir=fg_dir,
                    raw_mask_dir=mask_dir,
                    out_composite_dir=out_frames,
                    out_alpha_dir=mode_b_out / "frames_alpha_matte",
                    out_foreground_dir=mode_b_out / "frames_foreground_selected_only",
                    original_chair_mask_path=mask_path,
                    allowed_region_dilate_px=int(getattr(phys, "mode_b_background_allowed_region_dilate_px", 8)),
                    allowed_region_down_extend_px=int(
                        getattr(phys, "mode_b_background_allowed_region_down_extend_px", 220)
                    ),
                    allowed_region_up_exclude_px=int(
                        getattr(phys, "mode_b_background_allowed_region_up_exclude_px", 40)
                    ),
                    alpha_close_kernel_px=int(getattr(phys, "mode_b_background_alpha_close_kernel_px", 25)),
                    alpha_feather_blur_px=int(getattr(phys, "mode_b_background_alpha_feather_blur_px", 7)),
                )
                out_mp4 = mode_b_out / "video" / "final_chair_only_collapse.mp4"
            elif bool(getattr(phys, "mode_b_background_hard_chair_only", False)):
                # Hard chair-only compositing: only allow simulated pixels inside strict chair mask.
                means_pf = dbg / "selected_means3d_per_frame.npy"
                if not means_pf.is_file():
                    raise FileNotFoundError(
                        f"missing {means_pf} (enable physics.mode_b_save_selected_means3d_per_frame)"
                    )
                out_frames = mode_b_out / "frames_final_chair_only"
                n = composite_frames_over_plate_objectmask_hard(
                    frames_dir=frames_dir,
                    plate_path=plate,
                    out_frames_dir=out_frames,
                    cam_meta_npz=cfg.resolve("output/renders_views") / "cam_meta_00000.npz",
                    selected_means_per_frame_npy=means_pf,
                    original_chair_mask_path=mask_path,
                    point_radius_px=int(getattr(phys, "mode_b_background_objectmask_point_radius_px", 2)),
                    dilate_px=int(getattr(phys, "mode_b_background_objectmask_dilate_px", 6)),
                    blur_px=int(getattr(phys, "mode_b_background_objectmask_blur_px", 11)),
                    allowed_region_dilate_px=int(getattr(phys, "mode_b_background_allowed_region_dilate_px", 8)),
                    allowed_region_down_extend_px=int(getattr(phys, "mode_b_background_allowed_region_down_extend_px", 220)),
                    allowed_region_up_exclude_px=int(getattr(phys, "mode_b_background_allowed_region_up_exclude_px", 40)),
                )
                out_mp4 = mode_b_out / "video" / "final_chair_only_collapse.mp4"
            elif use_objmask:
                means_pf = dbg / "selected_means3d_per_frame.npy"
                if not means_pf.is_file():
                    raise FileNotFoundError(
                        f"missing {means_pf} (enable physics.mode_b_save_selected_means3d_per_frame)"
                    )
                out_frames = mode_b_out / "frames_composited_objmask"
                n = composite_frames_over_plate_objectmask(
                    frames_dir=frames_dir,
                    plate_path=plate,
                    out_frames_dir=out_frames,
                    cam_meta_npz=cfg.resolve("output/renders_views") / "cam_meta_00000.npz",
                    selected_means_per_frame_npy=means_pf,
                    point_radius_px=int(getattr(phys, "mode_b_background_objectmask_point_radius_px", 2)),
                    dilate_px=int(getattr(phys, "mode_b_background_objectmask_dilate_px", 6)),
                    blur_px=int(getattr(phys, "mode_b_background_objectmask_blur_px", 11)),
                    out_masks_dir=mode_b_out / "masks_objmask",
                )
                out_mp4 = mode_b_out / "video" / "output_bg_objmask_composited.mp4"
            else:
                out_frames = mode_b_out / "frames_composited_bg"
                n = composite_frames_over_plate(
                    frames_dir=frames_dir,
                    plate_path=plate,
                    out_frames_dir=out_frames,
                    diff_threshold_u8=int(getattr(phys, "mode_b_background_diff_threshold_u8", 18)),
                    dilate_px=int(getattr(phys, "mode_b_background_diff_dilate_px", 3)),
                )
                out_mp4 = mode_b_out / "video" / "output_bg_composited.mp4"
            if not bool(getattr(phys, "mode_b_fullframe_plate_cleanup", False)):
                compile_video_ffmpeg(frames_dir=out_frames, out_mp4=out_mp4, fps=fps)
                console.print(f"[mode-b] bg composite wrote {n} frames + video [green]{out_mp4}[/]")

            # Compile selected-only render dumps if present (skip matte / fullframe cleanup paths).
            try:
                if not bool(getattr(phys, "mode_b_background_selected_only_matte", False)) and not bool(
                    getattr(phys, "mode_b_fullframe_plate_cleanup", False)
                ):
                    extra_root = dbg / "extra_renders"
                    if extra_root.is_dir():
                        for kind in ("selected_only_normal", "selected_only_clamp_diag", "selected_only_tiny_splats"):
                            fr_dir = extra_root / kind
                            if fr_dir.is_dir():
                                mp4 = mode_b_out / "video" / f"{kind}.mp4"
                                compile_video_ffmpeg(frames_dir=fr_dir, out_mp4=mp4, fps=fps)
                        for kind in (
                            "selected_only_normal_mask",
                            "selected_only_clamp_diag_mask",
                            "selected_only_tiny_splats_mask",
                        ):
                            fr_dir = extra_root / kind
                            if fr_dir.is_dir():
                                mp4 = mode_b_out / "video" / f"{kind}.mp4"
                                compile_video_ffmpeg(frames_dir=fr_dir, out_mp4=mp4, fps=fps)
                        means_pf = dbg / "selected_means3d_per_frame.npy"
                        if means_pf.is_file():
                            for kind in ("selected_only_normal", "selected_only_clamp_diag", "selected_only_tiny_splats"):
                                fr_dir = extra_root / kind
                                if fr_dir.is_dir():
                                    out_fr = mode_b_out / f"frames_composited_{kind}"
                                    n2 = composite_frames_over_plate_objectmask(
                                        frames_dir=fr_dir,
                                        plate_path=plate,
                                        out_frames_dir=out_fr,
                                        cam_meta_npz=cfg.resolve("output/renders_views") / "cam_meta_00000.npz",
                                        selected_means_per_frame_npy=means_pf,
                                        point_radius_px=int(
                                            getattr(phys, "mode_b_background_objectmask_point_radius_px", 2)
                                        ),
                                        dilate_px=int(getattr(phys, "mode_b_background_objectmask_dilate_px", 6)),
                                        blur_px=int(getattr(phys, "mode_b_background_objectmask_blur_px", 11)),
                                        out_masks_dir=None,
                                    )
                                    mp4 = mode_b_out / "video" / f"{kind}_bg_objmask_composited.mp4"
                                    compile_video_ffmpeg(frames_dir=out_fr, out_mp4=mp4, fps=fps)
                                    console.print(f"[mode-b] composed {kind}: {n2} frames -> {mp4}")
            except Exception as exc2:  # noqa: BLE001
                console.print(f"[yellow][mode-b] extra render dump compile skipped: {exc2}[/]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow][mode-b] bg composite skipped: {exc}[/]")

    if bool(getattr(phys, "mode_b_minimal_output", False)):
        try:
            _mode_b_cleanup_minimal_artifacts(mode_b_out)
            console.print(
                f"[mode-b] minimal output: removed intermediates under [cyan]{mode_b_out}[/] "
                "(kept phys_config.json, indices npy, video/output.mp4, final_chair_only_collapse.mp4, "
                "final_cleanup_fullframe.mp4, frames_foreground_selected_only, frames_alpha_matte, "
                "frames_final_selected_only_plate if present)."
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow][mode-b] minimal cleanup skipped: {exc}[/]")


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

    sim_output_root = cfg.resolve(cfg.paths.sim_output)
    sim_run = get_sim_run_dir(sim_output_root, "mode_a", material)
    sim_run.mkdir(parents=True, exist_ok=True)
    console.print(f"[mode-a] sim_run (isolated output): [cyan]{sim_run}[/]")

    if no_physics:
        console.print(f"Merged: [cyan]{merged}[/]  (--no-physics: skipped simulation)")
        if mesh_render and mesh_verts_w is not None and mesh_faces_np is not None:
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

    phys_input_dir = merged.parent / "mode_a_phys_input"
    phys_input_ply = phys_input_dir / "point_cloud" / "iteration_0" / "point_cloud.ply"
    phys_input_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(merged, phys_input_ply)
    cameras_json = model_out / "cameras.json"
    if not cameras_json.is_file():
        raise FileNotFoundError(cameras_json)
    shutil.copy2(cameras_json, phys_input_dir / "cameras.json")
    console.print(
        f"[mode-a] PhysGaussian input checkpoint: [cyan]{phys_input_dir.resolve()}[/] "
        f"(staged from {merged.resolve()})"
    )

    # Simulate ONLY the generated object Gaussians (indices [n_base, n_base+n_obj)).
    # Simulating the full merged PLY applies physics to the background scene, which
    # destroys the render.
    sim_cfg = sim_run / "phys_mode_a.json"
    sim_idx_path = sim_run / "mode_a_obj_indices.npy"
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
        str(phys_input_dir),
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
