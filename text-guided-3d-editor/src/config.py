"""Load PRD-style pipeline_config.yaml into typed settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_sim_run_dir(sim_output_root: Path, mode: str, physics_type: str) -> Path:
    """Per-run directory under sim_output, e.g. mode_a_jelly, mode_b_sand.

    Keeps configs, indices, caches, and renders isolated per mode × material.
    """
    m = mode.strip()
    tag = physics_type.strip()
    if not m or not tag:
        raise ValueError("mode and physics_type must be non-empty")
    if any(sep in m for sep in ("/", "\\")) or ".." in m:
        raise ValueError(f"invalid mode for path segment: {mode!r}")
    if any(sep in tag for sep in ("/", "\\")) or ".." in tag:
        raise ValueError(f"invalid physics_type for path segment: {physics_type!r}")
    return sim_output_root / f"{m}_{tag}"


class SceneConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset_url: str = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    data_root: str = "data/mipnerf360"
    scene_name: str = "room"
    # Backward-compatible single output path. If `model_output_full` / `model_output_smoke` are set,
    # the pipeline will select between them based on smoke flags.
    model_output: str = "output/base_scene"
    model_output_full: str | None = None
    model_output_smoke: str | None = None
    training_iterations: int = 30000
    training_iterations_low: int = 8000


class ReconstructionConfig(BaseModel):
    render_width: int = 640
    render_height: int = 480
    render_view_stride: int = 10


class SegmentationConfig(BaseModel):
    grounding_dino_checkpoint: str = ""
    grounding_dino_config: str = ""
    sam2_checkpoint: str = ""
    sam2_config: str = ""
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    multi_view_count: int = 4
    consensus_min_votes: int = 2
    inference_width: int = 640
    inference_height: int = 480
    segmentation_device: str = "cuda"
    # GPU-only by default: if True, allow slow CPU retry / debug fallback mask.
    allow_cpu_fallback: bool = False
    # mask → 3DGS (mode-b / multi-view)
    mask_stride: int = 2
    mask_3d_distance_threshold_m: float = 0.07
    mask_use_depth_consistency: bool = True
    mask_depth_tolerance_abs_m: float = 0.035
    mask_depth_tolerance_rel: float = 0.03
    mask_knn: int = 12
    mask_pixel_tolerance_px: float = 5.0
    mask_min_votes: int = 2
    mask_erode_iters: int = 0
    mode_b_expand_bbox_margin_ratio: float = 0.06
    mode_b_expand_bbox_z_margin_ratio: float = 0.10


class GenerationConfig(BaseModel):
    stable_diffusion_model: str = "stabilityai/stable-diffusion-2-1-base"
    sds_steps: int = 500
    sds_steps_low: int = 200
    guidance_scale: float = 100.0
    object_scale_factor: float = 0.1


class PlacementConfig(BaseModel):
    surface_search_radius: float = 0.1
    opacity_threshold: float = 0.5


class PhysicsConfig(BaseModel):
    n_grid: int = 100
    n_grid_low: int = 40
    substep_dt: float = 2.0e-4
    frame_dt: float = 0.01
    frame_num: int = 200
    frame_num_test: int = 10
    # PhysGaussian subprocess: Taichi `device_memory_GB` (T4: try 4–5).
    taichi_device_memory_gb: float = 4.0
    # mode-b MPM only: frame count when CLI `--smoke` (not used for kinematic wobble).
    mode_b_mpm_smoke_frames: int = 48
    gravity: float = -9.8
    gravity_scale: float = 0.62
    floor_friction: float = 0.40
    sim_area_margin: float = 0.38
    wobble_velocity: float = 0.058
    wobble_end_time: float = 0.075
    placement_floor_contact_y_eps: float = 0.0
    enable_mpm_particle_filling: bool = False
    compile_video_playback_sec: float | None = None
    # mode-b: bbox expansion often swallows walls / neighbours; keep Gaussians near surface seeds.
    mode_b_shell_radius_m: float = 0.18
    # Optional PLY opacity logit floor (3DGS PLY ``opacity`` column); None disables.
    mode_b_opacity_logit_min: float | None = None
    mode_b_kinematic_wobble_amp: float = 0.008
    mode_b_kinematic_wobble_freq_hz: float = 1.0
    mode_b_kinematic_wobble_height_gamma: float = 1.5
    mode_b_kinematic_wobble_bottom_pin: float = 0.30
    mode_b_cc_link_radius_m: float = 0.055
    # mode-b: candidate Gaussians can form multiple disconnected components (e.g., seat vs backrest).
    # Default keeps a single component touching surface seeds. Optionally, keep multiple components
    # that project into the target mask (more robust for fragmented chair selections).
    # Options: "single" (legacy), "mask_components" (keep components with mask overlap score).
    mode_b_cc_keep_strategy: str = "single"
    # Minimum fraction of a component's projected points that pass mask + depth gate.
    mode_b_cc_mask_score_min: float = 0.02
    # Hard cap: keep up to this many components (sorted by score desc).
    mode_b_cc_max_components: int = 4
    mode_b_min_visible_selected: int = 250
    # mode-a MPM only, material=jelly: lower E = softer deformation; higher grid_v_damping_scale
    # (closer to 1) = less velocity damping in PhysGaussian = longer visible oscillation.
    mode_a_jelly_E: float | None = None
    mode_a_jelly_grid_v_damping_scale: float | None = None
    # mode-b MPM in-place jelly (scene object selection): support from selection + COM pin in PhysGaussian.
    mode_b_support_contact_percentile: float = 92.0
    mode_b_pin_initial_com: bool = True
    # False = fix COM in X,Y,Z (furniture won't slowly drift/lean sideways). True only if you want horizontal slosh.
    mode_b_pin_com_vertical_only: bool = False
    mode_b_pin_zero_mean_velocity_gs: bool = True
    mode_b_mpm_in_place_shear_wobble: bool = False
    mode_b_jelly_gravity_mult: float = 0.24
    # Furniture: stiffer than toy jelly; shear off by default (avoids twist/splay).
    mode_b_jelly_E: float | None = 200000.0
    mode_b_jelly_grid_v_damping_scale: float | None = 0.99990
    mode_b_shear_wobble_velocity: float = 0.12
    mode_b_shear_wobble_end_time: float = 0.18
    # L/R shear boxes cover full selected height (tabletop + legs) when True (mode-b only).
    mode_b_shear_wobble_full_volume: bool = True
    # PhysGaussian: sustained sinusoidal shear velocity Dirichlet (MPM physics), not kinematic fallback.
    mode_b_phys_sustained_wobble: bool = True
    mode_b_phys_wobble_frequency_hz: float = 1.35
    # Legacy single-axis peak (m/s into sin); used if mode_b_phys_wobble_velocity_peak_x is None.
    mode_b_phys_wobble_velocity_peak: float = 0.09
    mode_b_phys_wobble_velocity_peak_x: float | None = None
    mode_b_phys_wobble_velocity_peak_y: float = 0.055
    # Extra rad added to vertical sin vs horizontal (default π/2 for jelly lag).
    mode_b_phys_wobble_phase_y: float = 1.5707963267948966
    mode_b_phys_wobble_band_amp_x: list[float] | None = None
    mode_b_phys_wobble_band_amp_y: list[float] | None = None
    # Per vertical band (top→mid→bottom): squash top vs bottom to limit net Y impulse.
    mode_b_phys_wobble_band_vy_polarity: list[float] | None = None
    mode_b_phys_wobble_band_phase_y_offset_rad: list[float] | None = None
    mode_b_phys_wobble_velocity_peak_z: float = 0.08
    mode_b_phys_wobble_velocity_peak_diag1: float = 0.1
    mode_b_phys_wobble_velocity_peak_diag2: float = 0.1
    mode_b_phys_wobble_velocity_peak_twist: float = 0.12
    mode_b_phys_wobble_phase_z: float = 0.35
    mode_b_phys_wobble_phase_diag1_rad: float = 0.7853981633974483
    mode_b_phys_wobble_phase_diag2_rad: float = 2.356194490192345
    mode_b_phys_wobble_phase_twist_rad: float = 1.0471975511965976
    mode_b_phys_wobble_bundle_quad_phase_rad: list[float] | None = None
    mode_b_phys_wobble_bundle_band_phase_rad: list[float] | None = None
    mode_b_phys_wobble_band_amp_z: list[float] | None = None
    mode_b_phys_wobble_band_amp_diag1: list[float] | None = None
    mode_b_phys_wobble_band_amp_diag2: list[float] | None = None
    mode_b_phys_wobble_band_amp_twist: list[float] | None = None
    mode_b_phys_wobble_force_scale: float = 1.0
    mode_b_phys_wobble_decay_lambda_per_s: float = 0.0
    mode_b_phys_wobble_duration_s: float | None = None
    mode_b_phys_wobble_ramp_time_s: float = 0.45
    mode_b_phys_wobble_motion_seconds: float | None = None
    # Stronger snap-back fights one-way lean; lower so periodic driver survives retention blend.
    mode_b_mpm_displacement_retention: float | None = 0.86
    # Render: keep 3DGS ellipsoid shape from sim t=0 (MPM F often spikes splats into needles).
    mode_b_render_freeze_gaussian_cov: bool = True
    # --- Render-space artifact checks (debug; does not change physics) ---
    # If not "none", the PhysGaussian shim will override cov3D_precomp right before rasterization.
    # Options:
    # - "none": normal rendering
    # - "tiny_splats": render selected Gaussians as tiny isotropic splats (point-like)
    # - "clamp_diag": clamp selected cov diagonals into [min,max] and zero off-diagonals
    # - "freeze_first": freeze selected cov3D_precomp to frame-0 values
    mode_b_render_cov_override: str = "none"
    # Apply cov override to "selected" only (default) or "all".
    mode_b_render_cov_override_scope: str = "selected"
    # tiny_splats: isotropic variance value (in camera/world units of cov3D_precomp; small => point-like)
    mode_b_render_tiny_splats_var: float = 1.0e-6
    # clamp_diag: clamp range for diagonal cov entries.
    mode_b_render_cov_diag_min: float = 1.0e-6
    mode_b_render_cov_diag_max: float = 5.0e-3

    # --- Mode-b background compositing (debug/postprocess) ---
    # If True, after mode-b video render, build a cam0 inpainted background plate and composite
    # the rendered frames over it (fills black holes revealed by collapse).
    mode_b_background_composite_enabled: bool = False
    # Use this cam0 mask (generated by SAM2) to remove the chair from the background plate.
    mode_b_background_mask_path: str | None = None
    mode_b_background_inpaint_radius_px: int = 5
    # Per-frame composite mask: pixels with abs(frame - plate) > threshold (0..255) are treated as foreground.
    mode_b_background_diff_threshold_u8: int = 18
    mode_b_background_diff_dilate_px: int = 3
    # If True, composite using projected selected-object mask (per-frame means3D) rather than diff(frame,plate).
    mode_b_background_objectmask_composite: bool = True
    mode_b_background_objectmask_point_radius_px: int = 2
    mode_b_background_objectmask_dilate_px: int = 6
    mode_b_background_objectmask_blur_px: int = 11

    # Save selected means3D per frame for cam0 projection / compositing.
    mode_b_save_selected_means3d_per_frame: bool = False

    # --- Mode-b extra render dumps (debug) ---
    # If True, PhysGaussian shim writes additional per-frame PNGs for selected-only renders:
    # - normal selected-only
    # - clamp_diag selected-only
    # - tiny_splats selected-only
    mode_b_debug_dump_selected_only_renders: bool = False

    # --- Mode-b hard background compositing (final-only output) ---
    # If True, write <out>/video/final_chair_only_collapse.mp4 using:
    # - background plate for whole frame
    # - simulated render ONLY inside a strict chair mask
    mode_b_background_hard_chair_only: bool = False
    # Final composite: photoreal selected-only RGB + its render mask as alpha matte (not full sim / not clamp/tiny).
    mode_b_background_selected_only_matte: bool = False
    # Morphological close on binary mask to remove holes inside the chair silhouette before feathering.
    mode_b_background_alpha_close_kernel_px: int = 25
    # Light Gaussian blur on alpha after close (odd size; 0 = off).
    mode_b_background_alpha_feather_blur_px: int = 7
    # Clamp projected per-frame chair mask to an "allowed region" derived from the original chair mask.
    mode_b_background_allowed_region_dilate_px: int = 8
    mode_b_background_allowed_region_down_extend_px: int = 220
    # Exclude pixels above (min_v - up_exclude_px) from the allowed region to block window/curtain.
    mode_b_background_allowed_region_up_exclude_px: int = 40
    # Full sim frames + inpainted plate: replace only a cleanup mask (black blob / upper window / dark halo).
    # Writes video/final_cleanup_fullframe.mp4; does not use selected-only RGB.
    mode_b_fullframe_plate_cleanup: bool = False
    mode_b_fullframe_cleanup_dark_thresh_u8: int = 42
    mode_b_fullframe_cleanup_protect_erode_px: int = 5
    mode_b_fullframe_cleanup_near_chair_dilate_px: int = 64
    mode_b_fullframe_cleanup_feather_px: int = 5
    # PhysGaussian shim: "all" dumps normal+clamp+tiny; "normal_only" dumps only selected_only_normal (+mask).
    mode_b_extra_render_dump_variants: str = "all"
    # After mode-b: delete debug PNGs, frame folders, extra videos; keep small config + indices + main mp4s.
    mode_b_minimal_output: bool = False

    # --- Mode-b displacement propagation (render-time, debug) ---
    # If True, propagate displacement from strongly moved neighbors to weakly moved selected points.
    # This is a render-time fix (means3D only), does not change MPM state.
    mode_b_disp_propagation_enabled: bool = False
    mode_b_disp_propagation_k: int = 8
    mode_b_disp_low_percentile: float = 10.0
    mode_b_disp_min_neighbor_moved_m: float = 0.05
    mode_b_disp_propagation_alpha: float = 1.0
    # --- Mode-b sand: render-side appearance override (does not change physics / means3D) ---
    # If True, PhysGaussian subprocess will override selected Gaussians' render appearance
    # (opacity/scale/color) after simulation, to avoid preserving chair-like splat look.
    mode_b_sand_render_override: bool = False
    # Multiply selected Gaussians' *render* opacity (sigmoid space) by this factor (clamped).
    mode_b_sand_opacity_scale: float = 1.0
    # Clamp selected Gaussians' render opacity into [min,max] after scaling.
    mode_b_sand_opacity_min: float = 0.05
    mode_b_sand_opacity_max: float = 0.85
    # Multiply selected Gaussians' render scale/covariance (ellipsoid radii) by this factor.
    # Values < 1.0 make splats tighter and less chair-like.
    mode_b_sand_cov_scale: float = 1.0
    # Optional RGB override for selected Gaussians (0..1). If set, SH will be replaced and
    # higher-order SH zeroed to reduce texture bias.
    mode_b_sand_color_override_rgb: list[float] | None = None

    # --- Mode-b sand: motion-side post-simulation correction (render-time means3D only) ---
    # If True, enforce a coherent global downward settling component for all selected Gaussians.
    # This does NOT change PhysGaussian MPM state; it only adjusts the means3D tensor passed into rendering.
    mode_b_sand_motion_correction: bool = False
    # Blend factor α in: corrected_dy = (1-α)*dy_i + α*global_dy. Higher α → more uniform sinking.
    mode_b_sand_motion_alpha: float = 0.65
    # Use this percentile of per-gaussian dy as global_dy (robust; ignores extreme outliers).
    mode_b_sand_global_dy_percentile: float = 60.0
    # Optional: cap upward motion (dy<0) by clamping to at least this fraction of global_dy.
    mode_b_sand_min_dy_as_global_frac: float | None = 0.15

    # --- Mode-b sand: EXTREME binary sanity check (debug) ---
    # If True, ignore MPM dy for rendering and force a large coherent downward translation
    # for ALL selected Gaussians, linearly ramped from frame 0→(frame_num-1).
    mode_b_sand_extreme_all_selected_down: bool = False
    # Total downward offset (meters) applied by the final frame.
    mode_b_sand_extreme_total_down_m: float = 0.60

    # --- Mode-b render contribution audit (true subset rendering) ---
    # Render only a subset of gaussians (true: pass only that subset to rasterizer).
    # Options: "all", "selected_only_true", "unselected_only_true", "id_colored_sets"
    mode_b_render_gaussian_subset: str = "all"

    # --- Mode-b selection completion (fix missing visible parts like backrest) ---
    # After the initial mask→gaussians selection + bbox/shell/CC filtering, expand selection
    # by projecting gaussians into the chosen render camera and adding those whose projected
    # centers overlap the target mask, with a 3D AABB + rendered-depth gate.
    mode_b_selection_completion_enabled: bool = False
    mode_b_selection_completion_aabb_expand_ratio: float = 0.10
    mode_b_selection_completion_max_add: int = 25000
    # Dilation radius in pixels applied to the target mask before completion.
    # This approximates footprint overlap (not just center-in-mask) without needing per-Gaussian covariances.
    mode_b_selection_completion_mask_dilate_px: int = 8
    # Limit completion to a local 3D neighborhood around the *original* selected gaussians.
    # This is the main guardrail to avoid pulling in wall/window behind the object.
    mode_b_selection_completion_neighbor_radius_m: float = 0.28
    # After completion, optionally PRUNE the selection to remove background contamination:
    # Keep only gaussians that (a) are near the surface seeds in 3D and (b) project inside the mask
    # with a conservative depth gate (reject behind-chair).
    mode_b_selection_prune_enabled: bool = False
    mode_b_selection_prune_neighbor_radius_m: float = 0.28
    # Use mask dilation only for footprint tolerance; depth reference remains anchored to original mask.
    mode_b_selection_prune_mask_dilate_px: int = 0
    # Depth gate tolerances (absolute meters + relative fraction of z).
    mode_b_selection_prune_depth_abs_m: float | None = None
    mode_b_selection_prune_depth_rel: float | None = None
    # Off for mode-b furniture jelly: Kabsch strip projects out rigid motion each frame and hides whole-table wobble.
    mode_b_mpm_kabsch_rigid_strip: bool = False
    mode_b_mpm_kabsch_elastic_amp: float = 1.0
    # Higher percentile → fewer Gaussians locked at feet → legs can participate in jelly motion.
    mode_b_mpm_anchor_feet_y_percentile: float | None = 92.0
    # L/R mirrored shear boxes (mode-b): reduces net +X impulse from Y-band shear.
    mode_b_mpm_shear_symmetric_lr_split: bool = True
    # Per-frame tilt metrics JSON (mode_b_mpm_tilt_series.json) + summary in mode_b_mpm_debug.json.
    mode_b_mpm_tilt_diagnostics: bool = True
    # Rest MPM-Y at or below this population percentile = "top" slab for L/R height tilt (low Y = tabletop).
    mode_b_mpm_tilt_top_y_percentile: float = 32.0
    # Tighter AABB around selection for mode-b (still need index filter in PhysGaussian).
    mode_b_sim_area_margin: float | None = 0.2


class EvaluationConfig(BaseModel):
    test_views: list[int] = Field(default_factory=lambda: [0, 5, 10, 15, 20])
    lpips_net: str = "alex"


class PathsConfig(BaseModel):
    merged_output: str = "output/merged_scenes"
    sim_output: str = "output/sim_results"
    generated_objects: str = "output/generated_objects"
    eval_output: str = "output/eval"


class ModeBPresetConfig(BaseModel):
    """Named mode-b profiles (prompt/material/output + optional physics overrides)."""

    model_config = ConfigDict(extra="ignore")

    selection_prompt: str
    physics_type: str = "jelly"  # material preset name (see physics/material_presets.py)
    output_subdir: str = "mode_b_jelly"
    # Optional: force using smoke 3DGS checkpoint for this preset unless CLI overrides.
    use_smoke_3dgs: bool | None = None
    physics_overrides: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scene: SceneConfig = Field(default_factory=SceneConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    mode_b_presets: dict[str, ModeBPresetConfig] = Field(default_factory=dict)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> PipelineConfig:
        p = Path(path) if path else PROJECT_ROOT / "configs" / "pipeline_config.yaml"
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def resolve(self, rel: str) -> Path:
        return (PROJECT_ROOT / rel).resolve()
