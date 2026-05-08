"""Load PRD-style pipeline_config.yaml into typed settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SceneConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset_url: str = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    data_root: str = "data/mipnerf360"
    scene_name: str = "room"
    model_output: str = "output/base_scene"
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
