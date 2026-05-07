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


class EvaluationConfig(BaseModel):
    test_views: list[int] = Field(default_factory=lambda: [0, 5, 10, 15, 20])
    lpips_net: str = "alex"


class PathsConfig(BaseModel):
    merged_output: str = "output/merged_scenes"
    sim_output: str = "output/sim_results"
    generated_objects: str = "output/generated_objects"
    eval_output: str = "output/eval"


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scene: SceneConfig = Field(default_factory=SceneConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> PipelineConfig:
        p = Path(path) if path else PROJECT_ROOT / "configs" / "pipeline_config.yaml"
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def resolve(self, rel: str) -> Path:
        return (PROJECT_ROOT / rel).resolve()
