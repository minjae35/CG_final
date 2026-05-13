#!/usr/bin/env python3
"""Export Mode-B multi-view segmentation visuals for report figures.

This script intentionally reuses the Mode-B rendered RGB views in
``output/renders_views`` and does not run physics, 3DGS training, or video
rendering.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PipelineConfig  # noqa: E402
from segmentation.grounded_sam2_mask import grounded_dino_best_box_xyxy_confidence  # noqa: E402
from segmentation.text_to_mask import text_to_mask  # noqa: E402


PRESET_ALIASES = {
    "desk": ("desk_jelly", "desk"),
    "desk_jelly": ("desk_jelly", "desk"),
    "armchair": ("armchair_jelly", "armchair"),
    "armchair_jelly": ("armchair_jelly", "armchair"),
}


def _resolve_path(path: str | Path, *, documents_relative_to_main: bool = False) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    if documents_relative_to_main and p.parts and p.parts[0] == "documents":
        return (MAIN_REPO_ROOT / p).resolve()
    return (Path.cwd() / p).resolve()


def _resolve_asset(repo_root: Path, gs2_root: Path, path: str | Path | None, default_rel: str) -> Path:
    if path is None or str(path) == "":
        return (gs2_root / default_rel).resolve()
    p = Path(path)
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()


def _load_existing_mask(mask_path: Path, expected_size: tuple[int, int]) -> np.ndarray | None:
    if not mask_path.is_file():
        return None
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != expected_size:
        return None
    return np.asarray(mask_img) >= 128


def _save_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def _save_green_overlay(rgb_path: Path, mask: np.ndarray, out_path: Path, alpha: float = 0.4) -> None:
    rgb = Image.open(rgb_path).convert("RGB")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    green = Image.new("RGB", rgb.size, (0, 255, 0))
    tinted = Image.blend(rgb, green, alpha=float(alpha))
    overlay = Image.composite(tinted, rgb, mask_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def _save_bbox_overlay(
    rgb_path: Path,
    out_path: Path,
    *,
    prompt: str,
    cfg: PipelineConfig,
) -> tuple[list[float] | None, float | None]:
    seg = cfg.segmentation
    gs2_root = (PROJECT_ROOT / "submodules" / "Grounded-SAM-2").resolve()
    gd_cfg = _resolve_asset(
        PROJECT_ROOT,
        gs2_root,
        seg.grounding_dino_config or None,
        "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    )
    gd_ckpt = _resolve_asset(
        PROJECT_ROOT,
        gs2_root,
        seg.grounding_dino_checkpoint or None,
        "gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
    )

    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")

    result = grounded_dino_best_box_xyxy_confidence(
        rgb_path,
        prompt,
        gs2_root=gs2_root,
        gdino_config=gd_cfg,
        gdino_checkpoint=gd_ckpt,
        box_threshold=float(seg.box_threshold),
        text_threshold=float(seg.text_threshold),
        device=str(seg.segmentation_device),
    )
    if result is None:
        bbox_xyxy = None
        confidence = None
    else:
        xyxy, confidence = result
        bbox_xyxy = [float(x) for x in xyxy.tolist()]
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return bbox_xyxy, confidence


def _get_mask_for_view(
    rgb_path: Path,
    *,
    prompt: str,
    cfg: PipelineConfig,
    preset_out_subdir: str,
    stem: str,
    report_mask_path: Path,
    allow_cpu_fallback: bool,
) -> tuple[np.ndarray, str]:
    expected_size = Image.open(rgb_path).convert("RGB").size
    existing_report_mask = _load_existing_mask(report_mask_path, expected_size)
    if existing_report_mask is not None:
        return existing_report_mask, str(report_mask_path)

    mode_b_mask_path = (
        PROJECT_ROOT
        / "output"
        / "sim_results"
        / preset_out_subdir
        / "debug"
        / "masks"
        / f"rgb_{stem}_mask_binary.png"
    )
    existing_mode_b_mask = _load_existing_mask(mode_b_mask_path, expected_size)
    if existing_mode_b_mask is not None:
        return existing_mode_b_mask, str(mode_b_mask_path)

    seg = cfg.segmentation
    mask = text_to_mask(
        str(rgb_path),
        prompt,
        seg.box_threshold,
        seg.text_threshold,
        gdino_config=seg.grounding_dino_config or None,
        gdino_checkpoint=seg.grounding_dino_checkpoint or None,
        sam2_config=seg.sam2_config or None,
        sam2_checkpoint=seg.sam2_checkpoint or None,
        device=seg.segmentation_device,
        allow_cpu_fallback=allow_cpu_fallback,
        log_progress=True,
    )
    return mask.astype(bool), "generated"


def export_preset(
    *,
    preset_arg: str,
    cfg: PipelineConfig,
    output_dir: Path,
    allow_cpu_fallback: bool,
    overlay_alpha: float,
) -> list[Path]:
    if preset_arg not in PRESET_ALIASES:
        raise SystemExit(f"Unknown preset {preset_arg!r}. Use one of: {sorted(PRESET_ALIASES)}")
    config_key, prefix = PRESET_ALIASES[preset_arg]
    if config_key not in cfg.mode_b_presets:
        raise SystemExit(f"Missing config mode_b_presets.{config_key}")

    preset = cfg.mode_b_presets[config_key]
    prompt = str(preset.selection_prompt)
    preset_out_subdir = str(preset.output_subdir)
    render_dir = PROJECT_ROOT / "output" / "renders_views"
    rgbs = sorted(render_dir.glob("rgb_*.png"))[: int(cfg.segmentation.multi_view_count)]
    if len(rgbs) < int(cfg.segmentation.multi_view_count):
        raise SystemExit(
            f"Expected {cfg.segmentation.multi_view_count} rendered RGB views in {render_dir}, found {len(rgbs)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stride = int(cfg.reconstruction.render_view_stride)
    print(f"[{prefix}] prompt={prompt!r}  views={len(rgbs)}  stride={stride}", flush=True)

    for view_idx, rgb_path in enumerate(rgbs):
        stem = rgb_path.stem.replace("rgb_", "")
        try:
            train_camera_idx = int(stem) * stride
        except ValueError:
            train_camera_idx = view_idx * stride

        rgb_out = output_dir / f"{prefix}_view{view_idx}_rgb.png"
        bbox_out = output_dir / f"{prefix}_view{view_idx}_bbox.png"
        mask_out = output_dir / f"{prefix}_view{view_idx}_mask.png"
        overlay_out = output_dir / f"{prefix}_view{view_idx}_overlay.png"

        shutil.copy2(rgb_path, rgb_out)
        written.append(rgb_out)

        print(
            f"[{prefix}] view{view_idx}: rgb={rgb_path.name}  training_camera_index={train_camera_idx}",
            flush=True,
        )
        mask, mask_source = _get_mask_for_view(
            rgb_path,
            prompt=prompt,
            cfg=cfg,
            preset_out_subdir=preset_out_subdir,
            stem=stem,
            report_mask_path=mask_out,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        _save_mask(mask, mask_out)
        written.append(mask_out)

        bbox_xyxy, confidence = _save_bbox_overlay(rgb_path, bbox_out, prompt=prompt, cfg=cfg)
        written.append(bbox_out)

        _save_green_overlay(rgb_path, mask, overlay_out, alpha=overlay_alpha)
        written.append(overlay_out)
        print(
            f"[{prefix}] view{view_idx}: mask_source={mask_source}  "
            f"mask_px={int(mask.sum())}  bbox={bbox_xyxy}  conf={confidence}",
            flush=True,
        )

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preset",
        choices=["desk", "armchair", "desk_jelly", "armchair_jelly", "all"],
        required=True,
        help="Mode-B preset to export. Use 'all' for desk + armchair.",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipeline_config.yaml",
        help="Pipeline config path.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=MAIN_REPO_ROOT / "documents" / "Final_report" / "images" / "multiview_consensus",
        help="Output directory for report PNGs.",
    )
    ap.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow slow CPU retry if CUDA Grounded-SAM2 fails.",
    )
    ap.add_argument("--overlay-alpha", type=float, default=0.4, help="Green SAM2 mask overlay alpha.")
    args = ap.parse_args()

    config_path = _resolve_path(args.config)
    output_dir = _resolve_path(args.output, documents_relative_to_main=True)
    cfg = PipelineConfig.load(config_path)

    presets = ["desk", "armchair"] if args.preset == "all" else [args.preset]
    all_written: list[Path] = []
    for preset in presets:
        all_written.extend(
            export_preset(
                preset_arg=preset,
                cfg=cfg,
                output_dir=output_dir,
                allow_cpu_fallback=bool(args.allow_cpu_fallback),
                overlay_alpha=float(args.overlay_alpha),
            )
        )

    print("[done] wrote files:", flush=True)
    for path in all_written:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
