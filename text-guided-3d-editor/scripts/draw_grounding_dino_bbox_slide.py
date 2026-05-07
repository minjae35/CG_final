#!/usr/bin/env python3
"""
Draw Grounding DINO best box on the same RGB used for mode-b mask debug (01_rgb.png).

Uses repository segmentation.grounded_sam2_mask.grounded_dino_best_box_xyxy_confidence
(no SAM2, no generative models) + OpenCV.

Example:
  cd text-guided-3d-editor && export PYTHONPATH=src
  python scripts/draw_grounding_dino_bbox_slide.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PROJECT_ROOT as REPO_ROOT, PipelineConfig  # noqa: E402
from segmentation.debug_overlay_style import DEBUG_GREEN_BGR  # noqa: E402
from segmentation.grounded_sam2_mask import grounded_dino_best_box_xyxy_confidence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rgb",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "sim_results"
        / "mode_b_jelly"
        / "debug_selection"
        / "01_rgb.png",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "sim_results"
        / "mode_b_jelly"
        / "debug_selection"
        / "grounding_dino_bbox_wooden_table.png",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="wooden table in the foreground",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "pipeline_config.yaml",
    )
    ap.add_argument("--label", type=str, default="wooden table")
    args = ap.parse_args()

    rgb_path = args.rgb.resolve()
    if not rgb_path.is_file():
        raise SystemExit(f"Missing RGB: {rgb_path}")

    cfg = PipelineConfig.load(args.config)
    seg = cfg.segmentation
    gs2 = (REPO_ROOT / "submodules" / "Grounded-SAM-2").resolve()

    def _r(p: str | None, rel: str) -> Path:
        if not p:
            return (gs2 / rel).resolve()
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (REPO_ROOT / pp).resolve()

    gd_cfg = _r(seg.grounding_dino_config, "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    gd_ckpt = _r(seg.grounding_dino_checkpoint, "gdino_checkpoints/groundingdino_swinb_cogcoor.pth")

    box = grounded_dino_best_box_xyxy_confidence(
        rgb_path,
        args.prompt,
        gs2_root=gs2,
        gdino_config=gd_cfg,
        gdino_checkpoint=gd_ckpt,
        box_threshold=float(seg.box_threshold),
        text_threshold=float(seg.text_threshold),
        device=str(seg.segmentation_device),
    )

    img = cv2.imread(str(rgb_path))
    if img is None:
        raise SystemExit(f"Could not read {rgb_path}")

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if box is None:
        cv2.putText(
            img,
            "Grounding DINO: no box",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), img)
        print(f"source_image={rgb_path}", flush=True)
        print("bbox_xyxy=None  confidence=None", flush=True)
        print(f"wrote(no_box)={out_path}", flush=True)
        return

    xyxy, conf = box
    x1, y1, x2, y2 = [int(round(c)) for c in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), DEBUG_GREEN_BGR, 3, lineType=cv2.LINE_AA)
    label = f"{args.label}  ({conf:.3f})"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    ty = max(y1 - 8, th + 12)
    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 12, ty + baseline - 2), (32, 32, 32), -1)
    cv2.putText(img, label, (x1 + 6, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"source_image={rgb_path}", flush=True)
    print(f"bbox_xyxy_pixel={xyxy.tolist()}  confidence={conf:.6f}", flush=True)
    print(f"wrote={out_path}", flush=True)


if __name__ == "__main__":
    main()
