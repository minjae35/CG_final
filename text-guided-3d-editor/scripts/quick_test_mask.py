#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from segmentation.text_to_mask import text_to_mask


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate and save a binary mask for a rendered view.")
    ap.add_argument("--image", required=True, help="Path to rgb_*.png (e.g. output/renders_views/rgb_00000.png)")
    ap.add_argument("--text", required=True, help='Text prompt (e.g. "the chair")')
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument(
        "--device",
        default=None,
        help="Segmentation device: cuda or cpu (default: auto / config defaults).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output mask path (default: alongside image as mask_<stem>.png)",
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="Also save an overlay image (original RGB + red mask).",
    )
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(img_path)

    out_path = Path(args.out) if args.out else (img_path.parent / f"mask_{img_path.stem}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask = text_to_mask(
        str(img_path),
        args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        segmentation_device=args.device,
    )

    # Save as 8-bit image for quick inspection.
    out = (mask.astype(np.uint8) * 255)
    Image.fromarray(out).save(out_path)

    if args.overlay:
        img = Image.open(img_path).convert("RGB")
        mask_img = Image.fromarray(out).convert("L")
        # Red overlay with alpha
        red = Image.new("RGB", img.size, ImageColor.getrgb("#ff2d2d"))
        blended = Image.composite(red, img, mask_img)
        # Keep some of original for readability
        overlay = Image.blend(img, blended, alpha=0.45)
        overlay_path = out_path.with_name(f"overlay_{img_path.stem}.png")
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)
        print(f"saved: {overlay_path}")

    ratio = float(mask.mean()) if mask.size else 0.0
    print(f"saved: {out_path}")
    print(f"mask_ratio: {ratio:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

