"""Reference image generator for the image-to-3D path.

Uses Stable Diffusion 1.5 (already cached locally) to render a single
front-view, white-background image of the requested object.  The generated
PNG is then handed to DreamGaussian's ``process.py`` (rembg) which adds an
alpha channel and centres the subject before the image-to-3D stage 1 run.

Even with strong "single subject" prompts, SD 1.5 sometimes paints two ducks
side by side.  ``_keep_largest_subject`` enforces uniqueness at the pixel
level using rembg + connected components, so DreamGaussian only ever sees
one duck regardless of what SD produced.
"""
from __future__ import annotations

import gc
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_DEFAULT_NEG = (
    "two, three, four, five, six, multiple, many, several, group, set, collection, "
    "pair, twin, duplicate, copy, side by side, lined up, in a row, cluster, "
    "stacked, totem, on top of each other, "
    "two objects, two ducks, two vases, two balls, two mugs, two chairs, "
    "second object, extra object, additional object, "
    "from above, top-down view, bird's eye view, overhead shot, looking down, "
    "tilted view, lying down, on its side, on its back, on the ground, "
    "creepy, scary, grotesque, uncanny, deformed, melted, malformed, asymmetric, "
    "blurry, low quality, jpeg artifacts, extra limbs, watermark, "
    "text, label, logo, cluttered background, complex background, "
    "grey background, dark background, low-poly, painting, drawing, sketch, "
    "cartoon, anime"
)


def _build_image_prompt(prompt: str) -> str:
    """CLIP only keeps ~77 tokens — keep the subject + 'solo' constraints up front."""
    raw = prompt.strip().rstrip(".")
    low = raw.lower()

    if "duck" in low:
        if "red" in low:
            return (
                "exactly one red rubber duck toy, solo, single subject only, no second duck, "
                "yellow beak, black eyes, glossy plastic, rounded body, upright with flat base, "
                "eye-level three-quarter view, pure white background, centered, sharp product photo"
            )
        return (
            "exactly one classic yellow rubber duck bath toy, solo, single subject only, "
            "no second duck, orange beak, black eyes, glossy plastic, rounded body, "
            "upright with flat base, eye-level three-quarter view, pure white background, "
            "centered, sharp product photo"
        )

    if "ball" in low or "sphere" in low:
        colour = ""
        for c in ("red", "yellow", "blue", "green", "orange", "white", "black", "pink", "purple"):
            if c in low:
                colour = c + " "
                break
        return (
            f"exactly one {colour}rubber ball, solo, single object only, no duplicates, "
            f"perfectly round sphere, smooth matte rubber, soft seam, sitting on flat surface, "
            f"eye-level three-quarter view, pure white background, centered, sharp product photo"
        )

    if "vase" in low:
        colour = ""
        for c in ("white", "blue", "green", "black", "red", "pink", "yellow", "beige", "brown", "grey"):
            if c in low:
                colour = c + " "
                break
        return (
            f"exactly one empty {colour}ceramic vase, solo, single object only, no duplicates, "
            f"glossy porcelain, classic round body with a narrow neck, axially symmetric, "
            f"upright on a flat base, no flowers, no decorations, "
            f"eye-level three-quarter view, pure white background, centered, sharp product photo"
        )

    # Generic object: short prompt so CLIP does not truncate away the solo constraint.
    return (
        f"exactly one {raw}, solo, single object only, no duplicates, toy-like, upright, "
        f"flat base, eye-level three-quarter view, pure white background, centered, sharp product photo"
    )


def _count_projection_peaks(mask: np.ndarray, axis: int = 0) -> int:
    """Smooth column/row projection of ``mask`` and count distinct peaks.

    Touching subjects (e.g. four vases lined up) often merge into a single
    connected component but still produce multi-modal projections.  We detect
    that here so the multi-subject penalty kicks in even when CC counts say 1.
    """
    if mask.size == 0:
        return 0
    proj = mask.sum(axis=axis).astype(np.float32)
    if proj.max() <= 0:
        return 0
    k = max(5, int(proj.shape[0] * 0.01) | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(proj, kernel, mode="same")
    peak_thr = 0.55 * float(smooth.max())
    valley_thr = 0.30 * float(smooth.max())
    min_span = max(8, int(proj.shape[0] * 0.04))
    n_peaks = 0
    above = False
    counted = False
    span = 0
    for v in smooth:
        if v >= peak_thr:
            if not above:
                above = True
                span = 1
                counted = False
            else:
                span += 1
        else:
            if v <= valley_thr:
                if above and span >= min_span and not counted:
                    n_peaks += 1
                    counted = True
                above = False
                span = 0
    if above and span >= min_span and not counted:
        n_peaks += 1
    return int(n_peaks)


def _score_candidate_info(info: dict) -> float:
    """Higher is better. Heavy weight on TRUE single subject; size/centre auto-fixed downstream."""
    h = max(int(info.get("h", 0)), 1)
    w = max(int(info.get("w", 0)), 1)
    img_area = float(h * w)
    kept = float(info.get("kept_area", 0))
    removed = float(info.get("removed_area", 0))
    n_components = int(info.get("n_components", 0))
    n_peaks = int(info.get("n_peaks", 1))
    if kept <= 0:
        return 0.0
    purity = kept / max(kept + removed, 1.0)
    area_ratio = kept / img_area
    if area_ratio < 0.02:
        sweet = 0.2
    elif area_ratio < 0.05:
        sweet = 0.6
    elif area_ratio < 0.60:
        sweet = 1.0
    else:
        sweet = max(0.0, 1.0 - 1.5 * (area_ratio - 0.60))
    cc_penalty = 1.0 if n_components <= 1 else 1.0 / (1.0 + 1.5 * (n_components - 1))
    peaks_penalty = 1.0 if n_peaks <= 1 else 1.0 / (1.0 + 2.5 * (n_peaks - 1))
    return float(purity ** 2.0 * sweet * cc_penalty * peaks_penalty)


def _center_and_resize_subject(
    rgb: np.ndarray,
    keep_mask: np.ndarray,
    *,
    bg_rgb: tuple[int, int, int] = (245, 245, 245),
    fill_ratio: float = 0.70,
) -> np.ndarray:
    """Tight-crop the masked subject, square-pad it, then paste it centred on a fresh background.

    The subject ends up occupying ~``fill_ratio`` of the shorter image side, which
    is the canonical layout that DreamGaussian image-to-3D expects.  Without this,
    a tiny subject in a corner of the SD output produces a near-empty 3D model.
    """
    import cv2

    h, w = keep_mask.shape
    ys, xs = np.where(keep_mask)
    if ys.size == 0:
        return np.full((h, w, 3), bg_rgb, dtype=np.uint8)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub_h = y1 - y0
    sub_w = x1 - x0
    side = max(sub_h, sub_w)
    pad = max(int(round(side * 0.06)), 4)
    side_padded = side + 2 * pad

    sq = np.full((side_padded, side_padded, 3), bg_rgb, dtype=np.uint8)
    oy = pad + (side - sub_h) // 2
    ox = pad + (side - sub_w) // 2
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = keep_mask[y0:y1, x0:x1]
    paste = sq[oy : oy + sub_h, ox : ox + sub_w]
    paste[crop_mask] = crop_rgb[crop_mask]
    sq[oy : oy + sub_h, ox : ox + sub_w] = paste

    out_side = min(h, w)
    target_subject = max(8, int(round(out_side * float(fill_ratio))))
    scale = target_subject / float(side_padded)
    new_size = max(8, int(round(side_padded * scale)))
    sq_resized = cv2.resize(sq, (new_size, new_size), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((h, w, 3), bg_rgb, dtype=np.uint8)
    cy0 = (h - new_size) // 2
    cx0 = (w - new_size) // 2
    canvas[cy0 : cy0 + new_size, cx0 : cx0 + new_size] = sq_resized
    return canvas


def _keep_largest_subject(
    image: Image.Image,
    *,
    bg_rgb: tuple[int, int, int] = (245, 245, 245),
    min_area_ratio: float = 0.01,
    fill_ratio: float = 0.70,
) -> tuple[Image.Image, dict]:
    """Return ``(rgb, info)`` where rgb keeps only the largest subject blob on a flat background.

    Uses ``rembg`` to get an alpha mask, then ``cv2.connectedComponentsWithStats``
    to keep the single largest non-background component.  Other subjects (e.g. a
    second duck) are repainted with ``bg_rgb`` so DreamGaussian's image-to-3D
    path receives only one shape.
    """
    import cv2
    from rembg import remove

    if image.mode != "RGBA":
        image_rgba_bytes = io.BytesIO()
        image.convert("RGB").save(image_rgba_bytes, format="PNG")
        cut = remove(image_rgba_bytes.getvalue())
        cut_pil = Image.open(io.BytesIO(cut)).convert("RGBA")
    else:
        cut_pil = image

    rgba = np.asarray(cut_pil, dtype=np.uint8).copy()
    alpha = rgba[..., 3]
    binary = (alpha > 16).astype(np.uint8)
    h, w = binary.shape
    img_area = float(h * w)
    info: dict = {"h": int(h), "w": int(w), "n_components": 0, "kept_area": 0, "removed_area": 0}

    if int(binary.sum()) == 0:
        flat = np.full((h, w, 3), bg_rgb, dtype=np.uint8)
        return Image.fromarray(flat, mode="RGB"), info

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    info["n_components"] = int(max(0, n_labels - 1))
    if n_labels <= 1:
        flat = np.full((h, w, 3), bg_rgb, dtype=np.uint8)
        return Image.fromarray(flat, mode="RGB"), info

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    keep_area = int(areas.max())
    info["kept_area"] = keep_area
    info["removed_area"] = int(areas.sum() - keep_area)
    cx, cy = centroids[biggest]
    info["centroid_yx"] = (float(cy), float(cx))

    keep_mask = (labels == biggest)
    fg_rgb = rgba[..., :3]
    info["n_peaks"] = _count_projection_peaks(keep_mask.astype(np.uint8), axis=0)
    if keep_area / max(img_area, 1.0) < float(min_area_ratio):
        info["warning"] = (
            f"largest subject covers only {keep_area / img_area:.3%} of the image — "
            "consider a different --image-seed."
        )
    if float(fill_ratio) > 0.0:
        out = _center_and_resize_subject(fg_rgb, keep_mask, bg_rgb=bg_rgb, fill_ratio=fill_ratio)
        info["normalised_fill_ratio"] = float(fill_ratio)
    else:
        out = np.full((h, w, 3), bg_rgb, dtype=np.uint8)
        out[keep_mask] = fg_rgb[keep_mask]
    return Image.fromarray(out, mode="RGB"), info


def generate_reference_image(
    prompt: str,
    out_path: Path | str,
    *,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    width: int = 768,
    height: int = 768,
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    seed: int | None = 0,
    negative_prompt: str | None = None,
    enforce_single_subject: bool = True,
    num_candidates: int = 4,
) -> Path:
    """Generate ``num_candidates`` SD candidates with seeds ``[seed, seed+1, ...]``.

    All candidates are kept under ``out_path.parent / 'cute_candidates'``.  The
    one that scores best on (single-subject purity × area sweet spot × centring)
    is saved as ``out_path``.  Falls back to a single image if rembg/cv2 are
    unavailable.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    pos_prompt = _build_image_prompt(prompt)
    neg_prompt = negative_prompt if negative_prompt is not None else _DEFAULT_NEG

    n_cand = max(1, int(num_candidates))
    base_seed = int(seed) if seed is not None else 0
    candidates_dir = out_path.parent / "cute_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reference] generating {n_cand} candidate(s) from base_seed={base_seed} (steps={num_inference_steps}).")
    raw_images: list[Image.Image] = []
    for k in range(n_cand):
        gen = torch.Generator(device="cuda").manual_seed(base_seed + k)
        with torch.inference_mode():
            res = pipe(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                width=int(width),
                height=int(height),
                generator=gen,
            )
        raw_images.append(res.images[0])
        torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    best_image = raw_images[0]
    best_info: dict = {}
    best_score = -1.0
    best_idx = 0

    if not enforce_single_subject:
        for k, img in enumerate(raw_images):
            img.save(str(candidates_dir / f"raw_seed_{base_seed + k:04d}.png"))
        best_image.save(str(out_path))
        return out_path

    summary: list[tuple[int, float, dict]] = []
    for k, img in enumerate(raw_images):
        seed_k = base_seed + k
        try:
            cleaned, info = _keep_largest_subject(img)
            score = _score_candidate_info(info)
        except Exception as exc:  # noqa: BLE001 — fall back to raw image for this seed.
            print(f"[reference] seed {seed_k}: post-processing failed ({exc}); keeping raw.")
            cleaned, info, score = img.convert("RGB"), {}, 0.0
        cleaned.save(str(candidates_dir / f"cute_seed_{seed_k:04d}.png"))
        summary.append((seed_k, float(score), info))
        if score > best_score:
            best_score, best_image, best_info, best_idx = score, cleaned, info, k

    print("[reference] candidate scores (seed → score, n_components/n_peaks, kept/total area):")
    for seed_k, score, info in summary:
        h = max(int(info.get("h", 1)), 1)
        w = max(int(info.get("w", 1)), 1)
        kept = int(info.get("kept_area", 0))
        nc = int(info.get("n_components", 0))
        npks = int(info.get("n_peaks", 0))
        marker = "  <- chosen" if seed_k == base_seed + best_idx else ""
        print(f"  seed={seed_k:04d}  score={score:.4f}  n_subj={nc}  n_peaks={npks}  area={kept}/{h * w}{marker}")
    note = best_info.get("warning")
    if note:
        print(f"[reference] {note}")

    best_image.save(str(out_path))
    return out_path
