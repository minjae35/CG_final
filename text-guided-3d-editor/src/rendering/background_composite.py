from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy import ndimage


def _read_mask_u8(mask_path: Path) -> np.ndarray:
    import cv2

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    # Normalize to 0/255
    m = (m > 127).astype(np.uint8) * 255
    return m


def build_allowed_region_uint8(
    *,
    original_chair_mask_path: Path,
    plate_h: int,
    plate_w: int,
    dilate_px: int,
    down_extend_px: int,
    up_exclude_px: int,
) -> np.ndarray:
    """Binary mask (0/255): chair region + downward extension, with upper window/curtain stripped."""
    import cv2

    orig = _read_mask_u8(original_chair_mask_path)
    if orig.shape[:2] != (plate_h, plate_w):
        orig = cv2.resize(orig, (plate_w, plate_h), interpolation=cv2.INTER_NEAREST)
    allowed = (orig > 127).astype(np.uint8) * 255
    ad = int(max(0, dilate_px))
    if ad > 0:
        yy, xx = np.ogrid[-ad : ad + 1, -ad : ad + 1]
        kernel = ((xx * xx + yy * yy) <= (ad * ad)).astype(np.uint8)
        allowed = cv2.dilate(allowed, kernel, iterations=1)
    de = int(max(0, down_extend_px))
    if de > 0 and de < allowed.shape[0]:
        allowed[de:, :] = np.maximum(allowed[de:, :], allowed[:-de, :])
    ue = int(max(0, up_exclude_px))
    ys, _ = np.where(orig > 127)
    if ys.size:
        y_min = int(ys.min())
        y_cut = max(0, y_min - ue)
        allowed[:y_cut, :] = 0
    return allowed


def _sorted_frame_pngs(frames_dir: Path) -> list[Path]:
    """Sort ``*.png`` by numeric stem (0000.png …) for stable video / frame order."""
    out: list[tuple[int, Path]] = []
    for p in frames_dir.glob("*.png"):
        if p.stem.isdigit():
            out.append((int(p.stem), p))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def make_inpainted_plate(
    *,
    cam0_rgb_path: Path,
    chair_mask_path: Path,
    out_plate_path: Path,
    inpaint_radius_px: int,
) -> Path:
    """Create background plate by inpainting the chair region out of cam0."""
    import cv2

    img = cv2.imread(str(cam0_rgb_path))
    if img is None:
        raise FileNotFoundError(cam0_rgb_path)
    mask = _read_mask_u8(chair_mask_path)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    r = max(1, int(inpaint_radius_px))
    plate = cv2.inpaint(img, mask, r, cv2.INPAINT_TELEA)
    out_plate_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_plate_path), plate)
    return out_plate_path


def composite_frames_over_plate(
    *,
    frames_dir: Path,
    plate_path: Path,
    out_frames_dir: Path,
    diff_threshold_u8: int,
    dilate_px: int,
) -> int:
    """Composite rendered frames over static background plate using simple diff-based foreground mask."""
    import cv2

    plate = cv2.imread(str(plate_path))
    if plate is None:
        raise FileNotFoundError(plate_path)

    frames = _sorted_frame_pngs(frames_dir)
    if not frames:
        raise FileNotFoundError(f"no png frames under {frames_dir}")

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    thr = int(diff_threshold_u8)
    dpx = int(max(0, dilate_px))
    if dpx > 0:
        yy, xx = np.ogrid[-dpx : dpx + 1, -dpx : dpx + 1]
        kernel = ((xx * xx + yy * yy) <= (dpx * dpx)).astype(np.uint8)
    else:
        kernel = None

    wrote = 0
    for p in frames:
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        if fr.shape != plate.shape:
            fr = cv2.resize(fr, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = cv2.absdiff(fr, plate)
        m = (diff.max(axis=2) > thr).astype(np.uint8) * 255
        if kernel is not None:
            m = cv2.dilate(m, kernel, iterations=1)
        m3 = np.repeat(m[:, :, None], 3, axis=2)
        out = np.where(m3 > 0, fr, plate)
        cv2.imwrite(str(out_frames_dir / p.name), out)
        wrote += 1
    return wrote


def composite_frames_over_plate_objectmask(
    *,
    frames_dir: Path,
    plate_path: Path,
    out_frames_dir: Path,
    cam_meta_npz: Path,
    selected_means_per_frame_npy: Path,
    point_radius_px: int,
    dilate_px: int,
    blur_px: int,
    out_masks_dir: Path | None = None,
) -> int:
    """Composite using projected selected-object mask (ignores any artifacts outside object mask)."""
    import cv2

    plate = cv2.imread(str(plate_path))
    if plate is None:
        raise FileNotFoundError(plate_path)

    meta = np.load(cam_meta_npz)
    K = np.asarray(meta["K"], dtype=np.float64)
    W = np.asarray(meta["world_view_transform"], dtype=np.float64)

    means = np.load(selected_means_per_frame_npy).astype(np.float64)  # (T,N,3)
    T = int(means.shape[0])

    frames = _sorted_frame_pngs(frames_dir)
    if not frames:
        raise FileNotFoundError(f"no png frames under {frames_dir}")
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    if out_masks_dir is not None:
        out_masks_dir.mkdir(parents=True, exist_ok=True)

    pr = int(max(1, point_radius_px))
    dpx = int(max(0, dilate_px))
    bpx = int(max(0, blur_px))
    if bpx > 0 and bpx % 2 == 0:
        bpx += 1

    wrote = 0
    for i, p in enumerate(frames):
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        if fr.shape != plate.shape:
            fr = cv2.resize(fr, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Best-effort: assume frame index == i (0000.png..)
        if i >= T:
            Xi = means[-1]
        else:
            Xi = means[i]
        # Build object mask by projecting selected centers.
        mask = np.zeros((plate.shape[0], plate.shape[1]), dtype=np.uint8)
        from segmentation.mask_to_gaussians import project_world_to_pixels

        u, v, z = project_world_to_pixels(K, W, Xi)
        valid = z > 1e-6
        ui = np.floor(u[valid] + 0.5).astype(np.int32)
        vi = np.floor(v[valid] + 0.5).astype(np.int32)
        for uu, vv in zip(ui.tolist(), vi.tolist()):
            if 0 <= uu < mask.shape[1] and 0 <= vv < mask.shape[0]:
                cv2.circle(mask, (uu, vv), pr, 255, thickness=-1)
        if dpx > 0:
            yy, xx = np.ogrid[-dpx : dpx + 1, -dpx : dpx + 1]
            kernel = ((xx * xx + yy * yy) <= (dpx * dpx)).astype(np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        if bpx > 0:
            mask = cv2.GaussianBlur(mask, (bpx, bpx), 0)
        if out_masks_dir is not None:
            cv2.imwrite(str(out_masks_dir / p.name), mask)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        out = (alpha * fr.astype(np.float32) + (1.0 - alpha) * plate.astype(np.float32)).astype(np.uint8)
        cv2.imwrite(str(out_frames_dir / p.name), out)
        wrote += 1
    return wrote


def composite_frames_over_plate_objectmask_hard(
    *,
    frames_dir: Path,
    plate_path: Path,
    out_frames_dir: Path,
    cam_meta_npz: Path,
    selected_means_per_frame_npy: Path,
    original_chair_mask_path: Path,
    point_radius_px: int,
    dilate_px: int,
    blur_px: int,
    allowed_region_dilate_px: int,
    allowed_region_down_extend_px: int,
    allowed_region_up_exclude_px: int,
) -> int:
    """
    Hard compositing:
    - Base is always the static inpainted plate.
    - Simulated frame is ONLY allowed inside a strict chair alpha mask.
    - The chair alpha mask is projected per-frame from selected gaussians, then clamped to an
      allowed region derived from the original chair mask (+downward extension, -upper exclusion).
    """
    import cv2

    plate = cv2.imread(str(plate_path))
    if plate is None:
        raise FileNotFoundError(plate_path)

    h, w = int(plate.shape[0]), int(plate.shape[1])
    allowed = build_allowed_region_uint8(
        original_chair_mask_path=original_chair_mask_path,
        plate_h=h,
        plate_w=w,
        dilate_px=allowed_region_dilate_px,
        down_extend_px=allowed_region_down_extend_px,
        up_exclude_px=allowed_region_up_exclude_px,
    )

    meta = np.load(cam_meta_npz)
    K = np.asarray(meta["K"], dtype=np.float64)
    W = np.asarray(meta["world_view_transform"], dtype=np.float64)
    means = np.load(selected_means_per_frame_npy).astype(np.float64)  # (T,N,3)
    T = int(means.shape[0])

    frames = _sorted_frame_pngs(frames_dir)
    if not frames:
        raise FileNotFoundError(f"no png frames under {frames_dir}")
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    pr = int(max(1, point_radius_px))
    dpx = int(max(0, dilate_px))
    bpx = int(max(0, blur_px))
    if bpx > 0 and bpx % 2 == 0:
        bpx += 1

    from segmentation.mask_to_gaussians import project_world_to_pixels

    wrote = 0
    for i, p in enumerate(frames):
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        if fr.shape != plate.shape:
            fr = cv2.resize(fr, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)
        Xi = means[i] if i < T else means[-1]
        mask = np.zeros((plate.shape[0], plate.shape[1]), dtype=np.uint8)
        u, v, z = project_world_to_pixels(K, W, Xi)
        valid = z > 1e-6
        ui = np.floor(u[valid] + 0.5).astype(np.int32)
        vi = np.floor(v[valid] + 0.5).astype(np.int32)
        for uu, vv in zip(ui.tolist(), vi.tolist()):
            if 0 <= uu < mask.shape[1] and 0 <= vv < mask.shape[0]:
                cv2.circle(mask, (uu, vv), pr, 255, thickness=-1)
        if dpx > 0:
            yy, xx = np.ogrid[-dpx : dpx + 1, -dpx : dpx + 1]
            kernel = ((xx * xx + yy * yy) <= (dpx * dpx)).astype(np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        if bpx > 0:
            mask = cv2.GaussianBlur(mask, (bpx, bpx), 0)

        # HARD clamp: projected mask must be within allowed region.
        mask = cv2.bitwise_and(mask, allowed)

        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        out = (alpha * fr.astype(np.float32) + (1.0 - alpha) * plate.astype(np.float32)).astype(np.uint8)
        cv2.imwrite(str(out_frames_dir / p.name), out)
        wrote += 1
    return wrote


def composite_selected_only_matte_over_plate(
    *,
    plate_path: Path,
    chair_rgb_dir: Path,
    raw_mask_dir: Path,
    out_composite_dir: Path,
    out_alpha_dir: Path,
    out_foreground_dir: Path,
    original_chair_mask_path: Path,
    allowed_region_dilate_px: int,
    allowed_region_down_extend_px: int,
    allowed_region_up_exclude_px: int,
    alpha_close_kernel_px: int,
    alpha_feather_blur_px: int,
) -> int:
    """
    final = alpha * chair_rgb + (1 - alpha) * plate
    chair_rgb: normal selected-only PhysGaussian dumps (photoreal).
    alpha: selected-only render mask (soft threshold → close → hole-fill → slight dilate),
    then clamped to the allowed chair + downward-motion region (excludes upper window/curtain).
    """
    import cv2
    import shutil

    plate = cv2.imread(str(plate_path))
    if plate is None:
        raise FileNotFoundError(plate_path)
    h, w = int(plate.shape[0]), int(plate.shape[1])
    allowed = build_allowed_region_uint8(
        original_chair_mask_path=original_chair_mask_path,
        plate_h=h,
        plate_w=w,
        dilate_px=allowed_region_dilate_px,
        down_extend_px=allowed_region_down_extend_px,
        up_exclude_px=allowed_region_up_exclude_px,
    )

    chair_frames = _sorted_frame_pngs(chair_rgb_dir)
    if not chair_frames:
        raise FileNotFoundError(f"no chair RGB png under {chair_rgb_dir}")

    out_composite_dir.mkdir(parents=True, exist_ok=True)
    out_alpha_dir.mkdir(parents=True, exist_ok=True)
    out_foreground_dir.mkdir(parents=True, exist_ok=True)

    k_close = max(3, int(alpha_close_kernel_px))
    if k_close % 2 == 0:
        k_close += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    bfeather = int(max(0, alpha_feather_blur_px))
    if bfeather > 0 and bfeather % 2 == 0:
        bfeather += 1

    ref_bin = _read_mask_u8(original_chair_mask_path)
    if ref_bin.shape[:2] != (h, w):
        ref_bin = cv2.resize(ref_bin, (w, h), interpolation=cv2.INTER_NEAREST)
    ref_bin = cv2.bitwise_and(ref_bin, allowed)

    wrote = 0
    for p in chair_frames:
        mpath = raw_mask_dir / p.name
        if not mpath.is_file():
            continue
        chair = cv2.imread(str(p))
        raw = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if chair is None or raw is None:
            continue
        if chair.shape[:2] != (h, w):
            chair = cv2.resize(chair, (w, h), interpolation=cv2.INTER_LINEAR)
        if raw.shape[:2] != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)

        raw_f = raw.astype(np.float32) / 255.0
        # Soft coverage seed (avoid harsh 127 binarize that punches holes in the chair).
        seed = (raw_f > 0.004).astype(np.uint8) * 255
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k_open, iterations=1)
        m_closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        filled = ndimage.binary_fill_holes(m_closed > 127)
        m_fill = (filled.astype(np.uint8) * 255)
        m_fill = cv2.dilate(m_fill, k_bridge, iterations=1)
        # Fill interior holes using the SAM chair mask only near current splat coverage (no full-frame OR).
        band = cv2.dilate(m_fill, close_kernel, iterations=2)
        interior_fill = cv2.bitwise_and(ref_bin, band)
        m_union = cv2.bitwise_or(m_fill, interior_fill)
        m_restricted = cv2.bitwise_and(m_union, allowed)

        if bfeather > 0:
            a_f = cv2.GaussianBlur(
                m_restricted.astype(np.float32), (bfeather, bfeather), 0
            ) / 255.0
        else:
            a_f = m_restricted.astype(np.float32) / 255.0
        a_f = np.clip(a_f, 0.0, 1.0)
        a3 = a_f[:, :, None]

        out = (a3 * chair.astype(np.float32) + (1.0 - a3) * plate.astype(np.float32)).astype(
            np.uint8
        )
        cv2.imwrite(str(out_composite_dir / p.name), out)
        cv2.imwrite(
            str(out_alpha_dir / p.name),
            np.clip(a_f * 255.0, 0, 255).astype(np.uint8),
        )
        shutil.copy2(str(p), str(out_foreground_dir / p.name))
        wrote += 1
    return wrote


def _odd_kernel_size(radius_px: int) -> int:
    k = max(3, int(radius_px) * 2 + 1)
    if k % 2 == 0:
        k += 1
    return k


def fullframe_plate_cleanup_encode_mp4(
    *,
    frames_dir: Path,
    plate_path: Path,
    original_chair_mask_path: Path,
    out_mp4: Path,
    fps: int,
    allowed_region_dilate_px: int,
    allowed_region_down_extend_px: int,
    allowed_region_up_exclude_px: int,
    dark_thresh_u8: int = 42,
    protect_erode_px: int = 5,
    near_chair_dilate_px: int = 64,
    feather_px: int = 5,
) -> int:
    """Keep full simulation RGB; paste inpainted plate only where cleanup mask is on.

    Cleanup = (outside allowed geometry, e.g. upper window) OR (dark pixels outside
    eroded chair core but near dilated chair mask). Encodes a single mp4; no PNG dirs left behind.
    """
    import cv2
    import tempfile

    plate = cv2.imread(str(plate_path))
    if plate is None:
        raise FileNotFoundError(plate_path)
    h, w = int(plate.shape[0]), int(plate.shape[1])

    sam = _read_mask_u8(original_chair_mask_path)
    if sam.shape[:2] != (h, w):
        sam = cv2.resize(sam, (w, h), interpolation=cv2.INTER_NEAREST)

    allowed = build_allowed_region_uint8(
        original_chair_mask_path=original_chair_mask_path,
        plate_h=h,
        plate_w=w,
        dilate_px=allowed_region_dilate_px,
        down_extend_px=allowed_region_down_extend_px,
        up_exclude_px=allowed_region_up_exclude_px,
    )

    paths = _sorted_frame_pngs(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No numbered PNG frames under {frames_dir}")

    k_core = _odd_kernel_size(protect_erode_px)
    k_near = _odd_kernel_size(near_chair_dilate_px)
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core))
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_near, k_near))
    core = cv2.erode(sam, core_kernel, iterations=1)
    near_chair = cv2.dilate(sam, near_kernel, iterations=1)

    thr = int(np.clip(int(dark_thresh_u8), 0, 255))

    k_feather = _odd_kernel_size(feather_px) if feather_px > 0 else 0

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tg3d_fullframe_cleanup_") as td:
        tmp = Path(td)
        wrote = 0
        for fp in paths:
            sim = cv2.imread(str(fp))
            if sim is None:
                raise FileNotFoundError(fp)
            if sim.shape[:2] != (h, w):
                sim = cv2.resize(sim, (w, h), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY)

            outside_core = core < 127
            dark = gray < thr
            near_ok = near_chair > 127
            c_dark_halo = outside_core & near_ok & dark
            c_geom = allowed == 0
            cleanup = (c_geom | c_dark_halo).astype(np.uint8) * 255

            if k_feather > 0:
                alpha_u8 = cv2.GaussianBlur(cleanup.astype(np.float32), (k_feather, k_feather), 0)
                alpha_u8 = np.clip(alpha_u8, 0.0, 255.0).astype(np.uint8)
            else:
                alpha_u8 = cleanup
            a = alpha_u8.astype(np.float32) / 255.0
            a3 = a[:, :, None]
            out = (a3 * plate.astype(np.float32) + (1.0 - a3) * sim.astype(np.float32)).astype(
                np.uint8
            )
            cv2.imwrite(str(tmp / fp.name), out)
            wrote += 1

        compile_video_ffmpeg(frames_dir=tmp, out_mp4=out_mp4, fps=int(fps))

    return wrote


def compile_video_ffmpeg(
    *,
    frames_dir: Path,
    out_mp4: Path,
    fps: int,
) -> None:
    import subprocess
    import tempfile

    paths = _sorted_frame_pngs(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No PNG frames under {frames_dir}")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    scale_vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]

    ids = [int(p.stem) for p in paths]
    start = ids[0]
    contiguous = ids == list(range(start, start + len(ids)))

    if contiguous:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(int(fps)),
                "-start_number",
                str(start),
                "-i",
                str(frames_dir / "%04d.png"),
                "-frames:v",
                str(len(paths)),
                "-vf",
                scale_vf,
                *enc,
                str(out_mp4),
            ],
            check=True,
        )
        return

    frame_dur = 1.0 / float(max(1, int(fps)))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for p in paths:
            ap = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{ap}'\n")
            f.write(f"duration {frame_dur:.9f}\n")
        if paths:
            ap = str(paths[-1].resolve()).replace("'", "'\\''")
            f.write(f"file '{ap}'\n")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-vf",
                scale_vf,
                *enc,
                str(out_mp4),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)

