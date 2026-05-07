"""Render merged 3DGS with kinematic (non-MPM) in-place wobble for object Gaussians."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from physics.kinematic_wobble import object_displacement_kinematic
from physics.mpm_runner import _backup_existing_video
from rendering.gs_mesh_composite import build_gs_mesh_backdrop


def _append_wobble_motion_log(log_path: Path, frame: int, D: np.ndarray) -> None:
    norms = np.linalg.norm(D, axis=1)
    com_d = D.mean(axis=0)
    disp_com = float(np.linalg.norm(com_d))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"frame={frame}\tmax|D|={float(norms.max()):.8f}\tmean|D|={float(norms.mean()):.8f}\t"
            f"COM_dx={com_d[0]:.8f}\tCOM_dy={com_d[1]:.8f}\tCOM_dz={com_d[2]:.8f}\t|mean(D)|={disp_com:.8f}\n"
        )


def _print_wobble_frame_stats(
    frame: int,
    D: np.ndarray,
    xyz_obj_before: np.ndarray,
    xyz_obj_after: np.ndarray,
) -> None:
    norms = np.linalg.norm(D, axis=1)
    com_d = D.mean(axis=0)
    disp_com = float(np.linalg.norm(com_d))
    print(
        f"[wobble] frame={frame:4d}  |D| max={float(norms.max()):.6f} mean={float(norms.mean()):.6f} "
        f"min={float(norms.min()):.6f}  mean(D)_xyz={com_d}  |mean(D)|={disp_com:.9f}",
        flush=True,
    )
    bmin = xyz_obj_before.min(axis=0)
    bmax = xyz_obj_before.max(axis=0)
    amin = xyz_obj_after.min(axis=0)
    amax = xyz_obj_after.max(axis=0)
    print(
        f"         obj xyz before wobble min={bmin} max={bmax}",
        flush=True,
    )
    print(
        f"         obj xyz after  wobble min={amin} max={amax}",
        flush=True,
    )


def render_kinematic_wobble_sequence(
    *,
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    merged_ply: Path | str,
    n_base: int,
    n_obj: int,
    sim_run: Path | str,
    frame_num: int,
    frame_dt: float,
    playback_seconds: float,
    camera_index: int = 0,
    wobble_amp: float = 0.012,
    wobble_frequency: float = 1.15,
    wobble_height_weight: float = 1.75,
    wobble_bottom_pin: float = 0.24,
    track_debug: bool = False,
) -> str:
    """Write ``sim_run/frames/*.png`` and ``sim_run/video/output.mp4`` (same layout as PhysGaussian)."""
    if not torch.cuda.is_available():
        raise RuntimeError("Kinematic wobble rendering requires CUDA.")

    merged_ply = Path(merged_ply).resolve()
    sim_run = Path(sim_run).resolve()
    frames_dir = sim_run / "frames"
    video_dir = sim_run / "video"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _backup_existing_video(video_dir)

    dbg_dir = sim_run / "mesh_gauss_xyz"
    diag_xyz_dir = sim_run / "wobble_kinematic_diag"
    if track_debug and dbg_dir.exists():
        shutil.rmtree(dbg_dir)
    if track_debug and diag_xyz_dir.exists():
        shutil.rmtree(diag_xyz_dir)
    if track_debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)
        diag_xyz_dir.mkdir(parents=True, exist_ok=True)

    backdrop = build_gs_mesh_backdrop(
        colmap_scene=colmap_scene,
        model_path=model_path,
        iteration=iteration,
        base_ply=merged_ply,
        camera_index=camera_index,
    )

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # type: ignore

        separate_sh = True
    except ImportError:
        separate_sh = False

    from gaussian_renderer import render as gs_render  # type: ignore

    view = backdrop.view
    gaussians = backdrop.gaussians
    # ``GaussianModel.get_xyz`` / ``get_scaling`` / ``get_opacity`` are @property, not methods.
    with torch.no_grad():
        xyz_rest = gaussians.get_xyz.detach().clone()
    n_total = int(xyz_rest.shape[0])
    if n_base + n_obj > n_total or n_obj < 0 or n_base < 0:
        raise ValueError(f"Bad n_base={n_base} n_obj={n_obj} for merged ply N={n_total}")

    xyz_obj_rest = xyz_rest[n_base : n_base + n_obj].detach().cpu().numpy()
    y_np = xyz_obj_rest[:, 1]
    xyz_obj_np = xyz_obj_rest.astype(np.float64, copy=False)

    print(
        f"[wobble] merged_ply={merged_ply}  n_total={n_total}  n_base={n_base}  n_obj={n_obj}  "
        f"object_indices=[{n_base}:{n_base + n_obj}]",
        flush=True,
    )
    print(
        f"[wobble] object slice xyz (rest) min={xyz_obj_rest.min(axis=0)} max={xyz_obj_rest.max(axis=0)}",
        flush=True,
    )

    stat_frames = {0, 25, 50}

    for frame in tqdm(
        range(int(frame_num)),
        desc="Kinematic wobble frames",
        unit="fr",
        leave=True,
    ):
        D = object_displacement_kinematic(
            y_np,
            frame_index=frame,
            frame_dt=float(frame_dt),
            amp=float(wobble_amp),
            freq_hz=float(wobble_frequency),
            height_gamma=float(wobble_height_weight),
            bottom_pin=float(wobble_bottom_pin),
            xyz_world=xyz_obj_np,
        )
        d_t = torch.as_tensor(D, device=xyz_rest.device, dtype=xyz_rest.dtype)
        new_xyz = xyz_rest.clone()
        new_xyz[n_base : n_base + n_obj] += d_t
        with torch.no_grad():
            gaussians._xyz.data.copy_(new_xyz)
            if track_debug and frame == 0:
                chk = gaussians.get_xyz[n_base : n_base + n_obj]
                max_err = float(
                    (chk - new_xyz[n_base : n_base + n_obj]).abs().max().item()
                )
                print(
                    f"[wobble] post-copy_ max|get_xyz - new_xyz| on object slice = {max_err:.2e}",
                    flush=True,
                )

        if frame in stat_frames:
            after_np = (
                new_xyz[n_base : n_base + n_obj].detach().cpu().numpy().astype(np.float64)
            )
            _print_wobble_frame_stats(frame, D, xyz_obj_rest, after_np)

        with torch.no_grad():
            pkg = gs_render(
                view,
                gaussians,
                backdrop.pipe,
                backdrop.background,
                use_trained_exp=backdrop.dataset.train_test_exp,
                separate_sh=separate_sh,
            )
        image = pkg["render"].clamp(0.0, 1.0)
        if backdrop.dataset.train_test_exp:
            image = image[..., image.shape[-1] // 2 :]
        torchvision.utils.save_image(image, str(frames_dir / f"{frame:04d}.png"))

        if track_debug:
            obj_xyz = (
                new_xyz[n_base : n_base + n_obj].detach().cpu().numpy().astype(np.float32)
            )
            np.save(dbg_dir / f"gauss_{frame:04d}.npy", obj_xyz)
            if frame in (0, 50, 100):
                out_npy = diag_xyz_dir / f"frame_{frame:03d}_object_xyz.npy"
                np.save(out_npy, obj_xyz.astype(np.float32, copy=False))

    primary = video_dir / "output.mp4"
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames written under {frames_dir}")
    fps = max(1.0, len(frames) / float(playback_seconds))
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(frames_dir / "%04d.png"),
            "-vf",
            "scale='min(1280,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-profile:v",
            "main",
            "-level:v",
            "4.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(primary),
        ],
        check=True,
    )
    if not primary.is_file():
        raise FileNotFoundError(f"ffmpeg did not create {primary}")
    return str(primary)


def render_kinematic_wobble_scene_by_indices(
    *,
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    scene_ply: Path | str,
    object_indices: np.ndarray,
    sim_run: Path | str,
    frame_num: int,
    frame_dt: float,
    playback_seconds: float,
    camera_index: int = 0,
    wobble_amp: float = 0.008,
    wobble_frequency: float = 1.0,
    wobble_height_weight: float = 1.5,
    wobble_bottom_pin: float = 0.30,
    track_debug: bool = False,
    indices_source: str = "",
) -> str:
    """Kinematic wobble on a **subset** of Gaussians in a single scene PLY (mode-b)."""
    if not torch.cuda.is_available():
        raise RuntimeError("Kinematic wobble rendering requires CUDA.")

    scene_ply = Path(scene_ply).resolve()
    sim_run = Path(sim_run).resolve()
    frames_dir = sim_run / "frames"
    video_dir = sim_run / "video"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _backup_existing_video(video_dir)

    motion_log = Path(sim_run) / "kinematic_wobble_motion_log.txt"
    if motion_log.is_file():
        motion_log.unlink()

    print(
        "[mode-b kinematic] Loading 3DGS Scene for wobble render (again loads all training cameras if not cached)...",
        flush=True,
    )
    backdrop = build_gs_mesh_backdrop(
        colmap_scene=colmap_scene,
        model_path=model_path,
        iteration=iteration,
        base_ply=scene_ply,
        camera_index=camera_index,
    )
    print("[mode-b kinematic] Scene ready; rendering frame loop with tqdm...", flush=True)
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # type: ignore

        separate_sh = True
    except ImportError:
        separate_sh = False
    from gaussian_renderer import render as gs_render  # type: ignore

    view = backdrop.view
    _iname = getattr(view, "image_name", "?")
    print(
        f"[mode-b kinematic] active training view: camera_index={camera_index}  image_name={_iname}  "
        f"resolution={int(view.image_width)}x{int(view.image_height)}",
        flush=True,
    )
    gaussians = backdrop.gaussians
    with torch.no_grad():
        xyz_rest = gaussians.get_xyz.detach().clone()
    idx_t = torch.as_tensor(
        np.asarray(object_indices, dtype=np.int64).ravel(),
        device=xyz_rest.device,
        dtype=torch.long,
    )
    idx_t = idx_t[(idx_t >= 0) & (idx_t < xyz_rest.shape[0])]
    if idx_t.numel() == 0:
        raise ValueError("render_kinematic_wobble_scene_by_indices: empty object_indices")

    xyz_obj_rest = xyz_rest[idx_t].detach().cpu().numpy()
    y_np = xyz_obj_rest[:, 1]
    xyz_obj_np = xyz_obj_rest.astype(np.float64, copy=False)
    n_obj = int(idx_t.numel())
    print(
        f"[mode-b kinematic] scene_ply={scene_ply}  n_total={int(xyz_rest.shape[0])}  "
        f"n_selected={n_obj}  camera_index={camera_index}",
        flush=True,
    )
    if indices_source:
        print(f"[mode-b kinematic] indices_source={indices_source}", flush=True)
    print(
        f"[mode-b kinematic] wobble_amp={wobble_amp} m  freq_hz={wobble_frequency}  "
        f"height_gamma={wobble_height_weight}  bottom_pin={wobble_bottom_pin}",
        flush=True,
    )
    print(
        f"[mode-b kinematic] selected xyz (rest) min={xyz_obj_rest.min(axis=0)} max={xyz_obj_rest.max(axis=0)}",
        flush=True,
    )

    motion_log.write_text(
        "mode-b kinematic wobble motion log\n"
        f"indices_source={indices_source or '(caller did not pass)'}\n"
        f"n_selected={n_obj}  camera_index={camera_index}\n"
        f"wobble_amp_m={wobble_amp}  frame_dt={frame_dt}  frame_num={frame_num}\n"
        "columns: frame max|D| mean|D| COM_dxyz |mean(D)|\n\n",
        encoding="utf-8",
    )

    fn = int(frame_num)
    stat_frames = sorted({f for f in (0, 25, 50, 100, fn // 4, fn // 2) if 0 <= f < fn})
    try:
        for frame in tqdm(
            range(int(frame_num)),
            desc="Mode-b kinematic frames",
            unit="fr",
            leave=True,
        ):
            D = object_displacement_kinematic(
                y_np,
                frame_index=frame,
                frame_dt=float(frame_dt),
                amp=float(wobble_amp),
                freq_hz=float(wobble_frequency),
                height_gamma=float(wobble_height_weight),
                bottom_pin=float(wobble_bottom_pin),
                xyz_world=xyz_obj_np,
            )
            d_t = torch.as_tensor(D, device=xyz_rest.device, dtype=xyz_rest.dtype)
            new_xyz = xyz_rest.clone()
            new_xyz[idx_t] += d_t
            with torch.no_grad():
                gaussians._xyz.data.copy_(new_xyz)

            if frame == 0:
                chk = gaussians.get_xyz[idx_t]
                max_err = float((chk - new_xyz[idx_t]).abs().max().item())
                print(
                    f"[mode-b kinematic] frame0 verify max|get_xyz[idx]-new_xyz[idx]|={max_err:.2e} "
                    f"(should be ~0; else renderer not reading _xyz)",
                    flush=True,
                )

            if frame in stat_frames:
                after_np = new_xyz[idx_t].detach().cpu().numpy().astype(np.float64)
                _append_wobble_motion_log(motion_log, frame, D)
                print(
                    f"[mode-b kinematic] frame={frame}  |D| max={float(np.linalg.norm(D, axis=1).max()):.6f} "
                    f"mean={float(np.linalg.norm(D, axis=1).mean()):.6f}  |COM(D)|={float(np.linalg.norm(D.mean(axis=0))):.6f}",
                    flush=True,
                )
                if track_debug:
                    _print_wobble_frame_stats(frame, D, xyz_obj_rest, after_np)

            with torch.no_grad():
                pkg = gs_render(
                    view,
                    gaussians,
                    backdrop.pipe,
                    backdrop.background,
                    use_trained_exp=backdrop.dataset.train_test_exp,
                    separate_sh=separate_sh,
                )
            image = pkg["render"].clamp(0.0, 1.0)
            if backdrop.dataset.train_test_exp:
                image = image[..., image.shape[-1] // 2 :]
            torchvision.utils.save_image(image, str(frames_dir / f"{frame:04d}.png"))
    finally:
        with torch.no_grad():
            gaussians._xyz.data.copy_(xyz_rest)
    print(
        f"[mode-b kinematic] restored Gaussian xyz to rest state; motion log -> {motion_log}",
        flush=True,
    )

    primary = video_dir / "output.mp4"
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames written under {frames_dir}")
    fps = max(1.0, len(frames) / float(playback_seconds))
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(frames_dir / "%04d.png"),
            "-vf",
            "scale='min(1280,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-profile:v",
            "main",
            "-level:v",
            "4.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(primary),
        ],
        check=True,
    )
    if not primary.is_file():
        raise FileNotFoundError(f"ffmpeg did not create {primary}")
    return str(primary)
