"""Composite a triangle mesh over a base-scene 3DGS render (depth-aware).

Uses the same ``full_proj_transform`` convention as ``gaussian_renderer.render`` and
``nvdiffrast`` for the mesh (DreamGaussian-style clip space).
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from config import PROJECT_ROOT


def _ensure_gs_path() -> Path:
    gs = PROJECT_ROOT / "submodules" / "gaussian-splatting"
    if not gs.is_dir():
        raise FileNotFoundError(gs)
    p = str(gs.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    return gs


@dataclass
class GSMeshBackdrop:
    """Cached 3DGS scene + camera + raster context for repeated mesh composites."""

    gaussians: object
    view: object
    pipe: object
    background: torch.Tensor
    dataset: object
    separate_sh: bool
    glctx: object
    train_views: list | None = None


def build_gs_mesh_backdrop(
    *,
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    base_ply: Path | str,
    camera_index: int = 0,
) -> GSMeshBackdrop:
    """Load base-scene Gaussians and training camera ``camera_index`` once."""
    try:
        import nvdiffrast.torch as dr
    except ImportError as e:
        raise RuntimeError(
            "Mesh composite requires nvdiffrast (same as DreamGaussian). "
            "Install torch first, then nvdiffrast per project README."
        ) from e

    _ensure_gs_path()
    from argparse import ArgumentParser

    from arguments import ModelParams, PipelineParams, get_combined_args  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel  # type: ignore

    from reconstruction.render_views import _resolve_source_path

    colmap_scene = Path(colmap_scene).resolve()
    model_path = Path(model_path).resolve()
    base_ply = Path(base_ply)
    colmap_resolved, images = _resolve_source_path(colmap_scene, model_path)

    old_argv = sys.argv
    try:
        sys.argv = [
            "render",
            "-m",
            str(model_path),
            "-s",
            str(colmap_resolved),
            "--iteration",
            str(iteration),
            "--skip_test",
            "--quiet",
        ]
        if images:
            sys.argv.extend(["--images", images])
        parser = ArgumentParser()
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        args = get_combined_args(parser)
        dataset = model.extract(args)
        pipe = pipeline.extract(args)
    finally:
        sys.argv = old_argv

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # type: ignore

        separate_sh = True
    except ImportError:
        separate_sh = False

    print(
        "[3DGS backdrop] Building Scene (loads all training cameras + resizes images; "
        "often 1–5+ minutes with no intermediate logs from gaussian-splatting)...",
        flush=True,
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    print("[3DGS backdrop] Loading PLY weights into GaussianModel...", flush=True)
    gaussians.load_ply(str(base_ply))
    print("[3DGS backdrop] Scene + PLY ready.", flush=True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras(1.0)
    if camera_index < 0 or camera_index >= len(views):
        raise IndexError(f"camera_index={camera_index} out of range (n={len(views)})")
    view = views[camera_index]
    glctx = dr.RasterizeCudaContext()
    return GSMeshBackdrop(
        gaussians=gaussians,
        view=view,
        pipe=pipe,
        background=background,
        dataset=dataset,
        separate_sh=separate_sh,
        glctx=glctx,
        train_views=views,
    )


def composite_mesh_with_backdrop(
    backdrop: GSMeshBackdrop,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    mesh_vertex_rgb01: np.ndarray | None = None,
    depth_bias: float = 0.02,
) -> np.ndarray:
    """Single RGB frame (uint8 HxW3) using a pre-built :class:`GSMeshBackdrop`."""
    import nvdiffrast.torch as dr

    from gaussian_renderer import render as gs_render  # type: ignore

    view = backdrop.view
    with torch.no_grad():
        pkg = gs_render(
            view,
            backdrop.gaussians,
            backdrop.pipe,
            backdrop.background,
            use_trained_exp=backdrop.dataset.train_test_exp,
            separate_sh=backdrop.separate_sh,
        )
        gs_rgb = pkg["render"].clamp(0.0, 1.0)
        gs_inv_depth = pkg["depth"]
        H, W = int(view.image_height), int(view.image_width)

        inv = gs_inv_depth.squeeze()
        while inv.dim() > 2:
            inv = inv.squeeze(0)
        z_gs = torch.full((H, W), 1e6, dtype=torch.float32, device="cuda")
        ok = inv > 1e-6
        z_gs[ok] = 1.0 / inv[ok].clamp(min=1e-6)

        V = torch.as_tensor(mesh_vertices, dtype=torch.float32, device="cuda")
        F = torch.as_tensor(mesh_faces, dtype=torch.int32, device="cuda")
        if mesh_vertex_rgb01 is None:
            vc = torch.tensor([[0.98, 0.88, 0.15]], dtype=torch.float32, device="cuda").expand(
                V.shape[0], -1
            )
        else:
            vc = torch.as_tensor(mesh_vertex_rgb01, dtype=torch.float32, device="cuda")
            if vc.shape != V.shape:
                raise ValueError("mesh_vertex_rgb01 must be (N,3) float in [0,1]")

        v_h = torch.cat([V, torch.ones(V.shape[0], 1, device="cuda")], dim=1)
        v_clip = torch.matmul(v_h, view.full_proj_transform)
        w2c = view.world_view_transform
        v_cam = torch.matmul(v_h, w2c.T)
        vz = (v_cam[:, 2:3] / v_cam[:, 3:4].clamp(min=1e-6)).contiguous()

        rast, rast_db = dr.rasterize(backdrop.glctx, v_clip.unsqueeze(0), F, (H, W))
        alpha = (rast[..., 3:4] > 0).float()
        mesh_z, _ = dr.interpolate(
            vz.unsqueeze(0), rast, F, rast_db=rast_db, diff_attrs="all"
        )
        mesh_z = mesh_z.squeeze(0).squeeze(-1)

        rgb_i, _ = dr.interpolate(
            vc.unsqueeze(0).contiguous(), rast, F, rast_db=rast_db, diff_attrs="all"
        )
        rgb_i = dr.antialias(rgb_i, rast, v_clip.unsqueeze(0), F).squeeze(0).clamp(0.0, 1.0)

        a = alpha.squeeze(0).squeeze(-1)
        mz = mesh_z.squeeze()
        no_gs = z_gs >= 1e5 - 1.0
        closer = no_gs | ((mz + depth_bias) < z_gs)
        m = (a > 0.5) & closer
        m4 = m.unsqueeze(-1).float()

        out = gs_rgb.permute(1, 2, 0) * (1.0 - m4) + rgb_i * m4
        out_u8 = (out.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
        out_u8 = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
        out_u8 = cv2.cvtColor(out_u8, cv2.COLOR_BGR2RGB)
        return out_u8


def composite_mesh_over_base_gs(
    *,
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    base_ply: Path | str,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    mesh_vertex_rgb01: np.ndarray | None = None,
    camera_index: int = 0,
    depth_bias: float = 0.02,
) -> np.ndarray:
    """Render ``base_ply`` with 3DGS, draw ``mesh_*`` on top where closer in *metric* depth."""
    bd = build_gs_mesh_backdrop(
        colmap_scene=colmap_scene,
        model_path=model_path,
        iteration=iteration,
        base_ply=base_ply,
        camera_index=camera_index,
    )
    return composite_mesh_with_backdrop(
        bd, mesh_vertices, mesh_faces, mesh_vertex_rgb01=mesh_vertex_rgb01, depth_bias=depth_bias
    )


def write_png(path: Path | str, rgb_u8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def sorted_centroid_npy_paths(track_dir: Path) -> list[Path]:
    out: list[tuple[int, Path]] = []
    for p in track_dir.glob("centroid_*.npy"):
        try:
            n = int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((n, p))
    return [p for _, p in sorted(out, key=lambda t: t[0])]


def sorted_gauss_npy_paths(gauss_dir: Path) -> list[Path]:
    out: list[tuple[int, Path]] = []
    for p in gauss_dir.glob("gauss_*.npy"):
        try:
            n = int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((n, p))
    return [p for _, p in sorted(out, key=lambda t: t[0])]


def run_mesh_rigid_overlay_sequence(
    *,
    sim_run: Path | str,
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    base_ply: Path | str,
    mesh_vertices_world_rest: np.ndarray,
    mesh_faces: np.ndarray,
    frame_dt: float,
    playback_seconds: float | None = None,
    camera_index: int = 0,
    mesh_vertex_rgb01: np.ndarray | None = None,
    depth_bias: float = 0.02,
    mesh_skin_k: int = 12,
) -> Path:
    """Overwrite ``sim_run/frames/*.png`` with base-GS + mesh, then re-encode ``videos/output.mp4``.

    If ``sim_run/mesh_gauss_xyz/gauss_XXXX.npy`` exists (PhysGaussian ``--save_obj_gauss_xyz``), each mesh
    vertex follows a **KNN soft skinning** of object Gaussians so **jelly deformation** from the MPM sim
    transfers to the triangle mesh (best-effort; not exact elastica).

    Otherwise falls back to **centroid-only rigid** motion using ``mesh_centroids/centroid_XXXX.npy``.
    """
    from rendering.mesh_gaussian_skin import (
        deform_mesh_vertices,
        load_skinning_npz,
        save_skinning_npz,
        skinning_cache_valid,
        skinning_indices_weights,
    )

    sim_run = Path(sim_run)
    frames_dir = sim_run / "frames"
    gauss_dir = sim_run / "mesh_gauss_xyz"
    gauss_paths = sorted_gauss_npy_paths(gauss_dir) if gauss_dir.is_dir() else []

    bd = build_gs_mesh_backdrop(
        colmap_scene=colmap_scene,
        model_path=model_path,
        iteration=iteration,
        base_ply=base_ply,
        camera_index=camera_index,
    )

    if gauss_paths:
        gauss0 = np.load(gauss_paths[0]).astype(np.float64)
        skin_path = sim_run / "mesh_skin.npz"
        k_req = int(mesh_skin_k)
        nv = int(mesh_vertices_world_rest.shape[0])
        ng = int(gauss0.shape[0])
        idx: np.ndarray
        w: np.ndarray
        if skin_path.is_file():
            idx, w = load_skinning_npz(skin_path)
            if not skinning_cache_valid(
                idx, w, n_gauss=ng, n_vertices=nv, k_request=k_req
            ):
                idx, w = skinning_indices_weights(
                    mesh_vertices_world_rest, gauss0, k=k_req
                )
                save_skinning_npz(skin_path, idx, w, k_req)
        else:
            idx, w = skinning_indices_weights(mesh_vertices_world_rest, gauss0, k=k_req)
            save_skinning_npz(skin_path, idx, w, k_req)
        for gp in gauss_paths:
            frame = int(gp.stem.split("_", 1)[1])
            gausst = np.load(gp).astype(np.float64)
            if gausst.shape != gauss0.shape:
                raise ValueError(
                    f"gauss frame shape {gausst.shape} != bind pose {gauss0.shape} for {gp}"
                )
            Vt = deform_mesh_vertices(mesh_vertices_world_rest, gauss0, gausst, idx, w).astype(
                np.float64
            )
            img = composite_mesh_with_backdrop(
                bd, Vt, mesh_faces, mesh_vertex_rgb01=mesh_vertex_rgb01, depth_bias=depth_bias
            )
            write_png(frames_dir / f"{frame:04d}.png", img)
        return reencode_sim_run_video(sim_run, frame_dt=frame_dt, playback_seconds=playback_seconds)

    track_dir = sim_run / "mesh_centroids"
    paths = sorted_centroid_npy_paths(track_dir)
    if not paths:
        raise FileNotFoundError(
            f"No mesh_gauss_xyz/gauss_*.npy under {gauss_dir} and no centroid_*.npy under {track_dir}. "
            "Re-run simulation with --mesh-render so PhysGaussian saves tracking data."
        )
    c0 = np.load(paths[0]).astype(np.float64)
    for fp in paths:
        frame = int(fp.stem.split("_", 1)[1])
        ct = np.load(fp).astype(np.float64)
        delta = ct - c0
        Vt = mesh_vertices_world_rest + delta.reshape(1, 3)
        img = composite_mesh_with_backdrop(
            bd, Vt, mesh_faces, mesh_vertex_rgb01=mesh_vertex_rgb01, depth_bias=depth_bias
        )
        write_png(frames_dir / f"{frame:04d}.png", img)

    return reencode_sim_run_video(sim_run, frame_dt=frame_dt, playback_seconds=playback_seconds)


def reencode_sim_run_video(
    sim_run: Path | str,
    *,
    frame_dt: float,
    playback_seconds: float | None = None,
) -> Path:
    """Rebuild ``videos/output.mp4`` from ``frames/%04d.png`` (backs up existing mp4)."""
    from datetime import datetime

    sim_run = Path(sim_run)
    frames_dir = sim_run / "frames"
    video_dir = sim_run / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames under {frames_dir}")
    if cv2.imread(str(frames[0])) is None:
        raise FileNotFoundError(frames[0])

    primary = video_dir / "output.mp4"
    if primary.is_file():
        stamp = datetime.now().strftime("%m%d_%H%M%S")
        backup = primary.with_name(f"output_{stamp}.mp4")
        n = 1
        while backup.is_file():
            backup = primary.with_name(f"output_{stamp}_{n}.mp4")
            n += 1
        primary.rename(backup)

    if playback_seconds:
        fps = max(1.0, len(frames) / float(playback_seconds))
    else:
        fps = max(1.0, 1.0 / float(frame_dt))

    subprocess.run(
        [
            "ffmpeg",
            "-y",
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
            str(primary),
        ],
        check=True,
    )
    return primary
