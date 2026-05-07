"""Mode-b selection debug: strong tint, opacity, multi-camera, diff maps, visibility counts."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision

from rendering.gs_mesh_composite import build_gs_mesh_backdrop
from rendering.mode_b_visibility import count_selected_visible_in_view


def _ensure_gs_utils():
    from config import PROJECT_ROOT
    import sys

    gs = PROJECT_ROOT / "submodules" / "gaussian-splatting"
    p = str(gs.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    from utils.general_utils import inverse_sigmoid  # type: ignore
    from utils.sh_utils import RGB2SH  # type: ignore

    return RGB2SH, inverse_sigmoid


def _maybe_train_exp_crop(image: torch.Tensor, train_test_exp: bool) -> torch.Tensor:
    if train_test_exp:
        return image[..., image.shape[-1] // 2 :]
    return image


def _render_one(
    view: object,
    gaussians: object,
    pipe: object,
    background: torch.Tensor,
    train_test_exp: bool,
    separate_sh: bool,
) -> torch.Tensor:
    from gaussian_renderer import render as gs_render  # type: ignore

    with torch.no_grad():
        pkg = gs_render(
            view,
            gaussians,
            pipe,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
        )
    return _maybe_train_exp_crop(pkg["render"].clamp(0.0, 1.0), train_test_exp)


def _save_diff(baseline: torch.Tensor, other: torch.Tensor, out_path: Path) -> None:
    d = (other - baseline).abs().max(dim=0, keepdim=True)[0]
    d = (d * 4.0).clamp(0.0, 1.0).repeat(3, 1, 1)
    torchvision.utils.save_image(d, str(out_path))


def run_mode_b_selection_debug_suite(
    *,
    colmap_scene: Path | str,
    model_path: Path | str,
    iteration: int,
    scene_ply: Path | str,
    selected_indices: np.ndarray,
    dbg_dir: Path,
    forced_overlay_camera_index: int | None = None,
    package_dir: Path | str | None = None,
) -> dict[str, object]:
    """One Scene load; visibility over all training cams; strong overlays for view0 + extras + best.

    Writes ``frame00_*_view0.png``, ``frame00_*_best_c{k}.png``, ``frame00_*_primary.png`` (canonical
    overlay camera = ``forced_overlay_camera_index`` if valid, else visibility ``best_id``),
    copies ``frame00_selected_overlay_diff.png`` from that primary overlay,
    ``best_camera_index.txt``, and ``visible_selected_count_per_training_camera.tsv``.

    If ``package_dir`` is set, copies ``frame00_baseline_primary.png`` → ``package_dir/render_camera_rgb.png``
    and red overlay → ``package_dir/selected_red_overlay_training_cam.png``.

    Returns ``best_camera_index``, ``overlay_camera_used``, ``visible_view0``, ``visible_best``, ``counts_sample``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("mode-b selection debug requires CUDA.")

    RGB2SH, inverse_sigmoid = _ensure_gs_utils()
    dbg_dir = Path(dbg_dir)
    dbg_dir.mkdir(parents=True, exist_ok=True)
    scene_ply = Path(scene_ply).resolve()

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # type: ignore

        separate_sh = True
    except ImportError:
        separate_sh = False

    print("[mode-b debug] Loading 3DGS once for selection visualization...", flush=True)
    backdrop = build_gs_mesh_backdrop(
        colmap_scene=colmap_scene,
        model_path=model_path,
        iteration=iteration,
        base_ply=scene_ply,
        camera_index=0,
    )
    views = backdrop.train_views
    if views is None:
        raise RuntimeError("build_gs_mesh_backdrop must return train_views for mode-b debug")

    g = backdrop.gaussians
    idx = torch.as_tensor(
        np.asarray(selected_indices, dtype=np.int64).ravel(),
        device=g._features_dc.device,
        dtype=torch.long,
    )
    idx = idx[(idx >= 0) & (idx < g._features_dc.shape[0])]
    if idx.numel() == 0:
        raise ValueError("run_mode_b_selection_debug_suite: empty selection")

    with torch.no_grad():
        xyz = g.get_xyz.detach()

    n_views = len(views)
    counts_per_view: list[int] = []
    for i in range(n_views):
        counts_per_view.append(count_selected_visible_in_view(views[i], xyz, idx))
    best_id = int(np.argmax(np.array(counts_per_view, dtype=np.int64)))
    visible_best = counts_per_view[best_id]

    sample_ids = [0, 50, 100, 150]
    counts_sample: dict[int, int] = {}
    for sid in sample_ids:
        counts_sample[sid] = counts_per_view[sid] if sid < n_views else 0

    (dbg_dir / "best_camera_index.txt").write_text(str(best_id), encoding="utf-8")
    lines = [f"{i}\t{c}" for i, c in enumerate(counts_per_view)]
    (dbg_dir / "visible_selected_count_per_training_camera.tsv").write_text(
        "camera_index\tvisible_selected_centers\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    LOGIT_DIM = -14.0
    dev, dt = g._opacity.device, g._opacity.dtype

    snap_dc = g._features_dc.detach().clone()
    snap_rest = g._features_rest.detach().clone()
    snap_op = g._opacity.detach().clone()

    def restore() -> None:
        with torch.no_grad():
            g._features_dc.copy_(snap_dc)
            g._features_rest.copy_(snap_rest)
            g._opacity.copy_(snap_op)

    red = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dt)
    sh_red = RGB2SH(red).view(1, 1, 3).expand(idx.numel(), 1, 3)
    op_boost = inverse_sigmoid(torch.tensor([[0.999]], device=dev, dtype=dt)).expand(idx.numel(), -1)

    def render_camera_pack(cam_id: int, tag: str) -> None:
        view = views[cam_id]
        bg_white = backdrop.background
        bg_black = torch.zeros_like(bg_white)

        restore()
        base = _render_one(
            view, g, backdrop.pipe, bg_white, backdrop.dataset.train_test_exp, separate_sh
        )
        torchvision.utils.save_image(base, str(dbg_dir / f"frame00_baseline_{tag}.png"))

        with torch.no_grad():
            g._features_dc[idx] = sh_red
            g._features_rest[idx] = 0.0
            g._opacity[idx] = op_boost
        hi = _render_one(
            view, g, backdrop.pipe, bg_white, backdrop.dataset.train_test_exp, separate_sh
        )
        torchvision.utils.save_image(hi, str(dbg_dir / f"frame00_selected_red_dc_{tag}.png"))
        _save_diff(base, hi, dbg_dir / f"frame00_selected_overlay_diff_{tag}.png")

        restore()
        with torch.no_grad():
            g._opacity.fill_(LOGIT_DIM)
            g._features_dc[idx] = sh_red
            g._features_rest[idx] = 0.0
            g._opacity[idx] = op_boost
        solo = _render_one(
            view, g, backdrop.pipe, bg_black, backdrop.dataset.train_test_exp, separate_sh
        )
        torchvision.utils.save_image(solo, str(dbg_dir / f"frame00_selected_only_black_{tag}.png"))
        restore()

    cam_overlay = int(best_id)
    if forced_overlay_camera_index is not None:
        cci = int(forced_overlay_camera_index)
        if 0 <= cci < n_views:
            cam_overlay = cci

    try:
        render_camera_pack(0, "view0")
        for sid in (50, 100, 150):
            if sid < n_views:
                render_camera_pack(sid, f"view{sid:03d}")

        render_camera_pack(best_id, f"best_c{best_id}")
        render_camera_pack(cam_overlay, "primary")
        src_diff_p = dbg_dir / "frame00_selected_overlay_diff_primary.png"
        if src_diff_p.is_file():
            shutil.copy(src_diff_p, dbg_dir / "frame00_selected_overlay_diff.png")
        if package_dir is not None:
            pd = Path(package_dir)
            pd.mkdir(parents=True, exist_ok=True)
            bp = dbg_dir / "frame00_baseline_primary.png"
            rp = dbg_dir / "frame00_selected_red_dc_primary.png"
            if bp.is_file():
                shutil.copy(bp, pd / "render_camera_rgb.png")
            if rp.is_file():
                shutil.copy(rp, pd / "selected_red_overlay_training_cam.png")
    finally:
        restore()

    try:
        del backdrop
        del g
        del views
        del snap_dc
        del snap_rest
        del snap_op
    except NameError:
        pass
    torch.cuda.empty_cache()

    return {
        "best_camera_index": best_id,
        "overlay_camera_used": cam_overlay,
        "visible_view0": counts_per_view[0],
        "visible_best": visible_best,
        "counts_sample": counts_sample,
        "n_training_cameras": n_views,
    }
