"""
Compatibility shim for PhysGaussian subprocesses.

This file is auto-imported by Python if its directory is on PYTHONPATH.
We use it to patch external dependency API differences without modifying
submodule source files.
"""

from __future__ import annotations


def _patch_simulate_indices_selection() -> None:
    """Use ``simulate_indices_npy`` as an exact PhysGaussian dynamic subset.

    Upstream ``gs_simulation.py`` only uses ``sim_area`` as a bbox selector.  For
    mode-a, that bbox can include rug/floor Gaussians near the inserted object.
    This shim keeps the upstream bbox code as a fallback, but when
    ``simulate_indices_npy`` is present it changes the bbox mask's initial value
    from "all True" to the exact post-opacity selected-index mask.  The existing
    bbox comparisons can only remove points from that set, never add carpet.

    The legacy desk_jelly mode-b preset intentionally opts out so changing only
    its output folder does not change its original PhysGaussian sim_area behavior.
    """
    import builtins
    import json
    import os
    from pathlib import Path

    cfg_path = os.environ.get("PHYSGAUSSIAN_CONFIG_PATH")
    if not cfg_path:
        return
    try:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception:
        return

    if cfg.get("disable_exact_simulate_indices_for_preset") == "desk_jelly":
        return

    sim_npy = cfg.get("simulate_indices_npy")
    if not sim_npy:
        return

    try:
        import numpy as np
        import torch
    except Exception:
        return

    try:
        idx_np = np.load(sim_npy).astype(np.int64, copy=False).ravel()
    except Exception:
        return
    if idx_np.size == 0:
        return

    opacity_threshold = float(cfg.get("opacity_threshold", 0.02))
    state = {
        "mask_after_opacity": None,
        "used_for_bbox": False,
    }

    orig_ones = torch.ones

    def _ones_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        exact = state.get("mask_after_opacity")
        dtype = kwargs.get("dtype", None)
        try:
            requested_n = int(args[0]) if len(args) >= 1 else -1
        except Exception:
            requested_n = -1
        if (
            exact is not None
            and not bool(state.get("used_for_bbox", False))
            and dtype is torch.bool
            and requested_n == int(exact.shape[0])
        ):
            state["used_for_bbox"] = True
            print(
                "[physgaussian_shim] simulate_indices_npy exact selection active: "
                f"dynamic={int(exact.sum())} total_after_opacity={int(exact.shape[0])}",
                flush=True,
            )
            return torch.as_tensor(exact, dtype=torch.bool)
        return orig_ones(*args, **kwargs)

    torch.ones = _ones_wrapper  # type: ignore[assignment]

    orig_import = builtins.__import__

    def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        mod = orig_import(name, globals, locals, fromlist, level)
        try:
            if name == "utils.render_utils":
                fn = getattr(mod, "load_params_from_gs", None)
                if callable(fn) and getattr(fn, "__name__", "") != "_wrapped_load_params_from_gs":

                    def _wrapped_load_params_from_gs(*args, **kwargs):  # type: ignore[no-untyped-def]
                        params = fn(*args, **kwargs)
                        try:
                            opacity = params.get("opacity")
                            if not isinstance(opacity, torch.Tensor):
                                return params
                            keep = (opacity[:, 0] > opacity_threshold).detach().cpu().numpy()
                            total = int(keep.shape[0])
                            valid = idx_np[(idx_np >= 0) & (idx_np < total)]
                            selected_global = np.zeros(total, dtype=bool)
                            selected_global[valid] = True
                            exact = selected_global[keep]
                            state["mask_after_opacity"] = exact
                            print(
                                "[physgaussian_shim] loaded simulate_indices_npy: "
                                f"path={sim_npy} selected_global={int(selected_global.sum())} "
                                f"selected_after_opacity={int(exact.sum())} total_after_opacity={int(exact.shape[0])}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"[physgaussian_shim] simulate_indices_npy exact selection disabled: {exc!r}",
                                flush=True,
                            )
                        return params

                    mod.load_params_from_gs = _wrapped_load_params_from_gs  # type: ignore[assignment]
        except Exception:
            pass
        return mod

    builtins.__import__ = _import_hook  # type: ignore[assignment]


def _patch_mode_b_sand_render_override() -> None:
    """
    Sand mode rendering fix (main repo only).

    PhysGaussian updates selected means3D, but the default 3DGS splat appearance
    (scale/covariance, opacity, SH texture) can preserve chair-like look and bias
    visibility. Without modifying submodules, we wrap `gaussian_renderer.render`
    and apply a one-time appearance override for selected Gaussians when the
    PhysGaussian JSON config enables it:

      mode_b_sand_render_override: true
      mode_b_sand_opacity_scale/min/max
      mode_b_sand_cov_scale
      mode_b_sand_color_override_rgb
    """
    import os
    import json

    cfg_path = os.environ.get("PHYSGAUSSIAN_CONFIG_PATH")
    if not cfg_path:
        return
    try:
        cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
    except Exception:
        return

    if str(cfg.get("material", "")).lower() != "sand":
        return
    if not bool(cfg.get("mode_b_sand_render_override", False)):
        return

    # We can't import `gaussian_renderer` yet because PhysGaussian adds
    # "gaussian-splatting" to sys.path inside `gs_simulation.py` after startup.
    # Instead, install an import hook and patch once `gaussian_renderer` is imported.
    import builtins
    import importlib
    import types

    try:
        import numpy as np
        import torch
    except Exception:
        return

    applied = {"done": False}

    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _logit(p: torch.Tensor) -> torch.Tensor:
        return torch.log(p / (1.0 - p))

    def _apply_once(gaussians: object) -> None:
        if applied["done"]:
            return
        applied["done"] = True

        sim_npy = cfg.get("simulate_indices_npy", None)
        if not sim_npy:
            return
        try:
            idx_np = np.load(sim_npy).astype(np.int64, copy=False)
        except Exception:
            return

        try:
            dev = gaussians._opacity.device  # type: ignore[attr-defined]
            dt = gaussians._opacity.dtype  # type: ignore[attr-defined]
        except Exception:
            return
        idx = torch.as_tensor(idx_np, device=dev, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < gaussians._opacity.shape[0])]  # type: ignore[attr-defined]
        if int(idx.numel()) == 0:
            return

        opacity_scale = float(cfg.get("mode_b_sand_opacity_scale", 1.0))
        op_min = float(cfg.get("mode_b_sand_opacity_min", 0.05))
        op_max = float(cfg.get("mode_b_sand_opacity_max", 0.85))
        cov_scale = float(cfg.get("mode_b_sand_cov_scale", 1.0))
        rgb = cfg.get("mode_b_sand_color_override_rgb", None)

        with torch.no_grad():
            # --- opacity override (sigmoid space scale + clamp) ---
            logits = gaussians._opacity[idx]  # type: ignore[attr-defined]
            p = _sigmoid(logits) * float(opacity_scale)
            p = torch.clamp(p, min=op_min, max=op_max)
            gaussians._opacity[idx] = _logit(p)

            # --- covariance/scale override ---
            # In gaussian-splatting GaussianModel, `_scaling` stores log-scale.
            if hasattr(gaussians, "_scaling") and cov_scale != 1.0:
                try:
                    add = float(np.log(max(1e-6, cov_scale)))
                    gaussians._scaling[idx] = gaussians._scaling[idx] + add  # type: ignore[attr-defined]
                except Exception:
                    pass

            # --- color override (SH) ---
            if rgb is not None:
                try:
                    from utils.sh_utils import RGB2SH  # type: ignore
                except Exception:
                    RGB2SH = None
                if RGB2SH is not None:
                    c = torch.tensor([float(rgb[0]), float(rgb[1]), float(rgb[2])], device=dev, dtype=dt)
                    sh = RGB2SH(c).view(1, 1, 3).expand(int(idx.numel()), 1, 3)
                    gaussians._features_dc[idx] = sh  # type: ignore[attr-defined]
                    gaussians._features_rest[idx] = 0.0  # type: ignore[attr-defined]

        print(
            "[physgaussian_shim] sand render override applied: "
            f"opacity_scale={opacity_scale} clamp=[{op_min},{op_max}] "
            f"cov_scale={cov_scale} color_override={'yes' if rgb is not None else 'no'} "
            f"n_selected={int(idx.numel())}",
            flush=True,
        )

    def _try_patch_gaussian_renderer(mod: types.ModuleType) -> None:
        orig_render = getattr(mod, "render", None)
        if not callable(orig_render):
            return
        if getattr(orig_render, "__name__", "") == "_wrapped_render":
            return

        seen = {"once": False}

        def _wrapped_render(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Signature in gaussian-splatting: render(viewpoint_camera, pc: GaussianModel, pipe, bg_color, ...)
            try:
                gaussians = args[1]
                if not seen["once"]:
                    seen["once"] = True
                    print("[physgaussian_shim] sand override: first render() call", flush=True)
                _apply_once(gaussians)
            except Exception:
                pass
            return orig_render(*args, **kwargs)

        mod.render = _wrapped_render  # type: ignore[assignment]
        print("[physgaussian_shim] installed sand render override hook (render)", flush=True)

    orig_import = builtins.__import__

    def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        mod = orig_import(name, globals, locals, fromlist, level)
        try:
            if name == "gaussian_renderer" or (isinstance(fromlist, tuple) and "GaussianModel" in fromlist):
                gr = importlib.import_module("gaussian_renderer")
                _try_patch_gaussian_renderer(gr)
            # PhysGaussian entrypoint imports `render` into its module globals:
            #   from gaussian_renderer import render, GaussianModel
            # Depending on import timing, patch the `gs_simulation.render` symbol too.
            if name.endswith("gs_simulation") or name == "gs_simulation":
                import sys

                gr = importlib.import_module("gaussian_renderer")
                _try_patch_gaussian_renderer(gr)
                m = sys.modules.get(name)
                if m is not None and hasattr(gr, "render"):
                    setattr(m, "render", getattr(gr, "render"))
                    print("[physgaussian_shim] patched gs_simulation.render symbol", flush=True)
        except Exception:
            pass
        return mod

    builtins.__import__ = _import_hook  # type: ignore[assignment]


def _patch_diff_gaussian_rasterization() -> None:
    try:
        import atexit
        import inspect
        import os
        from pathlib import Path

        import diff_gaussian_rasterization as dgr
    except Exception:
        return

    try:
        orig = dgr.GaussianRasterizationSettings
        sig = inspect.signature(orig)
    except Exception:
        return

    if "antialiasing" not in sig.parameters:
        return

    def _wrapped_settings(*args, **kwargs):  # type: ignore[no-untyped-def]
        # PhysGaussian's original code calls without `antialiasing`.
        kwargs.setdefault("antialiasing", False)
        return orig(*args, **kwargs)

    dgr.GaussianRasterizationSettings = _wrapped_settings  # type: ignore[assignment]

    # Newer rasterizers may return extra outputs (e.g. invdepth) while PhysGaussian expects
    # exactly (rendering, radii). Normalize return arity here.
    try:
        orig_rasterize = dgr.rasterize_gaussians
    except Exception:
        orig_rasterize = None

    if callable(orig_rasterize):
        # Optional render-side sand override (works even if `render` was imported directly),
        # because everything funnels through rasterize_gaussians.
        import json

        sand_cfg = None
        cfg_path = os.environ.get("PHYSGAUSSIAN_CONFIG_PATH")
        if cfg_path:
            try:
                sand_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            except Exception:
                sand_cfg = None

        debug_dir_raw = os.environ.get("PHYSGAUSSIAN_DEBUG_DIR")
        try:
            debug_dir = Path(debug_dir_raw) if debug_dir_raw else None
        except Exception:
            debug_dir = None

        try:
            selected_n = int(os.environ.get("PHYSGAUSSIAN_SELECTED_N", "0"))
        except Exception:
            selected_n = 0
        try:
            frame_num_cfg = int(sand_cfg.get("frame_num", 0)) if isinstance(sand_cfg, dict) else 0
        except Exception:
            frame_num_cfg = 0
        if frame_num_cfg <= 0:
            try:
                frame_num_cfg = int(sand_cfg.get("frame_num_test", 0)) if isinstance(sand_cfg, dict) else 0
            except Exception:
                frame_num_cfg = 0
        frame_num_cfg = int(frame_num_cfg) if int(frame_num_cfg) > 0 else 200

        # We track per-render moved gaussians by comparing the 3D means
        # passed into rasterization against the first frame.
        moved_any: set[int] = set()
        cached_means = None
        first_means = None
        last_means = None
        per_frame_means: list = []
        sand_logged_once = False
        render_call_idx = 0

        def _write_debug() -> None:
            if not debug_dir:
                return
            if selected_n <= 0:
                return
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                out = debug_dir / "moved_selected_local_indices.npy"
                import numpy as np

                arr = np.array(sorted(moved_any), dtype=np.int64)
                np.save(out, arr)
                print(
                    f"[physgaussian_shim] wrote moved indices: {out} (n={arr.size}, selected_n={selected_n})",
                    flush=True,
                )

                if first_means is not None:
                    np.save(debug_dir / "selected_means3d_first.npy", first_means)
                if last_means is not None:
                    np.save(debug_dir / "selected_means3d_last.npy", last_means)
                # Optional: save per-frame means for object-mask compositing.
                try:
                    if (
                        isinstance(sand_cfg, dict)
                        and bool(sand_cfg.get("mode_b_save_selected_means3d_per_frame", False))
                        and per_frame_means
                    ):
                        arrf = np.stack(per_frame_means, axis=0).astype(np.float32, copy=False)
                        np.save(debug_dir / "selected_means3d_per_frame.npy", arrf)
                        print(
                            f"[physgaussian_shim] wrote per-frame means: {debug_dir/'selected_means3d_per_frame.npy'} "
                            f"shape={tuple(arrf.shape)}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[physgaussian_shim] failed writing per-frame means: {e!r}", flush=True)
            except Exception as e:
                print(f"[physgaussian_shim] failed writing moved indices: {e!r}", flush=True)

        atexit.register(_write_debug)

        def _wrapped_rasterize(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal cached_means, first_means, last_means
            nonlocal sand_logged_once
            nonlocal render_call_idx
            call_idx = int(render_call_idx)
            # Best-effort: infer means3D from the first positional arg.
            means3d = args[0] if args else None
            try:
                import torch

                if (
                    debug_dir is not None
                    and selected_n > 0
                    and isinstance(means3d, torch.Tensor)
                    and means3d.ndim == 2
                    and means3d.shape[0] >= selected_n
                    and means3d.shape[1] >= 3
                ):
                    m = means3d[:selected_n, :3].detach()
                    if cached_means is None:
                        cached_means = m.clone()
                        try:
                            first_means = m.cpu().numpy()
                        except Exception:
                            first_means = None
                    else:
                        # 1e-4 meters threshold to ignore tiny numeric noise.
                        d = (m - cached_means).norm(dim=1)
                        moved = torch.nonzero(d > 1e-4, as_tuple=False).view(-1)
                        if moved.numel():
                            moved_any.update(int(i) for i in moved.tolist())
                    try:
                        last_means = m.cpu().numpy()
                    except Exception:
                        last_means = None
                    try:
                        if (
                            sand_cfg is not None
                            and isinstance(sand_cfg, dict)
                            and bool(sand_cfg.get("mode_b_save_selected_means3d_per_frame", False))
                        ):
                            per_frame_means.append(m.cpu().numpy())
                    except Exception:
                        pass
            except Exception:
                pass

            # EXTREME binary sanity check: force all selected gaussians to move down together
            # by a large linear ramp over the whole video.
            try:
                import torch

                if (
                    sand_cfg is not None
                    and str(sand_cfg.get("material", "")).lower() == "sand"
                    and bool(sand_cfg.get("mode_b_sand_extreme_all_selected_down", False))
                    and selected_n > 0
                    and isinstance(means3d, torch.Tensor)
                    and means3d.ndim == 2
                    and means3d.shape[0] >= selected_n
                    and means3d.shape[1] >= 3
                ):
                    total_down = float(sand_cfg.get("mode_b_sand_extreme_total_down_m", 0.60))
                    frame_idx = int(min(render_call_idx, max(0, frame_num_cfg - 1)))
                    t = 0.0 if frame_num_cfg <= 1 else (frame_idx / float(frame_num_cfg - 1))
                    off = float(total_down) * float(t)
                    with torch.no_grad():
                        sel = means3d[:selected_n, :3]
                        y_before = sel[:, 1]
                        yb_min = float(y_before.min())
                        yb_mean = float(y_before.mean())
                        yb_max = float(y_before.max())
                        sel[:, 1] = y_before + off
                        ya = sel[:, 1]
                        ya_min = float(ya.min())
                        ya_mean = float(ya.mean())
                        ya_max = float(ya.max())

                    if frame_idx in (0, 1, 2, 50, 100, 150, frame_num_cfg - 1):
                        print(
                            "[physgaussian_shim] EXTREME all-selected-down: "
                            f"frame={frame_idx}/{frame_num_cfg-1} off={off:.4f}m selected_n={selected_n}  "
                            f"y_before(min/mean/max)=({yb_min:.4f},{yb_mean:.4f},{yb_max:.4f})  "
                            f"y_after(min/mean/max)=({ya_min:.4f},{ya_mean:.4f},{ya_max:.4f})  "
                            "means3D_tensor=arg0(rasterize_gaussians)",
                            flush=True,
                        )
            except Exception:
                pass

            # True subset rendering for contribution audit:
            # - selected_only_true: pass only selected gaussians to rasterizer
            # - unselected_only_true: pass only unselected gaussians (everything after selected_n)
            # - id_colored_sets: render all, but color selected=red, unselected=blue for visual audit
            try:
                import torch

                if sand_cfg is not None and isinstance(sand_cfg, dict):
                    subset = str(sand_cfg.get("mode_b_render_gaussian_subset", "all"))
                else:
                    subset = "all"

                if subset in ("selected_only_true", "unselected_only_true"):
                    if selected_n > 0 and len(args) >= 9 and isinstance(args[0], torch.Tensor):
                        # rasterize_gaussians signature:
                        # 0 means3D, 1 means2D, 2 shs, 3 colors_precomp, 4 opacities,
                        # 5 scales, 6 rotations, 7 cov3D_precomp, 8 raster_settings
                        start = 0
                        end = selected_n
                        if subset == "unselected_only_true":
                            start = selected_n
                            end = args[0].shape[0]
                        new_args = list(args)
                        for k in range(0, 8):
                            t = new_args[k]
                            if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == args[0].shape[0]:
                                new_args[k] = t[start:end]
                        args = tuple(new_args)
                        if not sand_logged_once:
                            sand_logged_once = True
                            print(
                                f"[physgaussian_shim] TRUE subset render active: {subset}  "
                                f"selected_n={selected_n}  total_n={int(new_args[0].shape[0] if isinstance(new_args[0], torch.Tensor) else -1)}",
                                flush=True,
                            )

                elif subset == "id_colored_sets":
                    # Set SH so selected render red, unselected render blue.
                    if selected_n > 0 and len(args) >= 3 and isinstance(args[2], torch.Tensor):
                        try:
                            from utils.sh_utils import RGB2SH  # type: ignore
                        except Exception:
                            RGB2SH = None
                        if RGB2SH is not None:
                            shs = args[2]
                            if shs.ndim == 3 and shs.shape[0] >= selected_n:
                                with torch.no_grad():
                                    red = RGB2SH(torch.tensor([1.0, 0.0, 0.0], device=shs.device, dtype=shs.dtype))
                                    blue = RGB2SH(torch.tensor([0.0, 0.0, 1.0], device=shs.device, dtype=shs.dtype))
                                    shs.zero_()
                                    # layout handling
                                    if shs.shape[1] == 3:
                                        shs[:selected_n, :, 0] = red.view(1, 3).expand(selected_n, 3)
                                        shs[selected_n:, :, 0] = blue.view(1, 3).expand(shs.shape[0] - selected_n, 3)
                                    elif shs.shape[2] == 3:
                                        shs[:selected_n, 0, :] = red.view(1, 3).expand(selected_n, 3)
                                        shs[selected_n:, 0, :] = blue.view(1, 3).expand(shs.shape[0] - selected_n, 3)
                            if not sand_logged_once:
                                sand_logged_once = True
                                print(
                                    f"[physgaussian_shim] id_colored_sets active: selected=red unselected=blue  selected_n={selected_n}",
                                    flush=True,
                                )
            except Exception:
                pass

            # Optional: dump selected-only renders for black-backside artifact analysis.
            # Writes additional PNGs per frame under <debug_dir>/extra_renders/.
            try:
                import torch

                if (
                    debug_dir is not None
                    and selected_n > 0
                    and sand_cfg is not None
                    and isinstance(sand_cfg, dict)
                    and bool(sand_cfg.get("mode_b_debug_dump_selected_only_renders", False))
                    and len(args) >= 9
                    and isinstance(args[0], torch.Tensor)
                ):
                    # Only dump on the main (all) render path to avoid recursion.
                    subset_now = str(sand_cfg.get("mode_b_render_gaussian_subset", "all"))
                    if subset_now == "all":
                        orig_r = orig_rasterize  # from closure

                        def _slice_for_selected(a):
                            if isinstance(a, torch.Tensor) and a.ndim >= 1 and a.shape[0] == args[0].shape[0]:
                                return a[:selected_n]
                            return a

                        base_args = list(args)
                        sel_args = [base_args[0][:selected_n]]
                        for k in range(1, 8):
                            sel_args.append(_slice_for_selected(base_args[k]))
                        sel_args.append(base_args[8])  # raster_settings
                        sel_args = tuple(sel_args)

                        # Render variants by cov override on the cov3D_precomp tensor (arg7).
                        def _render_variant(kind: str):
                            mdir = debug_dir / "extra_renders" / kind
                            mdir.mkdir(parents=True, exist_ok=True)
                            # Clone cov so we don't perturb the main render.
                            v_args = list(sel_args)
                            if isinstance(v_args[7], torch.Tensor):
                                cov = v_args[7].clone()
                                if kind == "selected_only_clamp_diag":
                                    cmin = float(sand_cfg.get("mode_b_render_cov_diag_min", 1.0e-6))
                                    cmax = float(sand_cfg.get("mode_b_render_cov_diag_max", 5.0e-3))
                                    cov[:, 0] = torch.clamp(cov[:, 0], min=cmin, max=cmax)
                                    cov[:, 1] = torch.clamp(cov[:, 1], min=cmin, max=cmax)
                                    cov[:, 2] = torch.clamp(cov[:, 2], min=cmin, max=cmax)
                                    cov[:, 3:6] = 0.0
                                elif kind == "selected_only_tiny_splats":
                                    var = float(sand_cfg.get("mode_b_render_tiny_splats_var", 1.0e-6))
                                    cov.zero_()
                                    cov[:, 0] = var
                                    cov[:, 1] = var
                                    cov[:, 2] = var
                                v_args[7] = cov
                            res = orig_r(*v_args)
                            rend = res[0] if isinstance(res, (tuple, list)) else res
                            # Save frame image + binary mask.
                            try:
                                import numpy as np
                                import cv2

                                img = rend.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                                bgr = (img[..., ::-1] * 255.0).astype(np.uint8)
                                stem = f"{call_idx:04d}.png"
                                cv2.imwrite(str(mdir / stem), bgr)
                                # Coverage mask for compositing: normal pass uses a softer threshold + morph
                                # so matte compositing does not punch plate holes through the chair.
                                eps = 1.0e-4 if kind == "selected_only_normal" else 1.0e-3
                                m = (img.max(axis=2) > eps).astype(np.uint8) * 255
                                if kind == "selected_only_normal":
                                    km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, km, iterations=1)
                                    m = cv2.dilate(m, km, iterations=1)
                                mm = debug_dir / "extra_renders" / (kind + "_mask")
                                mm.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(mm / stem), m)
                            except Exception:
                                pass

                        variants = str(
                            sand_cfg.get("mode_b_extra_render_dump_variants", "all")
                        ).lower()
                        _render_variant("selected_only_normal")
                        if variants != "normal_only":
                            _render_variant("selected_only_clamp_diag")
                            _render_variant("selected_only_tiny_splats")
            except Exception:
                pass

            # Advance the call index at the end so any per-call dumps use stable numbering.
            render_call_idx += 1

            # Render-space artifact checks: override cov3D_precomp just before rasterization.
            # This does not change the physics state; it only changes how splats are rendered.
            # Keys in phys_config.json:
            # - mode_b_render_cov_override: "none"|"tiny_splats"|"clamp_diag"|"freeze_first"
            # - mode_b_render_cov_override_scope: "selected"|"all"
            # - mode_b_render_tiny_splats_var, mode_b_render_cov_diag_min/max
            try:
                import torch

                cov_mode = "none"
                cov_scope = "selected"
                if sand_cfg is not None and isinstance(sand_cfg, dict):
                    cov_mode = str(sand_cfg.get("mode_b_render_cov_override", "none"))
                    cov_scope = str(sand_cfg.get("mode_b_render_cov_override_scope", "selected"))
                if cov_mode != "none" and len(args) >= 8 and isinstance(args[7], torch.Tensor):
                    cov = args[7]
                    if cov.ndim == 2 and cov.shape[1] >= 6 and cov.shape[0] == args[0].shape[0]:
                        start = 0
                        end = cov.shape[0]
                        if cov_scope == "selected" and selected_n > 0:
                            end = min(int(selected_n), end)

                        # Cache frame-0 cov for freeze_first.
                        if cov_mode == "freeze_first":
                            if cached_means is None:
                                pass
                            if "cached_cov0" not in locals():
                                pass
                        # Use a dict to persist across calls.
                        nonlocal_vars = _wrapped_rasterize.__dict__.setdefault("_cov_cache", {})
                        if cov_mode == "freeze_first" and "cov0" not in nonlocal_vars:
                            nonlocal_vars["cov0"] = cov[:end].detach().clone()

                        with torch.no_grad():
                            if cov_mode == "tiny_splats":
                                var = float(sand_cfg.get("mode_b_render_tiny_splats_var", 1.0e-6)) if isinstance(sand_cfg, dict) else 1.0e-6
                                cov[:end].zero_()
                                cov[:end, 0] = var
                                cov[:end, 1] = var
                                cov[:end, 2] = var
                            elif cov_mode == "clamp_diag":
                                cmin = float(sand_cfg.get("mode_b_render_cov_diag_min", 1.0e-6)) if isinstance(sand_cfg, dict) else 1.0e-6
                                cmax = float(sand_cfg.get("mode_b_render_cov_diag_max", 5.0e-3)) if isinstance(sand_cfg, dict) else 5.0e-3
                                # clamp diagonals; zero off-diagonals to reduce bleed from skew.
                                cov[:end, 0] = torch.clamp(cov[:end, 0], min=cmin, max=cmax)
                                cov[:end, 1] = torch.clamp(cov[:end, 1], min=cmin, max=cmax)
                                cov[:end, 2] = torch.clamp(cov[:end, 2], min=cmin, max=cmax)
                                cov[:end, 3:6] = 0.0
                            elif cov_mode == "freeze_first":
                                cov0 = nonlocal_vars.get("cov0", None)
                                if isinstance(cov0, torch.Tensor) and cov0.shape == cov[:end].shape:
                                    cov[:end].copy_(cov0)
            except Exception:
                pass

            # Sand motion correction: make global settling coherent across selected gaussians.
            try:
                import torch

                if (
                    sand_cfg is not None
                    and str(sand_cfg.get("material", "")).lower() == "sand"
                    and bool(sand_cfg.get("mode_b_sand_motion_correction", False))
                    and selected_n > 0
                    and isinstance(means3d, torch.Tensor)
                    and means3d.ndim == 2
                    and means3d.shape[0] >= selected_n
                    and means3d.shape[1] >= 3
                ):
                    alpha = float(sand_cfg.get("mode_b_sand_motion_alpha", 0.65))
                    pctl = float(sand_cfg.get("mode_b_sand_global_dy_percentile", 60.0))
                    min_frac = sand_cfg.get("mode_b_sand_min_dy_as_global_frac", 0.15)
                    min_frac = float(min_frac) if min_frac is not None else None

                    with torch.no_grad():
                        sel = means3d[:selected_n, :3]
                        if cached_means is None:
                            # Initialize baseline for dy computation.
                            cached_means = sel.detach().clone()
                        dy = sel[:, 1] - cached_means[:, 1]  # +Y is down in this repo's convention
                        # robust global dy
                        global_dy = torch.quantile(dy, torch.tensor(pctl / 100.0, device=dy.device, dtype=dy.dtype))
                        # blend towards global dy
                        dy_corr = (1.0 - alpha) * dy + alpha * global_dy
                        if min_frac is not None:
                            dy_corr = torch.maximum(dy_corr, global_dy * float(min_frac))
                        sel[:, 1] = cached_means[:, 1] + dy_corr

                        if not sand_logged_once:
                            # Print once (reuses the same flag bucket) to prove it's active.
                            sand_logged_once = True
                            print(
                                "[physgaussian_shim] sand motion correction active: "
                                f"selected_n={selected_n} alpha={alpha} pctl={pctl} "
                                f"global_dy={float(global_dy):.6f}",
                                flush=True,
                            )
            except Exception:
                pass

            # Displacement propagation (render-time only): help weakly moved selected points follow neighbors.
            try:
                import torch

                if (
                    sand_cfg is not None
                    and bool(sand_cfg.get("mode_b_disp_propagation_enabled", False))
                    and selected_n > 0
                    and isinstance(means3d, torch.Tensor)
                    and means3d.ndim == 2
                    and means3d.shape[0] >= selected_n
                    and means3d.shape[1] >= 3
                ):
                    k = int(sand_cfg.get("mode_b_disp_propagation_k", 8))
                    k = max(1, min(k, max(1, selected_n - 1)))
                    low_p = float(sand_cfg.get("mode_b_disp_low_percentile", 10.0))
                    low_p = float(max(0.0, min(100.0, low_p)))
                    min_nb = float(sand_cfg.get("mode_b_disp_min_neighbor_moved_m", 0.05))
                    alpha_p = float(sand_cfg.get("mode_b_disp_propagation_alpha", 1.0))
                    alpha_p = float(max(0.0, min(1.0, alpha_p)))

                    with torch.no_grad():
                        sel = means3d[:selected_n, :3]
                        if cached_means is None:
                            cached_means = sel.detach().clone()
                        disp = sel - cached_means
                        mag = torch.linalg.norm(disp, dim=1)
                        thr = torch.quantile(
                            mag,
                            torch.tensor(low_p / 100.0, device=mag.device, dtype=mag.dtype),
                        )
                        low = mag <= thr

                        cache = _wrapped_rasterize.__dict__.setdefault("_disp_prop_cache", {})
                        if "neighbors" not in cache:
                            try:
                                import numpy as np
                                from scipy.spatial import cKDTree

                                X0 = cached_means.detach().cpu().numpy()
                                tree = cKDTree(X0)
                                _, nn = tree.query(X0, k=k + 1)
                                nn = nn[:, 1:]
                                cache["neighbors"] = torch.as_tensor(nn, device=sel.device, dtype=torch.long)
                            except Exception:
                                cache["neighbors"] = None

                        nbr = cache.get("neighbors", None)
                        if isinstance(nbr, torch.Tensor) and nbr.shape[0] == selected_n:
                            nbr_disp = disp[nbr]  # (N,k,3)
                            nbr_mag = mag[nbr]  # (N,k)
                            w = (nbr_mag >= min_nb).to(disp.dtype)
                            wsum = w.sum(dim=1).clamp_min(1.0)
                            avg = (nbr_disp * w[..., None]).sum(dim=1) / wsum[:, None]
                            disp2 = torch.where(low[:, None], (1.0 - alpha_p) * disp + alpha_p * avg, disp)
                            sel.copy_(cached_means + disp2)
            except Exception:
                pass

            # Sand appearance override: adjust opacity / scale / SH for first `selected_n` gaussians.
            try:
                import torch

                if (
                    sand_cfg is not None
                    and str(sand_cfg.get("material", "")).lower() == "sand"
                    and bool(sand_cfg.get("mode_b_sand_render_override", False))
                    and selected_n > 0
                    and len(args) >= 6
                    and isinstance(args[4], torch.Tensor)  # opacities
                ):
                    opacities = args[4]
                    cov_scale = float(sand_cfg.get("mode_b_sand_cov_scale", 1.0))
                    opacity_scale = float(sand_cfg.get("mode_b_sand_opacity_scale", 1.0))
                    op_min = float(sand_cfg.get("mode_b_sand_opacity_min", 0.05))
                    op_max = float(sand_cfg.get("mode_b_sand_opacity_max", 0.85))
                    rgb = sand_cfg.get("mode_b_sand_color_override_rgb", None)

                    # opacities are already in [0,1] in gaussian-splatting renderer
                    with torch.no_grad():
                        if not sand_logged_once:
                            sand_logged_once = True
                            try:
                                sel_op = opacities[:selected_n]
                                msg = (
                                    f"[physgaussian_shim] sand rasterize override: selected_n={selected_n} "
                                    f"opacity(before) min={float(sel_op.min()):.4f} "
                                    f"mean={float(sel_op.mean()):.4f} max={float(sel_op.max()):.4f}  "
                                )
                                if isinstance(args[5], torch.Tensor):
                                    sel_sc = args[5][:selected_n]
                                    msg += (
                                        f"scale(before) min={float(sel_sc.min()):.4f} "
                                        f"mean={float(sel_sc.mean()):.4f} max={float(sel_sc.max()):.4f}  "
                                    )
                                if len(args) >= 3 and isinstance(args[2], torch.Tensor):
                                    shs = args[2]
                                    msg += f"shs(shape)={tuple(shs.shape)}  "
                                msg += (
                                    f"cfg(op_scale={opacity_scale}, clamp=[{op_min},{op_max}], "
                                    f"cov_scale={cov_scale}, color={'yes' if rgb is not None else 'no'})"
                                )
                                print(msg, flush=True)
                            except Exception:
                                print("[physgaussian_shim] sand rasterize override: (stats failed)", flush=True)

                        op = opacities[:selected_n] * float(opacity_scale)
                        op = torch.clamp(op, min=op_min, max=op_max)
                        opacities[:selected_n] = op

                        # scales tensor (if present) is args[5]
                        if isinstance(args[5], torch.Tensor) and cov_scale != 1.0:
                            scales = args[5]
                            scales[:selected_n] = scales[:selected_n] * float(cov_scale)

                        # SH tensor (if present) is args[2]
                        if rgb is not None and len(args) >= 3 and isinstance(args[2], torch.Tensor):
                            try:
                                from utils.sh_utils import RGB2SH  # type: ignore
                            except Exception:
                                RGB2SH = None
                            if RGB2SH is not None:
                                shs = args[2]
                                # Make a per-gaussian constant SH (zero higher orders).
                                c = torch.tensor(
                                    [float(rgb[0]), float(rgb[1]), float(rgb[2])],
                                    device=shs.device,
                                    dtype=shs.dtype,
                                )
                                dc = RGB2SH(c)
                                # Handle both (N, 3, C) and (N, C, 3) layouts.
                                if shs.ndim == 3 and shs.shape[0] >= selected_n:
                                    shs_sel = shs[:selected_n]
                                    shs_sel.zero_()
                                    if shs_sel.shape[1] == 3:
                                        shs_sel[:, :, 0] = dc.view(1, 3).expand(selected_n, 3)
                                    elif shs_sel.shape[2] == 3:
                                        shs_sel[:, 0, :] = dc.view(1, 3).expand(selected_n, 3)
            except Exception:
                pass

            out = orig_rasterize(*args, **kwargs)
            if isinstance(out, tuple) and len(out) > 2:
                return out[0], out[1]
            return out

        dgr.rasterize_gaussians = _wrapped_rasterize  # type: ignore[assignment]


def _patch_taichi_init_device_memory() -> None:
    """
    PhysGaussian submodule may hard-code `ti.init(device_memory_GB=8.0)`.

    Without modifying submodules, we override only the Taichi memory argument
    from an environment variable set by the main repo (`mpm_runner.py`):
      PHYSGAUSSIAN_TAICHI_DEVICE_MEMORY_GB
    """
    import os

    raw = os.environ.get("PHYSGAUSSIAN_TAICHI_DEVICE_MEMORY_GB")
    if raw is None:
        return
    try:
        wanted_gb = float(raw)
    except Exception:
        return

    try:
        import taichi as ti
    except Exception:
        return

    orig_init = getattr(ti, "init", None)
    if not callable(orig_init):
        return

    def _init_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        before = kwargs.get("device_memory_GB", None)
        kwargs["device_memory_GB"] = float(wanted_gb)
        print(
            f"[physgaussian_shim] taichi.init override device_memory_GB: {before!r} -> {wanted_gb}",
            flush=True,
        )
        return orig_init(*args, **kwargs)

    ti.init = _init_wrapper  # type: ignore[assignment]


_patch_simulate_indices_selection()
_patch_diff_gaussian_rasterization()
_patch_taichi_init_device_memory()
_patch_mode_b_sand_render_override()

