"""Convert a generated mesh surface into a PhysGaussian-compatible PLY."""
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from plyfile import PlyData, PlyElement

from generation.gltf_mesh_loader import (
    load_glb_colored_primitive_meshes,
    merge_glb_primitive_geometries,
)

if TYPE_CHECKING:
    import trimesh

C0 = 0.28209479177387814

OBJECT_MESH_GLB = "object_mesh.glb"
OBJECT_MESH_OBJ = "object_mesh.obj"


def resolve_insertion_mesh_path(gen_dir: Path | str) -> Path:
    """Prefer ``object_mesh.glb`` over ``object_mesh.obj`` under ``gen_dir``."""
    d = Path(gen_dir)
    glb = d / OBJECT_MESH_GLB
    if glb.is_file():
        return glb
    obj = d / OBJECT_MESH_OBJ
    if obj.is_file():
        return obj
    raise FileNotFoundError(
        f"Neither {glb} nor {obj} exists — place an imported mesh for --mesh-gaussians."
    )


def insertion_mesh_source_exists(gen_dir: Path | str) -> bool:
    d = Path(gen_dir)
    return (d / OBJECT_MESH_GLB).is_file() or (d / OBJECT_MESH_OBJ).is_file()


def stage_external_mesh_into_gen_dir(gen_dir: Path | str, src: Path | str) -> Path:
    """Install ``src`` as ``object_mesh.glb`` or ``object_mesh.obj`` under ``gen_dir`` for insertion.

    Use this for Sketchfab/Blender assets so ``resolve_insertion_mesh_path`` and
    ``--mesh-gaussians`` / ``--mesh-render`` see a canonical layout. Removes the
    sibling ``.glb``/``.obj`` slot when the other wins to avoid ambiguous picks.
    """
    import shutil

    gen_dir = Path(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    src_p = Path(src).expanduser().resolve()
    if not src_p.is_file():
        raise FileNotFoundError(f"Not a file: {src_p}")

    target_glb = gen_dir / OBJECT_MESH_GLB
    target_obj = gen_dir / OBJECT_MESH_OBJ
    ext = src_p.suffix.lower()

    if ext == ".glb":
        shutil.copy2(src_p, target_glb)
        if target_obj.is_file():
            target_obj.unlink()
        return target_glb

    if ext == ".obj":
        shutil.copy2(src_p, target_obj)
        if target_glb.is_file():
            target_glb.unlink()
        return target_obj

    # Other mesh formats (e.g. STL, PLY surface mesh): load and export OBJ for downstream.
    mesh = load_insertion_mesh(src_p)
    mesh.export(str(target_obj))
    if target_glb.is_file():
        target_glb.unlink()
    return target_obj


def _mesh_material_bonus_score(m: "trimesh.Trimesh") -> float:
    """Extra score when glTF / simple materials expose a solid tint (not texture-only)."""
    mat = getattr(m.visual, "material", None)
    if mat is None:
        return 0.0
    if getattr(mat, "baseColorFactor", None) is not None:
        return 5e8
    if getattr(mat, "diffuse", None) is not None:
        return 5e8
    return 0.0


def _mesh_color_score(m: "trimesh.Trimesh") -> float:
    """Prefer meshes that still carry vertex colors or UV+texture after import."""
    score = float(m.area)
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is not None:
        ca = np.asarray(vc)
        if ca.ndim == 2 and ca.shape[0] == len(m.vertices) and ca.shape[1] >= 3:
            score += 1e9
    uv = getattr(m.visual, "uv", None)
    if uv is not None and _texture_image_from_visual(m.visual) is not None:
        score += 1e9
    score += _mesh_material_bonus_score(m)
    return score


def _pick_scene_mesh_for_sampling(scene: "trimesh.Scene", log: Callable[[str], None] | None) -> "trimesh.Trimesh":
    """Pick one Trimesh from a GLB Scene without ``concatenate`` (which strips materials)."""
    import trimesh

    geoms = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if not geoms:
        raise ValueError("Scene has no triangle meshes")
    if len(geoms) == 1:
        mesh = geoms[0]
        if log:
            log("[mesh_import] Scene with 1 geometry — using it as-is (materials preserved).")
        return mesh
    geoms.sort(key=_mesh_color_score, reverse=True)
    mesh = geoms[0]
    if log:
        log(
            f"[mesh_import] Scene has {len(geoms)} geometries — using highest-scoring submesh "
            f"(area={mesh.area:.4f}) to avoid trimesh.concatenate stripping GLB materials. "
            "Merge parts in Blender if you need all pieces in one object."
        )
    return mesh


def load_insertion_mesh(
    mesh_path: Path | str,
    log: Callable[[str], None] | None = None,
) -> "trimesh.Trimesh":
    """Load OBJ/GLB for Gaussian sampling.

    For ``.glb``, ``pygltflib`` is tried first so embedded ``baseColorTexture`` /
    buffer-backed images (common on Sketchfab exports) survive; ``trimesh.load``
    is the fallback for Draco-only assets or unusual files.
    """
    import trimesh

    mesh_path = Path(mesh_path).resolve()
    if mesh_path.suffix.lower() == ".glb":
        meshes = load_glb_colored_primitive_meshes(mesh_path, log=log)
        if meshes:
            merged = merge_glb_primitive_geometries(meshes)
            if log:
                log(
                    f"[mesh_import] GLB: merged {len(meshes)} colored primitive(s) for geometry export "
                    f"→ verts={len(merged.vertices)} faces={len(merged.faces)}"
                )
            if merged.is_empty:
                raise ValueError(f"Empty mesh from pygltflib merge: {mesh_path}")
            return merged

    loaded = trimesh.load(str(mesh_path), process=False)
    if log:
        log(f"[mesh_import] path={mesh_path}  trimesh_type={type(loaded).__name__}")
    if isinstance(loaded, trimesh.Scene):
        mesh = _pick_scene_mesh_for_sampling(loaded, log)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported trimesh load type {type(loaded)} from {mesh_path}")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    return mesh


def _texture_image_from_visual(visual: object) -> object | None:
    """Return a PIL-friendly image object or None."""
    mat = getattr(visual, "material", None)
    if mat is None:
        return getattr(visual, "image", None)
    img = getattr(mat, "image", None)
    if img is not None:
        return img
    bt = getattr(mat, "baseColorTexture", None)
    if bt is None:
        return None
    return getattr(bt, "image", None) or getattr(bt, "source", None)


def log_mesh_visual_diagnostics(mesh: "trimesh.Trimesh", log: Callable[[str], None] | None) -> None:
    """Log vertex colors / UV / material hints for debugging gray splats."""
    if log is None:
        return
    vis = mesh.visual
    log(f"[mesh_color] mesh.visual type: {type(vis).__module__}.{type(vis).__name__}")
    vc = getattr(vis, "vertex_colors", None)
    if vc is None:
        log("[mesh_color] vertex_colors: absent")
    else:
        ca = np.asarray(vc)
        log(f"[mesh_color] vertex_colors: shape={ca.shape} dtype={ca.dtype}")
    uv = getattr(vis, "uv", None)
    if uv is None:
        log("[mesh_color] uv: absent")
    else:
        u = np.asarray(uv)
        log(f"[mesh_color] uv: shape={u.shape} (verts={len(mesh.vertices)} faces={len(mesh.faces)})")
    img = _texture_image_from_visual(vis)
    log(f"[mesh_color] texture image: {'yes' if img is not None else 'no'}")
    mat = getattr(vis, "material", None)
    if mat is None:
        log("[mesh_color] material: absent")
    else:
        bf = getattr(mat, "baseColorFactor", None)
        diff = getattr(mat, "diffuse", None)
        bct = getattr(mat, "baseColorTexture", None)
        img_attr = getattr(mat, "image", None)
        log(
            f"[mesh_color] material: type={type(mat).__module__}.{type(mat).__name__}  "
            f"baseColorFactor={bf!r} diffuse={diff!r}  baseColorTexture={bct!r}  mat.image={img_attr!r}"
        )
        pub = sorted(x for x in dir(mat) if not x.startswith("_"))
        tail = " …" if len(pub) > 48 else ""
        log(f"[mesh_color] material public attrs ({len(pub)}): {pub[:48]}{tail}")
    md = getattr(mesh, "metadata", None) or {}
    if md.get("khr_texture_transform"):
        log(f"[mesh_color] mesh.metadata KHR_texture_transform={md['khr_texture_transform']!r}")

# Blender/OBJ exports often match SuperSplat / 3DGS viewers after a fixed +X 180°
# on positions + normals (row vectors: ``p' = p @ R.T``).  See ``write_axis_rotation_variant_plys``.
def _rx180_matrix() -> np.ndarray:
    from scipy.spatial.transform import Rotation as SciRotation

    return SciRotation.from_euler("x", 180.0, degrees=True).as_matrix()


def _apply_rowvec_rotation(
    points: np.ndarray, normals: np.ndarray | None, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    P = np.asarray(points, dtype=np.float64)
    Pn = (P @ R.T).astype(points.dtype, copy=False)
    if normals is None:
        return Pn, None
    N = np.asarray(normals, dtype=np.float64)
    Nn = (N @ R.T).astype(normals.dtype, copy=False)
    return Pn, Nn


def _rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) / 255.0 - 0.5) / C0


def _apply_khr_texture_transform_uv(uv: np.ndarray, khr: dict) -> np.ndarray:
    """Apply ``KHR_texture_transform``: scale, CCW rotation, translation (column-vector semantics)."""
    o = khr.get("offset") or [0.0, 0.0]
    ox, oy = float(o[0]), float(o[1])
    th = float(khr.get("rotation") or 0.0)
    sc = khr.get("scale") or [1.0, 1.0]
    sx, sy = float(sc[0]), float(sc[1])
    c, sn = np.cos(th), np.sin(th)
    p = np.asarray(uv, dtype=np.float64)
    u = p[:, 0] * sx
    v = p[:, 1] * sy
    u2 = c * u - sn * v + ox
    v2 = sn * u + c * v + oy
    return np.stack([u2, v2], axis=1)


def _bilinear_sample_rgb(img: np.ndarray, uf: np.ndarray, vf: np.ndarray) -> np.ndarray:
    """``img`` (H,W,3) uint8; ``uf``/``vf`` float pixel coords in ``[0, w-1]`` / ``[0, h-1]``."""
    h, w = img.shape[:2]
    uf = np.clip(uf, 0.0, float(w - 1) - 1e-6)
    vf = np.clip(vf, 0.0, float(h - 1) - 1e-6)
    x0 = np.floor(uf).astype(np.int64)
    y0 = np.floor(vf).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (uf - x0.astype(np.float64)).astype(np.float64)
    wy = (vf - y0.astype(np.float64)).astype(np.float64)
    c00 = img[y0, x0].astype(np.float64)
    c10 = img[y0, x1].astype(np.float64)
    c01 = img[y1, x0].astype(np.float64)
    c11 = img[y1, x1].astype(np.float64)
    a0 = c00 * (1.0 - wx[:, None]) + c10 * wx[:, None]
    a1 = c01 * (1.0 - wx[:, None]) + c11 * wx[:, None]
    out = a0 * (1.0 - wy[:, None]) + a1 * wy[:, None]
    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def _write_debug_colored_points_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    """ASCII-ish debug PLY: ``x,y,z`` + ``red,green,blue`` (uchar) for UV/visual checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    v = np.empty(len(points), dtype=dtype)
    v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def _sample_texture_colours(mesh, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Sample RGB from UV × texture image (OBJ or glTF). Fallback yellow if missing."""
    import trimesh

    out = _try_sample_uv_texture_rgb(mesh, points, face_indices)
    if out is not None:
        return out
    return np.full((len(points), 3), [255, 220, 20], dtype=np.uint8)


def _try_sample_uv_texture_rgb(
    mesh: "trimesh.Trimesh", points: np.ndarray, face_indices: np.ndarray
) -> np.ndarray | None:
    """Barycentric-interpolate corner UVs on the hit triangle, then bilinear texture sample.

    * ``MESH_GAUSSIAN_UV_INTERP``: ``barycentric`` (default) per glTF; ``corner`` uses
      only the dominant barycentric corner (debug / seam experiments).
    * ``MESH_GAUSSIAN_UV_FLIP_V``: unset or ``0`` = **no** vertical flip (default; many
      **Sketchfab** GLBs match Blender this way). Set ``1`` (or ``true``) to sample with
      ``(1-v)`` if the texture looks **vertically mirrored** on the mesh.
    * ``MESH_GAUSSIAN_UV_WRAP``: ``0`` (default) clamps UV to ``[0,1]``. Set ``1`` to use
      fractional repeat (``u-floor(u)``) for REPEAT-style atlases only.
    * ``KHR_texture_transform`` on the mesh (from glTF) is read from ``mesh.metadata``.
    """
    import trimesh

    visual = mesh.visual
    uv_raw = getattr(visual, "uv", None)
    if uv_raw is None:
        return None

    image = _texture_image_from_visual(visual)
    if image is None:
        return None

    faces = mesh.faces[face_indices]
    triangles = mesh.vertices[faces]
    bary = trimesh.triangles.points_to_barycentric(triangles, points)
    uv_arr = np.asarray(uv_raw, dtype=np.float64)
    nvert = len(mesh.vertices)
    nface = len(mesh.faces)

    interp = os.environ.get("MESH_GAUSSIAN_UV_INTERP", "barycentric").strip().lower()
    if uv_arr.shape[0] == nvert:
        face_uv = uv_arr[faces]
        if interp == "corner":
            k = np.argmax(bary, axis=1)
            ii = np.arange(len(face_indices), dtype=np.int64)
            uv_interp = uv_arr[faces[ii, k]]
        else:
            uv_interp = (face_uv * bary[:, :, None]).sum(axis=1)
    elif uv_arr.shape[0] == 3 * nface:
        row = (3 * face_indices[:, None] + np.arange(3, dtype=np.int64)).ravel()
        corner_uv = uv_arr[row].reshape(-1, 3, 2)
        if interp == "corner":
            k = np.argmax(bary, axis=1)
            uv_interp = corner_uv[np.arange(len(face_indices)), k]
        else:
            uv_interp = (corner_uv * bary[:, :, None]).sum(axis=1)
    else:
        return None

    md = getattr(mesh, "metadata", None) or {}
    khr = md.get("khr_texture_transform")
    if isinstance(khr, dict) and khr:
        uv_interp = _apply_khr_texture_transform_uv(uv_interp, khr)

    if os.environ.get("MESH_GAUSSIAN_UV_WRAP", "0").strip() == "1":
        uv_interp = uv_interp - np.floor(uv_interp)
    uv_interp = np.clip(uv_interp, 0.0, 1.0)

    pil = image.convert("RGB")
    img = np.asarray(pil)
    h, w = img.shape[:2]
    u_tex = uv_interp[:, 0].astype(np.float64) * float(max(w - 1, 0))
    _flip_raw = (os.environ.get("MESH_GAUSSIAN_UV_FLIP_V") or "").strip().lower()
    flip_v = _flip_raw in ("1", "true", "yes")
    v_tex = uv_interp[:, 1].astype(np.float64)
    if flip_v:
        v_pix = (1.0 - v_tex) * float(max(h - 1, 0))
    else:
        v_pix = v_tex * float(max(h - 1, 0))
    return _bilinear_sample_rgb(img, u_tex, v_pix)


def _try_vertex_color_rgb(
    mesh: "trimesh.Trimesh", points: np.ndarray, face_indices: np.ndarray
) -> np.ndarray | None:
    """Barycentric-interpolated vertex RGBA → RGB, or ``None``."""
    import trimesh

    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is None:
        return None
    c = np.asarray(vc)
    if c.ndim != 2 or c.shape[1] < 3 or c.shape[0] != len(mesh.vertices):
        return None
    faces = mesh.faces[face_indices]
    tri_c = c[faces][:, :, :3].astype(np.float64)
    triangles = mesh.vertices[faces]
    bary = trimesh.triangles.points_to_barycentric(triangles, points)
    rgb = (tri_c * bary[..., None]).sum(axis=1)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _try_material_basecolor_rgb(mesh: "trimesh.Trimesh", n: int) -> np.ndarray | None:
    """Solid RGB from glTF ``baseColorFactor`` / simple diffuse, or ``None``."""
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return None
    factor = getattr(mat, "baseColorFactor", None)
    if factor is not None:
        f = np.asarray(factor, dtype=np.float64).ravel()[:3]
        rgb = np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.broadcast_to(rgb, (n, 3)).copy()
    diffuse = getattr(mat, "diffuse", None)
    if diffuse is not None:
        d = np.asarray(diffuse, dtype=np.float64).ravel()[:3]
        rgb = np.clip(d * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.broadcast_to(rgb, (n, 3)).copy()
    return None


def _sample_surface_point_rgb(
    mesh: "trimesh.Trimesh", points: np.ndarray, face_indices: np.ndarray
) -> tuple[np.ndarray, str]:
    """Per-point RGB: vertex colors → UV texture → material base color → gray."""
    n = len(points)
    out = _try_vertex_color_rgb(mesh, points, face_indices)
    if out is not None:
        return out, "vertex_colors"
    out = _try_sample_uv_texture_rgb(mesh, points, face_indices)
    if out is not None:
        return out, "uv_texture"
    out = _try_material_basecolor_rgb(mesh, n)
    if out is not None:
        return out, "material_base_or_diffuse"
    return np.full((n, 3), 128, dtype=np.uint8), "fallback_gray_128"


def _allocate_surface_sample_counts(weights: np.ndarray, num_points: int) -> np.ndarray:
    """Integer counts proportional to ``weights`` (non-negative), summing to ``num_points``."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.zeros(len(w), dtype=np.int64)
    w = w / s
    raw = num_points * w
    counts = np.floor(raw).astype(np.int64)
    for i in range(len(w)):
        if w[i] >= 0.005 and counts[i] == 0 and num_points >= len(w):
            counts[i] = 1
    deficit = int(num_points - counts.sum())
    if deficit > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        j = 0
        while deficit > 0:
            counts[int(order[j % len(order)])] += 1
            deficit -= 1
            j += 1
    elif deficit < 0:
        while deficit < 0:
            j = int(np.argmax(counts))
            if counts[j] > 1:
                counts[j] -= 1
                deficit += 1
            else:
                break
    return counts


def _sample_multi_mesh_surfaces(
    meshes: list["trimesh.Trimesh"],
    num_points: int,
    seed: int,
    log: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Sample surface points from several meshes with area-weighted counts."""
    import trimesh

    areas = np.array([max(float(m.area), 1e-12) for m in meshes], dtype=np.float64)
    counts = _allocate_surface_sample_counts(areas, int(num_points))
    all_pts: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    tags: list[str] = []

    for idx, (mesh, n_samp) in enumerate(zip(meshes, counts)):
        if int(n_samp) <= 0:
            continue
        np.random.seed(seed + 10007 * idx)
        p, fi = trimesh.sample.sample_surface(mesh, int(n_samp))
        nrm = mesh.face_normals[fi]
        col, tag = _sample_surface_point_rgb(mesh, p, fi)
        lab = f"prim{idx}:{tag}"
        all_pts.append(p)
        all_n.append(nrm)
        all_c.append(col)
        tags.extend([lab] * len(p))
        if log:
            log(
                f"[mesh_color] submesh[{idx}] samples={len(p)} area_w={areas[idx]/areas.sum():.4f}  "
                f"source={tag}  rgb_mean={tuple(float(x) for x in col.mean(axis=0))}"
            )

    if not all_pts:
        raise ValueError("_sample_multi_mesh_surfaces: zero samples (empty meshes?)")
    P = np.vstack(all_pts)
    N = np.vstack(all_n)
    C = np.vstack(all_c)
    if log:
        dist = Counter(tags)
        log(f"[mesh_color] sampled_rgb source distribution: {dict(dist)}")
    return P, N, C, tags


def _swap_red_yellow_rgb(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    r, g, b = out[:, 0], out[:, 1], out[:, 2]
    dark = (r < 45) & (g < 45) & (b < 45)
    yellowish = (r > 120) & (g > 85) & (b < 100) & ~dark
    reddish = (r > 100) & ~yellowish & ~dark
    out[reddish] = np.array([255, 215, 20], dtype=np.uint8)
    out[yellowish] = np.array([225, 25, 20], dtype=np.uint8)
    return out


def _stylize_yellow_duck_rgb(rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Turn noisy sampled texture into a clean yellow duck palette."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Texture from DG's mesh export can be very noisy.  Use geometry instead:
    # the beak is the forward protruding low-x portion in the exported duck mesh.
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())
    w, h, d = xmax - xmin + 1e-8, ymax - ymin + 1e-8, zmax - zmin + 1e-8
    beak = (
        (x < xmin + 0.28 * w)
        & (y > ymin + 0.42 * h)
        & (y < ymin + 0.82 * h)
        & (z > zmin + 0.25 * d)
        & (z < zmin + 0.85 * d)
    )

    out = np.full_like(rgb, [255, 215, 20], dtype=np.uint8)
    out[beak] = np.array([225, 25, 20], dtype=np.uint8)
    return out


def mesh_to_gaussian_ply(
    mesh_path: Path | str,
    out_ply: Path | str,
    *,
    num_points: int = 30000,
    scale_logit: float = -4.55,
    opacity_logit: float = 3.5,
    swap_red_yellow: bool = False,
    stylize_yellow_duck: bool = False,
    seed: int = 0,
    apply_export_axis_rotation: bool = True,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Sample a mesh surface into simple isotropic Gaussian splats.

    The output contains the same core properties DreamGaussian stage-1 PLYs use:
    position, normal, SH DC colour, opacity, scale, and quaternion rotation.  It
    is intentionally conservative: isotropic splats on the mesh surface produce
    a much cleaner silhouette than DreamGaussian's noisy stage-1 optimisation
    blobs, while still remaining compatible with the existing merge/physics path.

    By default applies a **+X axis 180°** rotation to sampled ``x,y,z`` and ``nx,ny,nz``
    so OBJ/Blender frames align with SuperSplat / downstream ``canonical_frame=mesh``
    rescaling (no extra DG text/image preset on top of this).
    Pass ``apply_export_axis_rotation=False`` to write raw OBJ coordinates.

    ``log`` receives human-readable diagnostics (mesh path, Scene vs Trimesh, UV,
    material, RGB stats, colour source branch).
    """
    import trimesh

    mesh_path = Path(mesh_path)
    out_ply = Path(out_ply)

    if mesh_path.suffix.lower() == ".glb":
        glb_meshes = load_glb_colored_primitive_meshes(mesh_path, log=log)
        if glb_meshes:
            for i, sub in enumerate(glb_meshes):
                if log:
                    log(f"[mesh_color] --- GLB colored primitive {i} (by area, largest first) ---")
                log_mesh_visual_diagnostics(sub, log)
            np.random.seed(seed)
            points, normals, colours, _tags = _sample_multi_mesh_surfaces(
                glb_meshes, int(num_points), seed, log
            )
            color_source = "multi_primitive_gltf"
        else:
            mesh = load_insertion_mesh(mesh_path, log=log)
            log_mesh_visual_diagnostics(mesh, log)
            np.random.seed(seed)
            points, face_indices = trimesh.sample.sample_surface(mesh, int(num_points))
            normals = mesh.face_normals[face_indices]
            colours, color_source = _sample_surface_point_rgb(mesh, points, face_indices)
    else:
        mesh = load_insertion_mesh(mesh_path, log=log)
        log_mesh_visual_diagnostics(mesh, log)
        np.random.seed(seed)
        points, face_indices = trimesh.sample.sample_surface(mesh, int(num_points))
        normals = mesh.face_normals[face_indices]
        colours, color_source = _sample_surface_point_rgb(mesh, points, face_indices)

    if log:
        log(
            f"[mesh_color] sampled_rgb aggregate source={color_source}  "
            f"min={tuple(int(x) for x in colours.min(axis=0))}  "
            f"mean={tuple(float(x) for x in colours.mean(axis=0))}  "
            f"max={tuple(int(x) for x in colours.max(axis=0))}"
        )
    if stylize_yellow_duck:
        colours = _stylize_yellow_duck_rgb(colours, points)
    elif swap_red_yellow:
        colours = _swap_red_yellow_rgb(colours)
    if log and (stylize_yellow_duck or swap_red_yellow):
        log(
            f"[mesh_color] final_rgb (into f_dc_)  "
            f"min={tuple(int(x) for x in colours.min(axis=0))}  "
            f"mean={tuple(float(x) for x in colours.mean(axis=0))}  "
            f"max={tuple(int(x) for x in colours.max(axis=0))}"
        )

    if apply_export_axis_rotation:
        Rm = _rx180_matrix()
        points, normals = _apply_rowvec_rotation(points, normals, Rm)

    dbg = os.environ.get("MESH_GAUSSIAN_DEBUG_UV_POINTS", "").strip()
    if dbg:
        dbg_path = Path(dbg)
        _write_debug_colored_points_ply(dbg_path, points, colours)
        if log:
            log(f"[mesh_color] debug UV surface points PLY (xyz+rgb, same frame as output): {dbg_path.resolve()}")

    sh = _rgb_to_sh(colours)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    v = np.zeros(len(points), dtype=dtype)
    v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
    v["nx"], v["ny"], v["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = sh[:, 0], sh[:, 1], sh[:, 2]
    v["opacity"] = float(opacity_logit)
    v["scale_0"] = v["scale_1"] = v["scale_2"] = float(scale_logit)
    # 3DGS / SuperSplat: Gaussian rotation as normalized quaternion ``w, x, y, z``
    # (``rot_0`` … ``rot_3``) — same as ``submodules/gaussian-splatting`` PLY I/O.
    v["rot_0"] = 1.0
    v["rot_1"] = 0.0
    v["rot_2"] = 0.0
    v["rot_3"] = 0.0

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(out_ply))
    return out_ply


def write_superplat_viewer_test_ply(
    in_ply: Path | str,
    out_ply: Path | str,
    *,
    scale_logit: float | None = None,
) -> Path:
    """Write a copy of a 3DGS PLY with **identity rotation** and **isotropic log-scales**.

    Use to check whether a viewer (e.g. SuperSplat) vs. Blender orientation issues
    come from per-splat ``rot_*`` / anisotropic ``scale_*`` rather than vertex ``x,y,z``.

    * ``scale_*`` in training PLYs are **log-space**; rasterizers apply ``exp``.
    * Quaternions are **w, x, y, z** in ``rot_0`` … ``rot_3``.
    """
    in_ply, out_ply = Path(in_ply), Path(out_ply)
    ply = PlyData.read(str(in_ply))
    el = ply["vertex"]
    names = el.data.dtype.names
    if names is None:
        raise ValueError(f"No vertex fields in {in_ply}")
    v = np.array(el.data, copy=True)
    for k in range(4):
        rk = f"rot_{k}"
        if rk in names:
            v[rk] = 1.0 if k == 0 else 0.0

    s_triplet = scale_logit
    if s_triplet is None and all(f"scale_{j}" in names for j in range(3)):
        s_triplet = float(
            np.median(np.concatenate([v[f"scale_{j}"].astype(np.float64) for j in range(3)]))
        )
    if s_triplet is None:
        s_triplet = -4.35
    for j in range(3):
        sj = f"scale_{j}"
        if sj in names:
            v[sj] = float(s_triplet)

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(out_ply))
    return out_ply


def write_axis_rotation_variant_plys(
    in_ply: Path | str,
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Copy a 3DGS PLY, rotating only ``x,y,z`` (and ``nx,ny,nz`` if present) by fixed axis angles.

    ``rot_*`` / ``scale_*`` / SH are unchanged — for isotropic identity-rot splats this
    matches “only move centres + normals” for SuperSplat axis-convention experiments.

    Note: :func:`mesh_to_gaussian_ply` already bakes **one +X 180°** by default; the
    ``*_rx180.ply`` variant here is an **additional** 180° on top of ``in_ply``.

    Row-vector convention (same as :func:`rescale_object_ply_to_scene`): for each row
    ``p`` of shape ``(3,)``, ``p_new = p @ R.T`` with ``R`` the active rotation matrix
    from ``scipy.spatial.transform.Rotation``.

    Writes (under ``out_dir`` or next to ``in_ply``):

    * ``object_mesh_gaussians_rx180.ply`` — +X axis 180°
    * ``object_mesh_gaussians_ry180.ply`` — +Y axis 180°
    * ``object_mesh_gaussians_rz180.ply`` — +Z axis 180°
    * ``object_mesh_gaussians_rx90.ply`` — +X axis +90°
    * ``object_mesh_gaussians_rxm90.ply`` — +X axis −90°
    """
    from scipy.spatial.transform import Rotation as SciRotation

    in_ply = Path(in_ply)
    out_dir = Path(out_dir) if out_dir is not None else in_ply.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, str, float]] = [
        ("object_mesh_gaussians_rx180.ply", "x", 180.0),
        ("object_mesh_gaussians_ry180.ply", "y", 180.0),
        ("object_mesh_gaussians_rz180.ply", "z", 180.0),
        ("object_mesh_gaussians_rx90.ply", "x", 90.0),
        ("object_mesh_gaussians_rxm90.ply", "x", -90.0),
    ]

    ply = PlyData.read(str(in_ply))
    el = ply["vertex"]
    names = el.data.dtype.names
    if names is None:
        raise ValueError(f"No vertex fields in {in_ply}")
    base = np.array(el.data, copy=True)
    if not all(c in names for c in ("x", "y", "z")):
        raise ValueError(f"{in_ply} missing x/y/z")

    def _apply_R(v: np.ndarray, Rm: np.ndarray) -> None:
        P = np.stack([v["x"].astype(np.float64), v["y"].astype(np.float64), v["z"].astype(np.float64)], axis=1)
        Pn = P @ Rm.T
        v["x"] = Pn[:, 0].astype(v["x"].dtype)
        v["y"] = Pn[:, 1].astype(v["y"].dtype)
        v["z"] = Pn[:, 2].astype(v["z"].dtype)
        if all(c in names for c in ("nx", "ny", "nz")):
            N = np.stack([v["nx"].astype(np.float64), v["ny"].astype(np.float64), v["nz"].astype(np.float64)], axis=1)
            Nn = N @ Rm.T
            v["nx"] = Nn[:, 0].astype(v["nx"].dtype)
            v["ny"] = Nn[:, 1].astype(v["ny"].dtype)
            v["nz"] = Nn[:, 2].astype(v["nz"].dtype)

    out_paths: dict[str, Path] = {}
    for fname, axis, deg in specs:
        v = np.array(base, copy=True)
        Rm = SciRotation.from_euler(axis, deg, degrees=True).as_matrix()
        _apply_R(v, Rm)
        outp = out_dir / fname
        PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(outp))
        out_paths[fname] = outp
    return out_paths
