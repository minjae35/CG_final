"""Build ``trimesh.Trimesh`` from GLB using ``pygltflib`` so PBR textures survive.

``trimesh.load`` often drops ``baseColorTexture`` / buffer-backed images on Sketchfab
and similar GLBs; Blender still renders them.  This module reads accessors + images
from the binary chunk and attaches ``TextureVisuals`` / ``ColorVisuals`` aligned with
the decoded vertex order.

Multi-primitive GLBs (body / beak / eyes as separate primitives) are returned as
**one Trimesh per primitive** for Gaussian sampling so each face keeps the correct
material/UV; ``merge_glb_primitive_geometries`` concatenates geometry only (no
textures) for placement export.
"""
from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# pygltflib is optional at import time; load_insertion_mesh checks suffix first.
try:
    from pygltflib import GLTF2
except ImportError:  # pragma: no cover
    GLTF2 = None  # type: ignore[misc, assignment]

_COMPONENT_NP = {
    5120: np.int8,  # BYTE
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


def _dtype_nbytes(dtype: np.dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def _decode_accessor(gltf: Any, blob: bytes, accessor_index: int) -> np.ndarray:
    acc = gltf.accessors[accessor_index]
    if getattr(acc, "sparse", None) is not None:
        raise ValueError("Sparse accessors are not supported")
    bv_idx = acc.bufferView
    if bv_idx is None:
        raise ValueError("Accessor has no bufferView")
    bv = gltf.bufferViews[bv_idx]
    base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = int(acc.count)
    ncomp = _TYPE_COMPONENTS[str(acc.type)]
    dtype = _COMPONENT_NP[int(acc.componentType)]
    el_bytes = _dtype_nbytes(dtype) * ncomp
    stride = int(bv.byteStride) if bv.byteStride else el_bytes
    out = np.empty((count, ncomp), dtype=dtype)
    for i in range(count):
        off = base + i * stride
        chunk = blob[off : off + el_bytes]
        if len(chunk) < el_bytes:
            raise ValueError("Buffer underrun while decoding accessor")
        out[i] = np.frombuffer(chunk, dtype=dtype, count=ncomp)
    if not acc.normalized:
        return out
    fn = out.astype(np.float32)
    if np.issubdtype(dtype, np.unsignedinteger):
        return fn / float(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.signedinteger):
        neg = fn < 0.0
        r = np.zeros_like(fn)
        r[neg] = fn[neg] / float(-np.iinfo(dtype).min)
        r[~neg] = fn[~neg] / float(np.iinfo(dtype).max)
        return np.clip(r, -1.0, 1.0)
    return fn


def _decode_accessor_scalar_indices(gltf: Any, blob: bytes, accessor_index: int) -> np.ndarray:
    """Return flat indices as int64 (triangle list indices)."""
    acc = gltf.accessors[accessor_index]
    arr = _decode_accessor(gltf, blob, accessor_index)
    if str(acc.type) != "SCALAR":
        raise ValueError("Expected SCALAR indices")
    return arr.astype(np.int64, copy=False).ravel()


def _node_matrix_4x4(node: Any) -> np.ndarray:
    if getattr(node, "matrix", None) is not None and node.matrix is not None and len(node.matrix) == 16:
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4, order="F")
    from scipy.spatial.transform import Rotation as SciRotation

    T = np.eye(4, dtype=np.float64)
    Rm = np.eye(4, dtype=np.float64)
    Sm = np.eye(4, dtype=np.float64)
    if node.translation is not None:
        T[0, 3], T[1, 3], T[2, 3] = (float(x) for x in node.translation)
    if node.rotation is not None:
        q = np.asarray(node.rotation, dtype=np.float64).ravel()
        Rm[:3, :3] = SciRotation.from_quat(q).as_matrix()
    if node.scale is not None:
        sx, sy, sz = (float(x) for x in node.scale)
        Sm[0, 0], Sm[1, 1], Sm[2, 2] = sx, sy, sz
    return T @ Rm @ Sm


def _iter_node_mesh_primitives(
    gltf: Any, node_index: int, world: np.ndarray
) -> Iterator[tuple[np.ndarray, int, int, Any]]:
    """Yield ``(world_matrix, mesh_index, prim_index, primitive)`` depth-first."""
    node = gltf.nodes[node_index]
    local = _node_matrix_4x4(node)
    W = world @ local
    if node.mesh is not None:
        mesh = gltf.meshes[int(node.mesh)]
        for pi, prim in enumerate(mesh.primitives):
            yield W, int(node.mesh), pi, prim
    for ch in node.children or []:
        yield from _iter_node_mesh_primitives(gltf, int(ch), W)


def _primitive_has_draco(prim: Any) -> bool:
    ext = getattr(prim, "extensions", None)
    if not ext:
        return False
    if isinstance(ext, dict):
        return "KHR_draco_mesh_compression" in ext
    return getattr(ext, "KHR_draco_mesh_compression", None) is not None


def _image_to_pil(gltf: Any, blob: bytes, image_index: int) -> Image.Image:
    img = gltf.images[int(image_index)]
    if getattr(img, "bufferView", None) is not None:
        bv = gltf.bufferViews[int(img.bufferView)]
        start = int(bv.byteOffset or 0)
        end = start + int(bv.byteLength)
        return Image.open(io.BytesIO(blob[start:end])).convert("RGB")
    if getattr(img, "uri", None):
        uri = str(img.uri)
        if uri.startswith("data:"):
            _, b64 = uri.split(",", 1)
            import base64

            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        raise ValueError(f"External image URI not supported in GLB: {uri[:80]}")
    raise ValueError("Image has no bufferView or uri")


def _triangle_faces_from_primitive(
    gltf: Any, blob: bytes, prim: Any, n_pos: int
) -> np.ndarray:
    mode = int(prim.mode) if prim.mode is not None else 4
    if mode != 4:
        raise ValueError(f"Unsupported primitive mode {mode} (need TRIANGLES)")
    if prim.indices is None:
        if n_pos % 3 != 0:
            raise ValueError("Non-indexed mesh vertex count not divisible by 3")
        return np.arange(n_pos, dtype=np.int64).reshape(-1, 3)
    idx = _decode_accessor_scalar_indices(gltf, blob, int(prim.indices))
    if len(idx) % 3 != 0:
        raise ValueError("Index count not divisible by 3")
    return idx.reshape(-1, 3)


def _transform_positions(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply 4x4 ``W`` (glTF column convention) to row-vector positions ``V`` (N,3)."""
    if V.size == 0:
        return V
    Ph = np.concatenate([V.astype(np.float64), np.ones((len(V), 1), dtype=np.float64)], axis=1)
    out = (W @ Ph.T).T[:, :3]
    return out.astype(np.float32, copy=False)


def _primitive_surface_area(V: np.ndarray, F: np.ndarray) -> float:
    if len(F) == 0:
        return 0.0
    tri = V[F.astype(np.int64)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _basecolor_texcoord_set(prim: Any, gltf: Any) -> int:
    """glTF ``TextureInfo.texCoord`` (default 0) selects ``TEXCOORD_n``."""
    if prim.material is None:
        return 0
    mat = gltf.materials[int(prim.material)]
    pbr = getattr(mat, "pbrMetallicRoughness", None)
    if pbr is None:
        return 0
    bct = getattr(pbr, "baseColorTexture", None)
    if bct is None or getattr(bct, "index", None) is None:
        return 0
    tc = getattr(bct, "texCoord", None)
    return int(tc) if tc is not None else 0


def _read_khr_texture_transform(gltf: Any, prim: Any) -> dict[str, Any] | None:
    """Return ``KHR_texture_transform`` dict from ``baseColorTexture`` if present."""
    if prim.material is None:
        return None
    mat = gltf.materials[int(prim.material)]
    pbr = getattr(mat, "pbrMetallicRoughness", None)
    if pbr is None:
        return None
    bct = getattr(pbr, "baseColorTexture", None)
    if bct is None:
        return None
    ext = getattr(bct, "extensions", None)
    if not ext:
        return None
    data: Any = None
    if isinstance(ext, dict):
        data = ext.get("KHR_texture_transform")
    else:
        data = getattr(ext, "KHR_texture_transform", None)
    if data is None:
        return None
    out: dict[str, Any] = {}
    for key in ("offset", "rotation", "scale"):
        if isinstance(data, dict):
            if key in data:
                out[key] = data[key]
        elif hasattr(data, key):
            val = getattr(data, key)
            if val is not None:
                out[key] = val
    return out or None


def _maybe_duplicate_vertices_for_per_corner_uv(
    V: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray | None,
    log: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """When ``TEXCOORD_0`` has ``3 * n_faces`` rows but POSITION is indexed (fewer verts), duplicate corners so UV rows align with ``mesh.faces`` order (glTF-style)."""
    if uv is None or uv.ndim != 2 or uv.shape[1] != 2:
        return V, F, uv
    nf = len(F)
    if nf == 0:
        return V, F, uv
    nv = len(V)
    if uv.shape[0] != 3 * nf:
        return V, F, uv
    if nv == 3 * nf:
        return V, F, uv
    idx_flat = F.ravel()
    V_new = V[idx_flat].reshape(3 * nf, 3).astype(np.float32, copy=False)
    F_new = np.arange(3 * nf, dtype=np.int64).reshape(-1, 3)
    if log:
        log(
            f"[mesh_gltf] duplicated vertices for per-corner TEXCOORD_0: "
            f"verts {nv}→{len(V_new)}  faces={nf}  (UV rows={uv.shape[0]})"
        )
    return V_new, F_new, uv


def _material_debug_line(gltf: Any, prim: Any) -> str:
    if prim.material is None:
        return "material=None"
    mat = gltf.materials[int(prim.material)]
    name = getattr(mat, "name", None) or ""
    pbr = getattr(mat, "pbrMetallicRoughness", None)
    bcf = getattr(pbr, "baseColorFactor", None) if pbr else None
    has_tex = bool(
        pbr is not None
        and getattr(pbr, "baseColorTexture", None) is not None
        and getattr(pbr.baseColorTexture, "index", None) is not None
    )
    return f"material[{prim.material}] name={name!r} baseColorFactor={bcf!r} baseColorTexture={'yes' if has_tex else 'no'}"


def _decode_primitive(
    gltf: Any,
    blob: bytes,
    prim: Any,
    world: np.ndarray,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    Image.Image | None,
    np.ndarray | None,
    dict[str, Any] | None,
]:
    attrs = prim.attributes
    pos_idx = getattr(attrs, "POSITION", None)
    if pos_idx is None:
        raise ValueError("Primitive missing POSITION")
    V = _decode_accessor(gltf, blob, int(pos_idx)).astype(np.float32, copy=False)
    if V.shape[1] != 3:
        raise ValueError("POSITION must be VEC3")
    V = _transform_positions(V, world)
    F = _triangle_faces_from_primitive(gltf, blob, prim, len(V))

    color = None
    cidx = getattr(attrs, "COLOR_0", None)
    if cidx is not None:
        color = _decode_accessor(gltf, blob, int(cidx))
        if color.ndim != 2 or color.shape[0] != len(V):
            color = None
        elif color.shape[1] < 3:
            color = None

    uv = None
    tc_set = _basecolor_texcoord_set(prim, gltf)
    tidx = getattr(attrs, f"TEXCOORD_{tc_set}", None)
    if tidx is None and tc_set != 0:
        tidx = getattr(attrs, "TEXCOORD_0", None)
    if tidx is not None:
        uv = _decode_accessor(gltf, blob, int(tidx)).astype(np.float64, copy=False)
        if uv.shape[1] != 2 or uv.shape[0] not in (len(V), 3 * len(F)):
            uv = None

    pil_img = None
    base_rgba = None
    if prim.material is not None:
        mat = gltf.materials[int(prim.material)]
        pbr = getattr(mat, "pbrMetallicRoughness", None)
        if pbr is not None:
            if getattr(pbr, "baseColorFactor", None) is not None:
                base_rgba = np.asarray(pbr.baseColorFactor, dtype=np.float64).ravel()[:4]
            bct = getattr(pbr, "baseColorTexture", None)
            if bct is not None and getattr(bct, "index", None) is not None:
                tex = gltf.textures[int(bct.index)]
                src = int(tex.source) if tex.source is not None else None
                if src is not None:
                    try:
                        pil_img = _image_to_pil(gltf, blob, src)
                    except Exception:
                        pil_img = None
    khr_tt = _read_khr_texture_transform(gltf, prim)
    nf = len(F)
    if (
        uv is not None
        and uv.ndim == 2
        and uv.shape[0] == 3 * nf
        and len(V) != 3 * nf
        and nf > 0
    ):
        if color is not None and color.shape[0] == len(V):
            color = color[F.ravel()].reshape(3 * nf, color.shape[1])
        elif color is not None:
            color = None
        V, F, uv = _maybe_duplicate_vertices_for_per_corner_uv(V, F, uv, log)
    return V, F, color, uv, pil_img, base_rgba, khr_tt


def _row_has_color(
    color: np.ndarray | None,
    uv: np.ndarray | None,
    pil_img: Image.Image | None,
    base_rgba: np.ndarray | None,
) -> bool:
    return bool(
        color is not None
        or (uv is not None and pil_img is not None)
        or base_rgba is not None
    )


def _build_trimesh_from_primitive_parts(
    V: np.ndarray,
    F: np.ndarray,
    color: np.ndarray | None,
    uv: np.ndarray | None,
    pil_img: Image.Image | None,
    base_rgba: np.ndarray | None,
    khr_transform: dict[str, Any] | None = None,
) -> Any:
    import trimesh
    from trimesh.visual import ColorVisuals, TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    visual: Any
    if color is not None:
        c = color.astype(np.float64)
        if c.max() <= 1.0 + 1e-6:
            c = np.clip(c * 255.0, 0.0, 255.0)
        rgba = np.zeros((len(V), 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(c[:, :3], 0.0, 255.0).astype(np.uint8)
        rgba[:, 3] = 255
        if color.shape[1] >= 4:
            a = c[:, 3]
            if float(np.max(a)) <= 1.0 + 1e-6:
                rgba[:, 3] = np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                rgba[:, 3] = np.clip(a, 0.0, 255.0).astype(np.uint8)
        visual = ColorVisuals(vertex_colors=rgba)
    elif uv is not None and pil_img is not None:
        mat = SimpleMaterial(image=pil_img)
        visual = TextureVisuals(uv=uv, material=mat)
    elif base_rgba is not None:
        rgb_u8 = np.clip(base_rgba[:3], 0.0, 1.0) * 255.0
        rgb_u8 = np.clip(rgb_u8, 0.0, 255.0).astype(np.uint8).reshape(1, 3)
        a = 255 if len(base_rgba) < 4 else int(np.clip(base_rgba[3], 0.0, 1.0) * 255.0)
        rgba = np.zeros((len(V), 4), dtype=np.uint8)
        rgba[:, :3] = np.broadcast_to(rgb_u8, (len(V), 3)).copy()
        rgba[:, 3] = a
        visual = ColorVisuals(vertex_colors=rgba)
    else:
        visual = ColorVisuals()

    mesh = trimesh.Trimesh(vertices=V, faces=F, visual=visual, process=False)
    if khr_transform:
        md = dict(mesh.metadata) if getattr(mesh, "metadata", None) else {}
        md["khr_texture_transform"] = khr_transform
        mesh.metadata = md
    return mesh


def log_glb_primitive_inventory(
    gltf: Any,
    blob: bytes,
    log: Callable[[str], None] | None,
    *,
    root_nodes: list[int],
) -> None:
    """Decode each primitive once for logging (material / counts / area)."""
    if log is None:
        return
    n_mat = len(gltf.materials or [])
    log(f"[mesh_gltf] materials in file: {n_mat}")
    prim_i = 0
    for root in root_nodes:
        for W, mi, pi, prim in _iter_node_mesh_primitives(gltf, int(root), np.eye(4, dtype=np.float64)):
            if _primitive_has_draco(prim):
                log(f"[mesh_gltf] primitive[{prim_i}] mesh={mi} prim={pi}  Draco — skipped in inventory")
                prim_i += 1
                continue
            try:
                V, F, _c, uv, pil, bcf, _khr = _decode_primitive(gltf, blob, prim, W, log=log)
                area = _primitive_surface_area(V, F)
                uv_s = "none" if uv is None else str(tuple(uv.shape))
                tex = "yes" if pil is not None else "no"
                mline = _material_debug_line(gltf, prim)
                log(
                    f"[mesh_gltf] primitive[{prim_i}] mesh={mi} prim={pi}  "
                    f"verts={len(V)} faces={len(F)} area={area:.6f}  uv={uv_s} texture={tex}  {mline}"
                )
            except Exception as exc:
                log(f"[mesh_gltf] primitive[{prim_i}] mesh={mi} prim={pi}  inventory decode error: {exc!r}")
            prim_i += 1


def load_glb_colored_primitive_meshes(
    glb_path: Path | str,
    log: Callable[[str], None] | None = None,
) -> list[Any] | None:
    """Return **one Trimesh per colored primitive** (same order as glTF scene DFS)."""
    if GLTF2 is None:
        if log:
            log("[mesh_import] pygltflib not installed — cannot decode GLB PBR; pip install pygltflib")
        return None

    path = Path(glb_path)
    try:
        gltf = GLTF2.load_binary(str(path))
    except Exception as exc:
        if log:
            log(f"[mesh_import] pygltflib load_binary failed: {exc!r}")
        return None

    blob = gltf.binary_blob()
    if blob is None or len(blob) == 0:
        if log:
            log("[mesh_import] pygltflib: empty binary_blob()")
        return None

    scene_idx = 0 if gltf.scene is None else int(gltf.scene)
    if not gltf.scenes:
        if log:
            log("[mesh_import] pygltflib: glTF has no scenes")
        return None
    scene = gltf.scenes[scene_idx]
    root_nodes = list(scene.nodes or [])

    log_glb_primitive_inventory(gltf, blob, log, root_nodes=root_nodes)

    rows: list[
        tuple[
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            Image.Image | None,
            np.ndarray | None,
            dict[str, Any] | None,
        ]
    ] = []

    for root in root_nodes:
        for W, _mi, _pi, prim in _iter_node_mesh_primitives(gltf, int(root), np.eye(4, dtype=np.float64)):
            if _primitive_has_draco(prim):
                if log:
                    log("[mesh_import] pygltflib: skipping Draco-compressed primitive")
                continue
            try:
                V, F, color, uv, pil_img, base_rgba, khr_tt = _decode_primitive(
                    gltf, blob, prim, W, log=log
                )
            except Exception as exc:
                if log:
                    log(f"[mesh_import] pygltflib: skip primitive decode error: {exc!r}")
                continue
            if not _row_has_color(color, uv, pil_img, base_rgba):
                continue
            area = _primitive_surface_area(V, F)
            rows.append((area, V, F, color, uv, pil_img, base_rgba, khr_tt))

    if not rows:
        if log:
            log(
                "[mesh_import] pygltflib: no primitive with COLOR_0 / UV+texture / baseColorFactor — "
                "falling back to trimesh.load"
            )
        return None

    rows.sort(key=lambda t: t[0], reverse=True)
    meshes = [_build_trimesh_from_primitive_parts(*t[1:8]) for t in rows]
    for m in meshes:
        if m.is_empty:
            if log:
                log("[mesh_import] pygltflib: empty submesh after build — abort GLB path")
            return None

    if log:
        log(f"[mesh_import] pygltflib: {len(meshes)} colored primitive(s) (area-sorted largest first)")
    return meshes


def merge_glb_primitive_geometries(meshes: list[Any]) -> Any:
    """Concatenate triangle soups for OBJ placement (visual discarded)."""
    import trimesh

    if not meshes:
        raise ValueError("merge_glb_primitive_geometries: empty mesh list")
    if len(meshes) == 1:
        return trimesh.Trimesh(
            vertices=meshes[0].vertices, faces=meshes[0].faces, process=False, visual=None
        )
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    off = 0
    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.faces, dtype=np.int64)
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    V = np.vstack(verts)
    F = np.vstack(faces)
    return trimesh.Trimesh(vertices=V.astype(np.float32), faces=F, process=False, visual=None)


def load_trimesh_from_glb_pygltflib(
    glb_path: Path | str,
    log: Callable[[str], None] | None = None,
) -> Any | None:
    """Return **largest-area** colored primitive only (legacy single-mesh callers)."""
    meshes = load_glb_colored_primitive_meshes(glb_path, log=log)
    if not meshes:
        return None
    return meshes[0]
