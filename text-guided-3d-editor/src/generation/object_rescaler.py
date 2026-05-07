"""Scale generated object PLY into scene units (PRD 4.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as SciRotation


# DreamGaussian's two stage-1 backbones use DIFFERENT canonical frames:
#
#   * ``text``  (mvdream / text→3D)     : +Z is up, +Y is depth-forward.
#   * ``image`` (zero123 / image→3D)    : +Y is up, +Z is depth-forward.
#
# Mip-NeRF 360's "room" scene is COLMAP-style and +Y points DOWN, so to plant
# the object on the floor standing upright we must rotate the object's UP axis
# onto world -Y.  Below are the rotation matrices and matching quaternions
# (``w, x, y, z``) for each canonical frame.
#
# text-to-3D path: rotate -90° about world +X to send DG-+z → world -y.
_R_TEXT_TO_WORLD_Y_DOWN = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_Q_TEXT_TO_WORLD_Y_DOWN = np.array(
    [np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0], dtype=np.float64
)
# image-to-3D path: rotate 180° about world +X so DG-+y (head up) becomes
# world -y (scene up), DG-+z (camera-forward) becomes world -z (face the
# scene's primary camera).
_R_IMAGE_TO_WORLD_Y_DOWN = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_Q_IMAGE_TO_WORLD_Y_DOWN = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

# Imported meshes: :func:`generation.mesh_to_gaussian.mesh_to_gaussian_ply` already
# applies +X 180° from Blender/OBJ into the SuperSplat-friendly frame.  Use
# ``mesh`` so :func:`rescale_object_ply_to_scene` only scales / centres without
# stacking DreamGaussian ``text`` / ``image`` rigid presets on top.
_R_MESH_IDENTITY = np.eye(3, dtype=np.float64)
_Q_MESH_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

_REORIENT_PRESETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "text": (_R_TEXT_TO_WORLD_Y_DOWN, _Q_TEXT_TO_WORLD_Y_DOWN),
    "image": (_R_IMAGE_TO_WORLD_Y_DOWN, _Q_IMAGE_TO_WORLD_Y_DOWN),
    "mesh": (_R_MESH_IDENTITY, _Q_MESH_IDENTITY),
}


def _hamilton(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 — apply q2 first, then q1.

    q1: (4,)   wxyz
    q2: (N, 4) wxyz
    returns (N, 4) wxyz
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _rot_x(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_from_axis_angle(axis: tuple[float, float, float], deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    half = 0.5 * rad
    return np.array([np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)], dtype=np.float64)


def _hamilton_one(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return _hamilton(q1, q2[None, :])[0]


def _bbox_xyz(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    x, y, z = np.array(v["x"]), np.array(v["y"]), np.array(v["z"])
    if len(x) == 0:
        raise ValueError(
            f"{ply_path} has zero vertices — DreamGaussian likely failed or wrote an empty stage-1 PLY. "
            f"Check {ply_path.parent / 'dreamgaussian.stderr.log'}."
        )
    pts = np.stack([x, y, z], axis=1)
    return pts.min(0), pts.max(0)


def _quat_wxyz_from_scipy(rot: SciRotation) -> np.ndarray:
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def rescale_object_ply_to_scene(
    object_ply: Path | str,
    scene_ply: Path | str,
    scale_factor: float,
    out_ply: Path | str,
    opacity_filter_logit: float = -0.5,
    reorient_to_world_y_down: bool = True,
    spatial_outlier_sigma: float | None = 3.0,
    canonical_frame: str = "text",
    post_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_y_axis_to: np.ndarray | None = None,
    contact_base_occlusion: float = 0.0,
) -> Path:
    """Uniform scale from object bbox diagonal vs scene bbox diagonal * scale_factor.

    Also scales Gaussian log-scale values by log(s) so ellipsoid sizes stay
    proportional to the repositioned object, filters out low-opacity Gaussians
    (logit < opacity_filter_logit) that DreamGaussian stage-1 produces as
    background noise, and (when ``reorient_to_world_y_down`` is True) rotates
    the object so its canonical ``+z``-up axis maps to world ``-y`` so the duck
    stands upright in the room scene.

    If ``align_y_axis_to`` is a unit 3-vector, the duck's foot direction (world
    ``+Y`` after the canonical preset) is rotated so it matches that vector —
    typically the local rug normal from :func:`placement.surface_aware.floor_downward_normal_from_slopes`.

    ``contact_base_occlusion`` in ``[0, 1)`` scales down ``f_dc_*`` near the
    feet (high ``y`` before the final foot-plane shift) for a subtle contact-darkening hint.
    """
    object_ply, scene_ply, out_ply = Path(object_ply), Path(scene_ply), Path(out_ply)
    omin, omax = _bbox_xyz(object_ply)
    smin, smax = _bbox_xyz(scene_ply)
    diag_o = np.linalg.norm(omax - omin) + 1e-8
    diag_s = np.linalg.norm(smax - smin) + 1e-8
    s = scale_factor * (diag_s / diag_o)
    log_s = np.log(s + 1e-12)

    ply = PlyData.read(str(object_ply))
    v = ply["vertex"].data
    names = v.dtype.names
    arr = np.array(v.tolist(), dtype=np.float64)

    xi, yi, zi = names.index("x"), names.index("y"), names.index("z")

    # Filter out low-opacity noise Gaussians from DreamGaussian stage-1.
    if "opacity" in names:
        op_col = names.index("opacity")
        mask = arr[:, op_col] > opacity_filter_logit
        arr = arr[mask]
    if len(arr) < 100:
        raise ValueError(
            f"Generated object has only {len(arr)} usable Gaussians after opacity filtering; "
            f"refusing to scale placeholder or failed SDS output: {object_ply}"
        )

    # Remove spatial outliers: Gaussians more than ``spatial_outlier_sigma``
    # standard deviations from the centroid on any axis are almost certainly
    # DreamGaussian background noise.  Pass ``None`` to disable (useful when
    # diagnosing whether this filter is over-cropping fine surface detail).
    if spatial_outlier_sigma is not None and len(arr) > 0:
        for col_idx in (xi, yi, zi):
            col = arr[:, col_idx]
            mu, sigma = col.mean(), col.std() + 1e-8
            arr = arr[np.abs(col - mu) < spatial_outlier_sigma * sigma]

    # Centre the canonical object on (0,0) horizontally and bring its bottom to z=0.
    arr[:, xi] -= 0.5 * (arr[:, xi].min() + arr[:, xi].max())
    arr[:, yi] -= 0.5 * (arr[:, yi].min() + arr[:, yi].max())
    arr[:, zi] -= arr[:, zi].min()

    arr[:, xi] *= s
    arr[:, yi] *= s
    arr[:, zi] *= s

    # Scale log-scale values so Gaussian ellipsoids resize with the object.
    for sname in ("scale_0", "scale_1", "scale_2"):
        if sname in names:
            arr[:, names.index(sname)] += log_s

    if reorient_to_world_y_down:
        if canonical_frame not in _REORIENT_PRESETS:
            raise ValueError(
                f"Unknown canonical_frame={canonical_frame!r}; expected one of "
                f"{sorted(_REORIENT_PRESETS)}"
            )
        R_preset, Q_preset = _REORIENT_PRESETS[canonical_frame]
        R_use = R_preset
        Q_use = Q_preset
        if align_y_axis_to is not None:
            tgt = np.asarray(align_y_axis_to, dtype=np.float64).reshape(3)
            tgt = tgt / (np.linalg.norm(tgt) + 1e-12)
            R_align, _ = SciRotation.align_vectors(
                np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
                tgt.reshape(1, 3),
            )
            R_align_m = R_align.as_matrix()
            R_use = R_align_m @ R_preset
            q_align = _quat_wxyz_from_scipy(SciRotation.from_matrix(R_align_m))
            Q_use = _hamilton_one(q_align, Q_preset)

        pts = arr[:, [xi, yi, zi]]
        arr[:, [xi, yi, zi]] = pts @ R_use.T

        if "nx" in names and "ny" in names and "nz" in names:
            nxi, nyi, nzi = names.index("nx"), names.index("ny"), names.index("nz")
            nrm = arr[:, [nxi, nyi, nzi]]
            arr[:, [nxi, nyi, nzi]] = nrm @ R_use.T

        if all(f"rot_{k}" in names for k in range(4)):
            ri = [names.index(f"rot_{k}") for k in range(4)]
            q_old = arr[:, ri]
            q_new = _hamilton(Q_use, q_old)
            for k in range(4):
                arr[:, ri[k]] = q_new[:, k]

        rx, ry, rz = post_rotation_deg
        if abs(rx) > 1e-6 or abs(ry) > 1e-6 or abs(rz) > 1e-6:
            R_post = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
            pts = arr[:, [xi, yi, zi]]
            arr[:, [xi, yi, zi]] = pts @ R_post.T

            if "nx" in names and "ny" in names and "nz" in names:
                nxi, nyi, nzi = names.index("nx"), names.index("ny"), names.index("nz")
                nrm = arr[:, [nxi, nyi, nzi]]
                arr[:, [nxi, nyi, nzi]] = nrm @ R_post.T

            if all(f"rot_{k}" in names for k in range(4)):
                ri = [names.index(f"rot_{k}") for k in range(4)]
                q_post = _hamilton_one(
                    _quat_from_axis_angle((0.0, 0.0, 1.0), rz),
                    _hamilton_one(
                        _quat_from_axis_angle((0.0, 1.0, 0.0), ry),
                        _quat_from_axis_angle((1.0, 0.0, 0.0), rx),
                    ),
                )
                q_new = _hamilton(q_post, arr[:, ri])
                for k in range(4):
                    arr[:, ri[k]] = q_new[:, k]

        if (
            contact_base_occlusion > 1e-6
            and all(f"f_dc_{k}" in names for k in range(3))
        ):
            ycol = arr[:, yi]
            foot = float(ycol.max())
            span = float(ycol.max() - ycol.min()) + 1e-8
            band = max(0.05 * span, 1e-5)
            t = (ycol - (foot - band)) / band
            t = np.clip(t, 0.0, 1.0)
            w = (t**2) * float(np.clip(contact_base_occlusion, 0.0, 0.85))
            for k in range(3):
                ci = names.index(f"f_dc_{k}")
                arr[:, ci] *= 1.0 - 0.55 * w

        # After the rotation the duck stands upright (height now along world -y).
        # Place its base at world y = 0 so callers can translate by the floor's y
        # value to land it exactly on the surface.
        arr[:, yi] -= arr[:, yi].max()
        arr[:, xi] -= 0.5 * (arr[:, xi].min() + arr[:, xi].max())
        arr[:, zi] -= 0.5 * (arr[:, zi].min() + arr[:, zi].max())

    new_el = np.array([tuple(row) for row in arr], dtype=v.dtype)
    el = PlyElement.describe(new_el, "vertex")
    PlyData([el], text=False).write(str(out_ply))
    return out_ply


def transform_mesh_vertices_like_rescaled_ply(
    vertices_xyz: np.ndarray,
    object_ply: Path | str,
    scene_ply: Path | str,
    scale_factor: float,
    canonical_frame: str,
    post_rotation_deg: tuple[float, float, float],
    align_y_axis_to: np.ndarray | None,
    *,
    merge_translation: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the same rigid + scale geometry as :func:`rescale_object_ply_to_scene` to mesh vertices.

    ``vertices_xyz`` must live in the same canonical space as ``object_ply`` (DreamGaussian export).
    Opacity / spatial outlier filters are skipped. ``merge_translation`` is the final world shift
    (same vector passed to :func:`placement.ply_merger.merge_ply_files` for the object).
    """
    object_ply, scene_ply = Path(object_ply), Path(scene_ply)
    omin, omax = _bbox_xyz(object_ply)
    smin, smax = _bbox_xyz(scene_ply)
    diag_o = np.linalg.norm(omax - omin) + 1e-8
    diag_s = np.linalg.norm(smax - smin) + 1e-8
    s = scale_factor * (diag_s / diag_o)

    V = np.asarray(vertices_xyz, dtype=np.float64).copy()
    xm = 0.5 * (omin[0] + omax[0])
    ym = 0.5 * (omin[1] + omax[1])
    zmin = omin[2]
    V[:, 0] -= xm
    V[:, 1] -= ym
    V[:, 2] -= zmin
    V *= s

    xi, yi, zi = 0, 1, 2
    if canonical_frame not in _REORIENT_PRESETS:
        raise ValueError(f"Unknown canonical_frame={canonical_frame!r}")
    R_preset, _Q_preset = _REORIENT_PRESETS[canonical_frame]
    R_use = R_preset
    if align_y_axis_to is not None:
        tgt = np.asarray(align_y_axis_to, dtype=np.float64).reshape(3)
        tgt = tgt / (np.linalg.norm(tgt) + 1e-12)
        R_align, _ = SciRotation.align_vectors(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
            tgt.reshape(1, 3),
        )
        R_align_m = R_align.as_matrix()
        R_use = R_align_m @ R_preset

    V = V @ R_use.T
    rx, ry, rz = post_rotation_deg
    if abs(rx) > 1e-6 or abs(ry) > 1e-6 or abs(rz) > 1e-6:
        R_post = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
        V = V @ R_post.T

    V[:, yi] -= V[:, yi].max()
    V[:, xi] -= 0.5 * (V[:, xi].min() + V[:, xi].max())
    V[:, zi] -= 0.5 * (V[:, zi].min() + V[:, zi].max())

    if merge_translation is not None:
        t = np.asarray(merge_translation, dtype=np.float64).reshape(3)
        V += t
    return V
