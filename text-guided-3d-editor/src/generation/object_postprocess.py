"""Post-processing for generated Gaussian objects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def is_red_duck_prompt(prompt: str) -> bool:
    """True only when the user explicitly asks for a *red* duck (not generic rubber duck)."""
    p = prompt.lower()
    return "duck" in p and "red" in p


_REALISM_KEYWORDS = (
    "ball",
    "sphere",
    "mug",
    "cup",
    "vase",
    "bottle",
    "chair",
    "pot",
    "bowl",
    "lamp",
    "duck",
)


def wants_duck_realism(prompt: str) -> bool:
    """Backward-compatible alias for the broader realism preset."""
    return wants_object_realism_preset(prompt)


def wants_object_realism_preset(prompt: str) -> bool:
    """Whether mode-a should apply automatic realism defaults (image-mode, mesh-render, floor refine).

    Fires for compact, mostly convex toy-like objects that DreamGaussian
    image-to-3D handles well.  Animals other than ducks are intentionally not
    in this list because SDS-based animal silhouettes are still unreliable.
    """
    p = prompt.lower()
    if "duck" in p:
        return any(k in p for k in ("rubber", "bath", "red", "yellow"))
    return any(k in p for k in _REALISM_KEYWORDS)


def enhance_generation_prompt(prompt: str) -> str:
    """Make terse prompts specific enough for text-only SDS."""
    p = prompt.lower()
    if is_red_duck_prompt(prompt):
        return (
            "a high quality 3D model of a red rubber duck toy, yellow beak, "
            "small black eyes, cute duck silhouette, smooth glossy plastic, "
            "single centered isolated object, product render"
        )
    if "duck" in p and ("rubber" in p or "bath" in p or "yellow" in p):
        return (
            "a high quality 3D model of a classic yellow rubber duck bath toy, "
            "orange beak, black eyes, smooth glossy plastic, cute proportions, "
            "single centered isolated object, product render"
        )
    if any(k in p for k in ("ball", "sphere")):
        colour = ""
        for c in ("red", "yellow", "blue", "green", "orange", "white", "black", "pink", "purple"):
            if c in p:
                colour = c + " "
                break
        return (
            f"a high quality 3D model of a single {colour}rubber ball, "
            f"perfectly round sphere, smooth matte rubber, soft seam, sitting on a flat surface, "
            f"single centered isolated object, pure white background, product render"
        )
    if "vase" in p:
        colour = ""
        for c in ("white", "blue", "green", "black", "red", "pink", "yellow", "beige", "brown", "grey"):
            if c in p:
                colour = c + " "
                break
        return (
            f"a high quality 3D model of a single empty {colour}ceramic vase, "
            f"smooth glossy porcelain surface, classic round body with a narrow neck, "
            f"sitting upright on a flat base, axially symmetric, no flowers, no decorations, "
            f"single centered isolated object, pure white background, product render"
        )
    return prompt


def _sh(rgb: tuple[float, float, float]) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def recolor_red_duck_ply(src_ply: Path | str, out_ply: Path | str | None = None) -> Path:
    """Stabilize red-duck colors while preserving generated geometry.

    Text-only SDS often learns a duck silhouette but collapses color toward a
    noisy yellow duck prior. This edits only SH DC color fields.
    """
    src_ply = Path(src_ply)
    out_ply = src_ply if out_ply is None else Path(out_ply)
    ply = PlyData.read(str(src_ply))
    v = np.array(ply["vertex"].data, copy=True)
    names = v.dtype.names or ()
    if not {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        return src_ply

    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())
    w, d, h = xmax - xmin + 1e-8, ymax - ymin + 1e-8, zmax - zmin + 1e-8

    # DreamGaussian's duck front is the protruding low-x region for the
    # generated red-duck prompt. Keep that region yellow as the beak.
    beak = (
        (x < xmin + 0.23 * w)
        & (z > zmin + 0.32 * h)
        & (z < zmin + 0.62 * h)
    )

    ex = xmin + 0.30 * w
    ez = zmin + 0.67 * h
    side = 0.33 * d
    eye_left = (
        ((x - ex) / (0.035 * w)) ** 2
        + ((y - (ymin + side)) / (0.055 * d)) ** 2
        + ((z - ez) / (0.050 * h)) ** 2
        < 1
    )
    eye_right = (
        ((x - ex) / (0.035 * w)) ** 2
        + ((y - (ymax - side)) / (0.055 * d)) ** 2
        + ((z - ez) / (0.050 * h)) ** 2
        < 1
    )
    eye = eye_left | eye_right
    underside = z < zmin + 0.18 * h

    red = _sh((0.90, 0.015, 0.012))
    dark_red = _sh((0.62, 0.0, 0.0))
    yellow = _sh((1.0, 0.68, 0.02))
    black = _sh((0.0, 0.0, 0.0))

    for i, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        v[name] = red[i]
        v[name][underside] = dark_red[i]
        v[name][beak] = yellow[i]
        v[name][eye] = black[i]

    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(out_ply))
    return out_ply


def swap_red_yellow_duck_ply(src_ply: Path | str, out_ply: Path | str | None = None) -> Path:
    """Swap a generated duck's red and yellow regions in SH DC colour space.

    This is intentionally colour-based rather than geometry-based because the
    image-to-3D duck already has a recognizable shape, but its canonical axes
    and back-side extrapolation can vary between runs.  Red/pink body regions
    become rubber-duck yellow; yellow beak/wing regions become red.
    """
    src_ply = Path(src_ply)
    out_ply = src_ply if out_ply is None else Path(out_ply)
    ply = PlyData.read(str(src_ply))
    v = np.array(ply["vertex"].data, copy=True)
    names = v.dtype.names or ()
    if not {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        return src_ply

    dc_names = ("f_dc_0", "f_dc_1", "f_dc_2")
    rgb = np.stack([np.asarray(v[name], dtype=np.float32) for name in dc_names], axis=1)
    rgb = np.clip(rgb * C0 + 0.5, 0.0, 1.0)

    # Keep black eyes / very dark details untouched.
    dark = (rgb[:, 0] < 0.16) & (rgb[:, 1] < 0.16) & (rgb[:, 2] < 0.16)
    yellowish = (rgb[:, 0] > 0.50) & (rgb[:, 1] > 0.35) & (rgb[:, 2] < 0.35) & ~dark
    # The generated back side often becomes pink / magenta; treat that as part
    # of the red body so the swapped result becomes a clean yellow duck instead
    # of retaining a creepy purple patch.
    reddish = (rgb[:, 0] > 0.42) & ~yellowish & ~dark

    yellow = _sh((1.0, 0.78, 0.04))
    red = _sh((0.92, 0.02, 0.015))
    for i, name in enumerate(dc_names):
        v[name][reddish] = yellow[i]
        v[name][yellowish] = red[i]

    # Avoid stale high-order SH terms reintroducing view-dependent red/purple
    # tint after the DC colour has been swapped.
    for name in names:
        if name.startswith("f_rest_"):
            v[name] = 0.0

    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(out_ply))
    return out_ply
