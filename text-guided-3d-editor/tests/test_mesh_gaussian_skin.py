from __future__ import annotations

import numpy as np

from rendering.mesh_gaussian_skin import deform_mesh_vertices, skinning_indices_weights


def test_skinning_identity_when_no_motion() -> None:
    mesh = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    g0 = np.random.RandomState(0).randn(20, 3) * 0.05
    g0[:, 0] += 0.5
    idx, w = skinning_indices_weights(mesh, g0, k=6)
    gt = g0.copy()
    out = deform_mesh_vertices(mesh, g0, gt, idx, w)
    np.testing.assert_allclose(out, mesh, atol=1e-10)


def test_skinning_follows_single_gaussian_shift() -> None:
    """One mesh vertex at same location as one Gaussian; shift that Gaussian only."""
    g0 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
    mesh = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    idx, w = skinning_indices_weights(mesh, g0, k=2)
    gt = g0.copy()
    gt[0] += np.array([0.0, 0.3, 0.0], dtype=np.float64)
    out = deform_mesh_vertices(mesh, g0, gt, idx, w)
    assert out.shape == (1, 3)
    assert abs(float(out[0, 1]) - 0.3) < 0.05
