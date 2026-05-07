from __future__ import annotations

from pathlib import Path

import numpy as np

from segmentation.grounded_sam2_mask import sam2_hydra_config_name
from segmentation.mask_to_gaussians import (
    cam_pinhole_xyz_to_world_row,
    mask_to_gaussian_indices,
    world_view_transform_apply,
)
from segmentation.multi_view_consensus import consensus_indices
from segmentation.object_volume import expand_indices_to_bbox_volume


def test_world_view_identity_roundtrip() -> None:
    W = np.eye(4, dtype=np.float64)
    Xw = np.array([[2.0, -1.0, 3.0]], dtype=np.float64)
    Xc = world_view_transform_apply(W, Xw)
    assert np.allclose(Xc, Xw)
    Xb = cam_pinhole_xyz_to_world_row(W, Xc)
    assert np.allclose(Xb, Xw)


def test_mask_to_gaussians() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    depth = np.ones((10, 10), dtype=np.float32) * 1.0
    K = np.array([[5.0, 0, 5.0], [0, 5.0, 5.0], [0, 0, 1.0]], dtype=np.float64)
    I = np.eye(4, dtype=np.float64)
    pos = np.array([[0.0, 0.0, 1.0], [10.0, 10.0, 10.0]], dtype=np.float64)
    idx = mask_to_gaussian_indices(mask, depth, K, I, pos, distance_threshold=5.0, stride=1)
    assert len(idx) >= 1


def test_consensus() -> None:
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    c = consensus_indices([a, b], min_votes=2)
    assert 2 in c and 3 in c


def test_sam2_hydra_config_name_from_repo_path() -> None:
    p = Path("/x/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    assert sam2_hydra_config_name(p) == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_expand_indices_to_bbox_volume() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float64,
    )
    idx = expand_indices_to_bbox_volume(pos, np.array([0, 1]), margin_ratio=0.01, z_margin_ratio=0.01)
    assert set(idx.tolist()) == {0, 1, 2}
