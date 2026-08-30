import json

import torch

from radio_gs.v4.carrier import Camera
from radio_gs.v4.geometry import DepthObservation, SparseSurfaceFusion, fit_constrained_affine_depth
from radio_gs.v4.evaluation.geometry_ladder import _load_cameras
from radio_gs.v4.evaluation.source_mask_ladder import _mutually_exclusive_purity


def test_nerf_transform_is_converted_from_opengl_to_opencv(tmp_path):
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({
        "fl_x": 4, "fl_y": 4, "cx": 2, "cy": 2, "w": 4, "h": 4,
        "frames": [{"file_path": "color/7", "transform_matrix": torch.eye(4).tolist()}],
    }))
    loaded = _load_cameras(path, [{"frame_id": 7}], 4, 4)[0]
    assert torch.equal(
        loaded.camera_to_world,
        torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64)),
    )


def test_purity_uses_only_mutually_exclusive_masks():
    masks = torch.tensor([
        [[1, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, 1]],
    ]).float()
    posterior = torch.tensor([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    values, pair_count = _mutually_exclusive_purity(posterior, masks)
    assert pair_count == 2
    assert values == [1.0, 1.0]


def test_constrained_affine_depth_recovers_small_calibration_and_rejects_large_one():
    predicted = torch.linspace(1, 4, 100)
    reference = 1.05 * predicted + 0.02
    fit = fit_constrained_affine_depth(
        predicted,
        reference,
        scale_prior_weight=0.1,
        offset_prior_weight=0.1,
    )
    assert fit.accepted
    assert abs(fit.scale - 1.05) < 1e-3
    assert abs(fit.offset - 0.02) < 1e-3
    rejected = fit_constrained_affine_depth(
        predicted,
        1.5 * predicted,
        scale_prior_weight=0.1,
        offset_prior_weight=0.1,
    )
    assert not rejected.accepted
    assert rejected.rejection_reason == "scale_outside_preregistered_bound"


def test_sparse_surface_fusion_requires_cross_view_support():
    intrinsic = torch.tensor([[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1.0]])
    first = Camera("first", intrinsic, torch.eye(4), 1, 1)
    second = Camera("second", intrinsic, torch.eye(4), 1, 1)
    observation = lambda value, camera: DepthObservation(
        camera=camera,
        depth=torch.tensor([[value]]),
        validity=torch.ones(1, 1, dtype=torch.bool),
        confidence=torch.ones(1, 1),
        normals_camera=torch.tensor([[[0.0, 0.0, -1.0]]]),
    )
    result = SparseSurfaceFusion(0.1, minimum_views=2).fuse(
        [observation(2.0, first), observation(2.01, second)]
    )
    assert result.centres.shape == (1, 3)
    assert result.view_count.tolist() == [2]
    assert float(result.dispersion[0]) < 0.01
