import pytest
import torch

from radio_gs.rendering.camera_clearance import (
    camera_plane_clearance_confidence,
    quaternion_to_rotation_matrix,
)


def test_identity_quaternion_rotation() -> None:
    rotation = quaternion_to_rotation_matrix(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(rotation, torch.eye(3)[None])


def test_two_sigma_camera_plane_clearance_is_scale_aware() -> None:
    result = camera_plane_clearance_confidence(
        torch.tensor([[0.0, 0.0, 0.15], [0.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.01, 0.01, 0.10], [0.01, 0.01, 0.10]]),
        torch.eye(4),
        near_plane=0.01,
        support_sigma=2.0,
    )
    torch.testing.assert_close(result.center_depth, torch.tensor([0.15, 1.0]))
    torch.testing.assert_close(
        result.axial_standard_deviation, torch.tensor([0.10, 0.10])
    )
    torch.testing.assert_close(result.confidence, torch.tensor([0.0, 1.0]))


def test_camera_clearance_is_invariant_to_joint_scene_scaling() -> None:
    args = (
        torch.tensor([[0.2, -0.1, 0.5]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.1, 0.2, 0.3]]),
    )
    view = torch.eye(4)
    base = camera_plane_clearance_confidence(
        *args, view, near_plane=0.01, support_sigma=2.0
    )
    scaled = camera_plane_clearance_confidence(
        args[0] * 7.0,
        args[1],
        args[2] * 7.0,
        view,
        near_plane=0.07,
        support_sigma=2.0,
    )
    torch.testing.assert_close(base.confidence, scaled.confidence)


def test_camera_clearance_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="means"):
        camera_plane_clearance_confidence(
            torch.zeros(3),
            torch.zeros(1, 4),
            torch.ones(1, 3),
            torch.eye(4),
            near_plane=0.01,
        )
