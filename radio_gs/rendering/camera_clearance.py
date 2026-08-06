"""Query-independent visibility guards for Gaussian camera-plane collisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


CAMERA_PLANE_CLEARANCE_CONTRACT = (
    "gaussian_two_sigma_support_strictly_in_front_of_camera_near_plane_v1"
)


@dataclass(frozen=True)
class CameraPlaneClearance:
    """Per-row camera-plane clearance and a hard physical visibility gate."""

    confidence: torch.Tensor
    center_depth: torch.Tensor
    axial_standard_deviation: torch.Tensor
    lower_support_depth: torch.Tensor


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``(w, x, y, z)`` quaternions to rotation matrices."""

    value = torch.as_tensor(quaternion).float()
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError("quaternion must have shape [N,4]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quaternion must be finite")
    value = torch.nn.functional.normalize(value, dim=-1, eps=1e-12)
    w, x, y, z = value.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def camera_plane_clearance_confidence(
    means: torch.Tensor,
    quaternions: torch.Tensor,
    scales: torch.Tensor,
    viewmat: torch.Tensor,
    *,
    near_plane: float,
    support_sigma: float = 2.0,
) -> CameraPlaneClearance:
    """Reject ellipsoids whose Gaussian support intersects the near plane.

    A 3D Gaussian with covariance ``R diag(scale**2) R.T`` has axial standard
    deviation ``sqrt(v.T covariance v)`` along the camera's optical axis
    ``v``.  The gate retains a row only when the lower ``support_sigma``
    support bound is strictly in front of the renderer near plane.  It uses no
    image, feature, prompt, query, or benchmark label and is invariant to a
    joint scaling of the scene, camera translations, Gaussian scales, and near
    plane.
    """

    xyz = torch.as_tensor(means).float()
    quat = torch.as_tensor(quaternions, device=xyz.device).float()
    sigma = torch.as_tensor(scales, device=xyz.device).float()
    view = torch.as_tensor(viewmat, device=xyz.device).float()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("means must have shape [N,3]")
    if quat.shape != (xyz.shape[0], 4):
        raise ValueError("quaternions must align with means as [N,4]")
    if sigma.shape != xyz.shape:
        raise ValueError("scales must align with means as [N,3]")
    if view.shape != (4, 4):
        raise ValueError("viewmat must have shape [4,4]")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(sigma).all()):
        raise ValueError("means and scales must be finite")
    if bool((sigma < 0).any()):
        raise ValueError("scales must be non-negative")
    if not bool(torch.isfinite(view).all()):
        raise ValueError("viewmat must be finite")
    if not torch.isfinite(torch.tensor(float(near_plane))) or float(near_plane) < 0:
        raise ValueError("near_plane must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(support_sigma))) or float(support_sigma) < 0:
        raise ValueError("support_sigma must be finite and non-negative")

    optical_axis = view[2, :3]
    center_depth = xyz @ optical_axis + view[2, 3]
    rotation = quaternion_to_rotation_matrix(quat)
    axis_in_local_frame = torch.einsum(
        "nij,j->ni", rotation.transpose(1, 2), optical_axis
    )
    axial_variance = torch.sum((axis_in_local_frame * sigma) ** 2, dim=-1)
    axial_standard_deviation = torch.sqrt(axial_variance.clamp_min(0.0))
    lower_support_depth = center_depth - float(support_sigma) * axial_standard_deviation
    confidence = (lower_support_depth > float(near_plane)).to(dtype=xyz.dtype)
    return CameraPlaneClearance(
        confidence=confidence,
        center_depth=center_depth,
        axial_standard_deviation=axial_standard_deviation,
        lower_support_depth=lower_support_depth,
    )
