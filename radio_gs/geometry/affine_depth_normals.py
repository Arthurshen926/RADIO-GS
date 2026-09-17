"""Transport surface normals through a calibrated pinhole depth warp."""
import math
import torch
from torch.nn import functional as F


def affine_depth_normals(points, normals, scale, offset):
    """For X'=(a+b/Z)X, return J(X)^(-T)n, normalized.

    X and normals are already expressed in the target camera's ray geometry.
    A depth offset is not a rigid translation: it displaces each ray in a
    different direction. Positive uniform scale alone leaves normals unchanged.
    """
    if points.shape != normals.shape or points.shape[-1] != 3:
        raise ValueError('aligned camera points and normals required')
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
        raise ValueError('finite positive scale and finite offset required')
    if not torch.isfinite(points).all() or not torch.isfinite(normals).all():
        raise ValueError('finite geometry required; filter invalid pixels first')
    z = points[..., 2]
    if (z <= 0).any() or (scale*z + offset <= 0).any():
        raise ValueError('positive original and calibrated depths required')
    # Multiplying the inverse-transpose result by the positive scalar
    # (a+b/Z) does not change its unit direction.
    result = normals.clone()
    result[..., 2] += offset / (scale*z.square()) * (points*normals).sum(-1)
    return F.normalize(result, dim=-1)
