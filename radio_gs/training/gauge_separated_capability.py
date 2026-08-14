"""Gauge-separated views for capability-preservation objectives.

Capability heads should improve the direction of a reconstructed RADIO vector
without learning a shortcut through its magnitude.  Raw RADIO reconstruction
remains the sole authority for that radial degree of freedom.  The helper in
this module preserves the exact forward value while projecting gradients onto
the tangent space of every feature vector.
"""

from __future__ import annotations

import torch


def gauge_separated_radio(
    radio: torch.Tensor,
    *,
    feature_dim: int,
    norm_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return unchanged RADIO values with zero radial capability gradient.

    ``feature_dim`` identifies the RADIO axis, so the same interface handles
    primitive rows and rendered maps.  The caller that owns a concrete RADIO
    contract remains responsible for enforcing its feature width.
    """

    values = torch.as_tensor(radio)
    if not values.dtype.is_floating_point or not bool(torch.isfinite(values).all()):
        raise ValueError("RADIO capability values must be finite floating point")
    axis = int(feature_dim)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim or values.shape[axis] <= 0:
        raise ValueError("RADIO capability feature dimension is invalid")
    gauge = torch.linalg.vector_norm(values.float(), dim=axis, keepdim=True)
    # Rendered maps legitimately contain zero background pixels.  Their
    # detached zero gauge makes this branch contribute neither a forward value
    # nor a capability gradient; positive feature vectors retain the tangent
    # Jacobian used by the gauge-separation contract.
    direction = values.float() / gauge.clamp_min(float(norm_epsilon))
    return direction * gauge.detach()


__all__ = ["gauge_separated_radio"]
