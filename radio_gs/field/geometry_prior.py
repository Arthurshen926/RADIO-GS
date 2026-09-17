"""Query-free depth compatibility for Gaussian semantic observation writes.

This changes teacher weights, never the exact compositor authority or query
scores. Gaussian centre depth is an approximation; the bounded residual keeps
support when that approximation or the monocular prior is imperfect.
"""
from __future__ import annotations

import math
import torch

CONTRACT = "gaussian_semantic_depth_residual_v1"


def depth_conditioned_weights(
    weights: torch.Tensor,
    gaussian_depth: torch.Tensor,
    observed_depth: torch.Tensor,
    confidence: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.02,
    strength: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded write weights and a valid-depth-evidence indicator.

    Missing depth is unknown and leaves the original write unchanged. Both
    foreground free-space and background occlusion mismatches reduce a write;
    neither supplies a semantic negative or deletes a Gaussian. No probability
    calibration is claimed for the compatibility kernel.
    """
    if not all(math.isfinite(v) for v in (absolute_tolerance, relative_tolerance, strength)):
        raise ValueError("depth policy must be finite")
    if absolute_tolerance <= 0 or relative_tolerance < 0 or not 0 <= strength < 1:
        raise ValueError("require positive tolerance and strength in [0,1)")
    base = torch.as_tensor(weights)
    if not base.is_floating_point() or base.ndim != 1:
        raise ValueError("weights must be floating [hits]")
    z, d, c = [torch.as_tensor(v, device=base.device, dtype=base.dtype)
               for v in (gaussian_depth, observed_depth, confidence)]
    if any(v.shape != base.shape for v in (z, d, c)):
        raise ValueError("depth evidence must align with compositor hits")
    if not bool(torch.isfinite(base).all()) or bool((base < 0).any()):
        raise ValueError("base weights must be finite and nonnegative")
    if not bool(torch.isfinite(c).all()) or bool(((c < 0) | (c > 1)).any()):
        raise ValueError("confidence must be finite in [0,1]")
    known = torch.isfinite(z) & (z > 0) & torch.isfinite(d) & (d > 0) & (c > 0)
    if strength == 0:
        return base.clone(), known
    # Avoid propagating NaNs even in inactive branches.
    safe_d = torch.where(known, d, torch.ones_like(d))
    residual = torch.where(known, z - d, torch.zeros_like(d))
    tolerance = absolute_tolerance + relative_tolerance * safe_d
    compatibility = torch.exp(-0.5 * (residual / tolerance).square())
    factor = torch.where(known, 1 - strength * c * (1 - compatibility), torch.ones_like(c))
    return base * factor, known
