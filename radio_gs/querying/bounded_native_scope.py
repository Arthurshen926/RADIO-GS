"""Opt-in bounded scope correction of calibrated native semantic logits.

Inputs must already share a calibrated logit scale. This module neither
calibrates features nor establishes object identity. Zero correction is default.
"""
import math
import torch


def refine_native_scope_logits(native_logits: torch.Tensor,
                               scope_logits: torch.Tensor, *,
                               max_correction: float = 0.0) -> torch.Tensor:
    """Limit each scope correction; native class margins > 2*bound survive.

The guarantee applies to calibrated native logits, not raw cosine scores.
Unobserved elements must be excluded by the caller's existing output contract.
"""
    bound = float(max_correction)
    if not math.isfinite(bound) or bound < 0:
        raise ValueError('max_correction must be finite and nonnegative')
    if native_logits.shape != scope_logits.shape:
        raise ValueError('native and scope logits must have identical shapes')
    if native_logits.device != scope_logits.device or native_logits.dtype != scope_logits.dtype:
        raise ValueError('native and scope logits must share device and dtype')
    if not native_logits.is_floating_point():
        raise ValueError('logits must be floating point')
    if not torch.isfinite(native_logits).all() or not torch.isfinite(scope_logits).all():
        raise ValueError('logits must be finite')
    if bound == 0:
        return native_logits
    return native_logits + (scope_logits - native_logits).clamp(-bound, bound)
