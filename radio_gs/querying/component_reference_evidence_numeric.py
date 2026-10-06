"""FP32 renderer-adjoint boundary for component reference evidence.

Separate CUDA adjoint reductions can differ by several ULPs even for identical
support.  Correct only roundoff-sized foreground-over-total excess; larger
partition inconsistency still fails closed.  The decision rule is unchanged.
"""
import torch
from .component_reference_evidence import retain_reference_supported_components


def retain_numerically_checked_components(labels, candidate, retained, foreground_mass,
                                         visibility_mass, **kwargs):
    fg = torch.as_tensor(foreground_mass)
    total = torch.as_tensor(visibility_mass, device=fg.device)
    if not fg.dtype.is_floating_point or not total.dtype.is_floating_point:
        raise ValueError('Renderer-adjoint masses must be floating point')
    excess = (fg - total).clamp_min(0)
    tolerance = 64 * torch.finfo(torch.float32).eps * total.abs().clamp_min(1)
    if (excess > tolerance).any():
        raise ValueError('Foreground/reference visibility partition mismatch exceeds FP32 tolerance')
    result, record = retain_reference_supported_components(
        labels, candidate, retained, torch.minimum(fg, total), total, **kwargs)
    record.update(numeric_subset_clamped_rows=int((excess > 0).sum()),
                  numeric_subset_max_excess=float(excess.max()) if excess.numel() else 0.,
                  numeric_subset_max_relative_excess=float((excess/total.abs().clamp_min(1)).max()) if excess.numel() else 0.,
                  numeric_tolerance_fp32_epsilon_multiplier=64)
    return result, record
