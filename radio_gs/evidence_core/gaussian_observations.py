"""Geometry-adapter operations for region evidence, without teacher decisions.

Gaussian compositing mass differs from equal supported-view normalization used
by registered maps. Keep this policy explicit and retain both region-conditioned
mass and unconditioned geometric mass when requested.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GaussianViewEvidence:
    region_contributions: object
    supported_mass: object
    geometric_mass: object


def compile_gaussian_view(hit, membership, element_count, *, retain_visibility=False):
    """Preserve the historical overlap splitting and sparse multiplication order.

membership is [regions, pixels]. Unobserved/no-region elements are not negative.
geometric_mass is independent of the region masks, but is still specific to the
renderer, camera, resolution and occlusion settings used to produce hit.
"""
    import torch
    from .core import _record
    if membership.ndim != 2 or not membership.is_floating_point():
        raise ValueError('membership must be a floating region/pixel matrix')
    pixel, element, weights = (hit[k] for k in ('pixel_ids', 'gaussian_ids', 'weights'))
    if (pixel.ndim != 1 or element.ndim != 1 or weights.ndim != 1
            or len(pixel) != len(element) or len(element) != len(weights)
            or pixel.dtype not in (torch.int32, torch.int64)
            or element.dtype not in (torch.int32, torch.int64)
            or bool(((pixel < 0) | (pixel >= membership.shape[1])).any())
            or bool(((element < 0) | (element >= element_count)).any())
            or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
            or not bool(torch.isfinite(membership).all()) or bool((membership < 0).any())):
        raise ValueError('invalid Gaussian contribution domains/values')
    pixel, element = pixel.long(), element.long()
    count = membership.sum(0)
    weight = weights.float() * (count[pixel] > 0)
    shape = (element_count, membership.shape[1])
    indices = torch.stack([element, pixel])
    adjoint = torch.sparse_coo_tensor(indices, weight, shape).coalesce()
    transfer = torch.sparse.mm(adjoint, (membership/count.clamp_min(1)).T)
    ones = torch.ones(membership.shape[1], 1, device=membership.device)
    mass = torch.sparse.mm(adjoint, ones)
    geometric = None
    if retain_visibility:
        unconditioned = torch.sparse_coo_tensor(indices, weights.float(), shape).coalesce()
        geometric = torch.sparse.mm(unconditioned, ones)[:, 0]
    _record('compile_gaussian_view')
    return GaussianViewEvidence(transfer, mass, geometric)
