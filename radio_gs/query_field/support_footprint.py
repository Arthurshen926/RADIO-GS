"""Construction-time analytic support measure, stored without source cameras.

This is the v10-reference unoccluded projected Gaussian integral (over the
infinite image plane for projected/in-frustum members), NOT a clipped raster
area, visibility mass, or full-object extent certificate.
"""
import math
import torch
from torch import nn


class SupportFootprint(nn.Module):
    def __init__(self, state, count):
        super().__init__()
        if set(state) != {'cost'}:
            raise ValueError('invalid footprint state')
        cost = state['cost']
        if cost.ndim != 2 or cost.shape[1] != count or not cost.is_floating_point() or not torch.isfinite(cost).all() or bool((cost < 0).any()):
            raise ValueError('invalid footprint costs')
        self.register_buffer('cost', cost.detach())

    def forward(self, active_views):
        if active_views.dtype != torch.bool or active_views.shape != (len(self.cost),) or active_views.device != self.cost.device:
            raise ValueError('invalid active footprint views')
        return self.cost[active_views].sum(0)


def projected_integral(conics, opacities, eligible):
    """Match the source reference's determinant clamp, not sample hit counts."""
    if conics.shape != (len(opacities), 3) or eligible.shape != opacities.shape or eligible.dtype != torch.bool:
        raise ValueError('invalid projected footprint domain')
    result = torch.zeros_like(opacities)
    c = conics[eligible]
    result[eligible] = (2 * math.pi / (c[:, 0]*c[:, 2]-c[:, 1].square()).clamp_min(1e-12).sqrt()
                        * opacities[eligible])
    if not torch.isfinite(result).all() or bool((result < 0).any()):
        raise ValueError('invalid projected footprint')
    return result


@torch.no_grad()
def compile_support_footprint(field, camera, height, width, *, member_domain=None):
    from gsplat.cuda._wrapper import fully_fused_projection
    from .geometry import member_domain_mask
    domain = member_domain_mask(field, member_domain)
    k = camera.intrinsic.to(field.xyz).clone()
    k[0] *= width/camera.width; k[1] *= height/camera.height
    radii, _, _, conics, _ = fully_fused_projection(
        field.xyz, None, field.quaternions, field.scales,
        camera.world_to_camera.to(field.xyz)[None], k[None], width, height,
        near_plane=.001*field.world_units_per_metre, far_plane=100*field.world_units_per_metre)
    return projected_integral(conics[0], field.opacities.flatten(), (radii[0] > 0) & domain)
