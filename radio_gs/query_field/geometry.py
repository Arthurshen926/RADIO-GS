"""Source-free Gaussian visibility and prompt registration numerics."""
from types import SimpleNamespace

import torch

from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions


def projection(geometry, camera, height, width, world_scale):
    intrinsic = camera.intrinsic.clone().to(geometry.xyz)
    intrinsic[0] *= width / camera.width
    intrinsic[1] *= height / camera.height
    renderer = SimpleNamespace(scaled_intrinsics=lambda w, h: intrinsic,
                               near_plane=0.001*world_scale, far_plane=100*world_scale)
    hit = rasterize_single_view_contributions(geometry, renderer,
            camera.world_to_camera.to(geometry.xyz), height=height, width=width, include_depths=False)
    indices = torch.stack([hit["pixel_ids"].long(), hit["gaussian_ids"].long()])
    raw = torch.sparse_coo_tensor(indices, hit["weights"].float(),
                                 (height*width, len(geometry.xyz))).coalesce()
    mass = torch.sparse.sum(raw, dim=1).to_dense()
    normalized = torch.sparse_coo_tensor(raw.indices(), raw.values()/mass[raw.indices()[0]].clamp_min(1e-8),
                                        raw.shape).coalesce()
    best = torch.full((height*width,), len(geometry.xyz), dtype=torch.long, device=geometry.xyz.device)
    if raw._nnz():
        pixels, elements = raw.indices()
        maximum = torch.zeros_like(mass).scatter_reduce_(0, pixels, raw.values(), reduce="amax")
        ties = raw.values() == maximum[pixels]
        best.scatter_reduce_(0, pixels[ties], elements[ties], reduce="amin")
    best[best == len(geometry.xyz)] = -1
    return raw, normalized, mass, best


@torch.no_grad()
def selected_alpha(field, selected, camera, height, width, world_scale=1.):
    """Recompute transmittance after selection, not a weighted full-scene mask."""
    if not bool(selected.any()):
        return torch.zeros(height*width, device=field.xyz.device)
    xyz = field.xyz[selected]
    geometry = SimpleNamespace(xyz=xyz, get_xyz=lambda:xyz,
        get_rotation=lambda:field.quaternions[selected],
        get_scaling=lambda:field.scales[selected],
        get_opacity=lambda:field.opacities[selected])
    return projection(geometry, camera, height, width, world_scale)[2]


def member_domain_mask(field, member_domain=None):
    """Validate an explicit geometric selection domain; legacy default is observed.

    This does not expand source labels or certify semantic support as foreground.
    Callers using semantic-only members must pass their actual available domain.
    """
    domain = field.observed if member_domain is None else member_domain
    if (domain.dtype != torch.bool or domain.shape != (len(field.xyz),)
            or domain.device != field.xyz.device):
        raise ValueError('member domain must be a Boolean mask aligned with field rows and device')
    return domain


def relaxed_subset_alpha(field, membership, camera, height, width, world_scale=1., *, member_domain=None):
    """Differentiate a query-selected subset's opacity, keeping geometry frozen.

    This is a soft selection surrogate. It is not the full-scene normalized
    feature renderer and does not change stored Gaussian opacity. Every eligible
    Gaussian may receive a gradient, including ones hidden in the original view.
    Rasterizer alpha clipping/culling remains an approximation to a discrete set.
    """
    from gsplat import rasterization
    if membership.shape != (len(field.xyz),):
        raise ValueError("membership must align with Gaussian rows")
    intrinsic = camera.intrinsic.clone().to(field.xyz)
    intrinsic[0] *= width/camera.width
    intrinsic[1] *= height/camera.height
    # Geometry and base opacity are immutable buffers. Only transient selection
    # comes from the trainable member decoder.
    opacity = field.opacities.flatten() * membership * member_domain_mask(field, member_domain)
    _, alpha, _ = rasterization(means=field.xyz, quats=field.quaternions, scales=field.scales,
        opacities=opacity, colors=torch.zeros(len(field.xyz),1,device=field.xyz.device),
        viewmats=camera.world_to_camera.to(field.xyz)[None], Ks=intrinsic[None], width=width,height=height,
        near_plane=.001*world_scale,far_plane=100*world_scale,backgrounds=torch.zeros(1,1,device=field.xyz.device),
        render_mode="RGB",packed=False)
    return alpha[0,...,0].flatten()
