"""Sampled joint opacity factors over Gaussian members, not a member decoder.

Construction can use cameras. Stored factors contain only ordered Gaussian
incidences, unclipped footprint alpha, and bindings to field-owned scope atoms.
No image dimensions, pixel locations, camera, mask raster, or query answers are
needed to evaluate the compiled function. Sampling is NOT full-image coverage.
"""
import torch
from torch import nn


class CoverageView(nn.Module):
    def __init__(self, state, count, atoms):
        super().__init__()
        if set(state) != {'pointer', 'index', 'alpha', 'atom'}:
            raise ValueError('incomplete coverage factor state')
        for k, v in state.items():
            self.register_buffer(k, v.detach())
        self.count = count
        if any(x.dtype != torch.long or x.ndim != 1 for x in [self.pointer, self.index, self.atom]):
            raise ValueError('invalid coverage coordinates')
        if self.pointer.shape != (len(self.atom)+1,) or int(self.pointer[0]) != 0 or int(self.pointer[-1]) != len(self.index) or bool((self.pointer.diff()<0).any()):
            raise ValueError('invalid coverage pointer')
        if self.alpha.shape != self.index.shape or not torch.isfinite(self.alpha).all() or bool(((self.alpha<0)|(self.alpha>1)).any()):
            raise ValueError('invalid footprint alpha')
        if bool(((self.index<0)|(self.index>=count)).any()) or bool(((self.atom<0)|(self.atom>=atoms)).any()):
            raise ValueError('unaligned coverage factor binding')

    def forward(self, membership):
        if membership.shape != (self.count,) or not torch.isfinite(membership).all() or bool(((membership<0)|(membership>1)).any()):
            raise ValueError('invalid membership')
        if not len(self.index):
            return membership.sum()*0 + membership.new_zeros(len(self.atom))
        # Reapply the renderer's cutoff AFTER member opacity scaling. Retain
        # hidden footprints: never reuse full-scene visibility contributions.
        alpha = (self.alpha * membership[self.index]).clamp(max=.999)
        alpha = torch.where(alpha >= 1/255, alpha, 0.)
        group = torch.repeat_interleave(torch.arange(len(self.atom), device=alpha.device), self.pointer.diff())
        log_step = torch.log1p(-alpha.double())
        prefix = torch.cumsum(log_step, 0)
        padded = torch.cat([prefix.new_zeros(1), prefix])
        inclusive = prefix - padded[self.pointer[:-1]][group]
        exclusive = inclusive - log_step
        # gsplat omits the hit that would make T < 1e-4, and all later hits.
        weight = alpha.double()*exclusive.exp()*(inclusive.exp() >= 1e-4)
        result = torch.zeros(len(self.atom), device=alpha.device, dtype=torch.float64)
        return result.scatter_add(0, group, weight).to(membership.dtype)


class CoverageFactors(nn.Module):
    def __init__(self, state, count, constraints):
        super().__init__()
        ids = {int(k.split('.')[1]) for k in state if k.startswith('views.')}
        if ids != set(range(len(constraints.views))):
            raise ValueError('coverage and constraint views differ')
        self.views = nn.ModuleList([CoverageView({k[len(f'views.{i}.'):]:v for k,v in state.items() if k.startswith(f'views.{i}.')}, count, len(c.positive)) for i,c in enumerate(constraints.views)])
        if set(state) != set(self.state_dict()):
            raise ValueError('unknown coverage factor state')


@torch.no_grad()
def compile_coverage_view(field, camera, height, width, pixels, atom_ids):
    """Enumerate ALL footprints at construction samples, including occluded ones.

    Use gsplat's own tile intersection list and depth ordering. Do not use its
    early-terminated visible hit list: removal of foreground can expose any
    later member. Original sample indices are returned nowhere in the state.
    """
    from gsplat import rasterization
    k = camera.intrinsic.to(field.xyz).clone()
    k[0] *= width/camera.width; k[1] *= height/camera.height
    _, _, info = rasterization(means=field.xyz, quats=field.quaternions, scales=field.scales,
        opacities=field.opacities.flatten(), colors=torch.zeros(len(field.xyz),1,device=field.xyz.device),
        viewmats=camera.world_to_camera.to(field.xyz)[None], Ks=k[None], width=width, height=height,
        near_plane=.001*field.world_units_per_metre, far_plane=100*field.world_units_per_metre,
        packed=False)
    offsets = info['isect_offsets'].flatten()
    offsets = torch.cat([offsets, offsets.new_tensor([len(info['flatten_ids'])])])
    ids=[]; values=[]; pointer=[0]; tile=int(info['tile_size'])
    tiles_w=info['isect_offsets'].shape[-1]
    for pixel in pixels.tolist():
        x,y=pixel%width,pixel//width; t=(y//tile)*tiles_w+x//tile
        g=info['flatten_ids'][int(offsets[t]):int(offsets[t+1])].long()
        d=info['means2d'][0,g]-field.xyz.new_tensor([x+.5,y+.5]); c=info['conics'][0,g]
        sigma=.5*(c[:,0]*d[:,0].square()+c[:,2]*d[:,1].square())+c[:,1]*d[:,0]*d[:,1]
        a=info['opacities'][0,g]*torch.exp(-sigma)
        keep=(sigma>=0)&(a>=1/255)&field.observed[g]
        ids.append(g[keep].cpu()); values.append(a[keep].cpu()); pointer.append(pointer[-1]+int(keep.sum()))
    return dict(pointer=torch.tensor(pointer,dtype=torch.long),index=torch.cat(ids),alpha=torch.cat(values),atom=atom_ids.long().cpu())
