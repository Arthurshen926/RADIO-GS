"""Uncompressed, query-free whole-phase optical factors on one Gaussian field.

The carrier has no source cameras, rasters, pixel coordinates or query answers.
It preserves every ray in a construction quadrature, including hidden hits.
It does not by itself supply query scope or certify unseen-view coverage.
"""
import numpy as np
import torch
from torch import nn
from .coverage_factors import CoverageView


class ExtentFactors(nn.Module):
    def __init__(self, state, count):
        super().__init__()
        ids = {int(k.split('.')[1]) for k in state if k.startswith('views.')}
        if not ids or ids != set(range(len(ids))):
            raise ValueError('noncontiguous extent factors')
        self.views = nn.ModuleList()
        for i in range(len(ids)):
            prefix = f'views.{i}.'
            view = CoverageView({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, count, 1)
            self.views.append(view)
        if set(state) != set(self.state_dict()):
            raise ValueError('unknown extent factor state')


@torch.no_grad()
def compile_extent_view(field, camera, height, width, pixels, *, member_domain):
    """Batch by construction tile, then restore exact supplied ray/depth order.

    Full phases require tens of thousands of rays; the per-ray Python compiler
    is retained as an independent small-sample construction check.
    """
    from gsplat import rasterization
    from .geometry import member_domain_mask
    domain = member_domain_mask(field, member_domain)
    pixels = pixels.cpu().long()
    if pixels.ndim != 1 or bool(((pixels < 0) | (pixels >= height * width)).any()) or len(pixels.unique()) != len(pixels):
        raise ValueError('invalid construction rays')
    k = camera.intrinsic.to(field.xyz).clone()
    k[0] *= width / camera.width; k[1] *= height / camera.height
    _, _, info = rasterization(means=field.xyz, quats=field.quaternions, scales=field.scales,
        opacities=field.opacities.flatten(), colors=torch.zeros(len(field.xyz), 1, device=field.xyz.device),
        viewmats=camera.world_to_camera.to(field.xyz)[None], Ks=k[None], width=width, height=height,
        near_plane=.001 * field.world_units_per_metre, far_plane=100 * field.world_units_per_metre, packed=False)
    offsets = info['isect_offsets'].flatten().cpu().tolist() + [len(info['flatten_ids'])]
    tile = int(info['tile_size']); tiles_w = info['isect_offsets'].shape[-1]
    tile_ids = ((pixels // width // tile) * tiles_w + pixels % width // tile).numpy()
    order = np.argsort(tile_ids, kind='stable'); boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(tile_ids[order])), len(order)]
    rows, ids, values = [], [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if a == b: continue
        t = int(tile_ids[order[a]])
        g = info['flatten_ids'][offsets[t]:offsets[t+1]].long()
        g = g[domain[g]]
        if not len(g): continue
        ray = torch.from_numpy(order[a:b]).to(field.xyz.device)
        px = pixels[order[a:b]].to(field.xyz.device)
        xy = torch.stack([px % width, px // width], -1).to(field.xyz.dtype) + .5
        # A tile has at most 256 rays; no scene-sized dense incidence matrix.
        delta = info['means2d'][0, g][None] - xy[:, None]
        c = info['conics'][0, g]
        sigma = .5 * (c[None, :, 0] * delta[..., 0].square() + c[None, :, 2] * delta[..., 1].square()) + c[None, :, 1] * delta[..., 0] * delta[..., 1]
        alpha = info['opacities'][0, g][None] * torch.exp(-sigma)
        r, col = torch.where((sigma >= 0) & (alpha >= 1/255))
        rows.append(ray[r].cpu().numpy()); ids.append(g[col].cpu().numpy()); values.append(alpha[r, col].cpu().numpy())
    if not rows:
        row = np.empty(0, np.int64); index = row.copy(); alpha = np.empty(0, np.float32)
    else:
        row = np.concatenate(rows); permutation = np.argsort(row, kind='stable')
        row = row[permutation]; index = np.concatenate(ids)[permutation]; alpha = np.concatenate(values)[permutation]
    pointer = np.r_[0, np.bincount(row, minlength=len(pixels)).cumsum()]
    return dict(pointer=torch.from_numpy(pointer), index=torch.from_numpy(index), alpha=torch.from_numpy(alpha),
                atom=torch.zeros(len(pixels), dtype=torch.long))
