"""Query-independent proposal atoms bound to the field's observation rays.

The state retains proposal membership and a radius-one neighbourhood relation,
not pre-eroded positives. It therefore represents union BEFORE morphology.
This is an uncompressed segmentation-information reference, not a claim that
source segmentation information has disappeared. No source file is read here.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ScopeTopologyView(nn.Module):
    def __init__(self, state):
        super().__init__()
        if set(state) != {'incidence', 'neighborhood', 'semantic', 'retained'}:
            raise ValueError('invalid scope topology state')
        a, n, s, r = (state[k] for k in ('incidence', 'neighborhood', 'semantic', 'retained'))
        if (a.ndim != 2 or a.dtype != torch.bool or not len(a) or a.shape[1] == 0
                or n.ndim != 2 or n.shape[1] != 9 or n.dtype != torch.int64
                or s.ndim != 2 or s.shape[0] != a.shape[1] or not s.is_floating_point()
                or not torch.isfinite(s).all() or bool((s.float().norm(dim=1) == 0).any())
                or r.shape != (a.shape[1],) or r.dtype != torch.bool
                or bool(((n < 0) | (n >= len(a))).any())):
            raise ValueError('invalid scope topology dimensions or values')
        for k, v in state.items(): self.register_buffer(k, v.detach())

    @torch.no_grad()
    def compose(self, accepted, *, unknown=None):
        if accepted.shape != self.retained.shape or accepted.dtype != torch.bool:
            raise ValueError('invalid accepted proposal domain')
        if unknown is None: unknown = torch.zeros_like(accepted)
        if unknown.shape != accepted.shape or unknown.dtype != torch.bool or bool((unknown & accepted).any()):
            raise ValueError('invalid unknown proposal domain')
        occupied = self.incidence[:, accepted].any(1)
        neighbors = occupied[self.neighborhood]
        target = neighbors[:, 4].float()
        positive = neighbors.all(1)
        negative = ~neighbors.any(1)
        # An empty hypothesis grants no complement exclusion permission.
        if not bool(accepted.any()): negative.zero_()
        uncertain = self.incidence[:, unknown].any(1)[self.neighborhood[:, 4]]
        negative &= ~uncertain
        return target, positive, negative

    @torch.no_grad()
    def native_admission(self, query, negatives):
        s = F.normalize(self.semantic.float(), dim=-1)
        q = F.normalize(query.to(s).reshape(1, -1), dim=-1)
        n = F.normalize(negatives.to(s), dim=-1)
        p = (10 * ((s @ q.T)[:, 0] - (s @ n.T).max(1).values)).sigmoid()
        accepted = p > .5
        confidence = (2 * p[accepted] - 1).max() if bool(accepted.any()) else p.new_zeros(())
        return accepted, confidence, p


class ScopeTopology(nn.Module):
    def __init__(self, state, preparation):
        super().__init__()
        ids = {int(k.split('.')[1]) for k in state if k.startswith('views.')}
        if ids != set(range(len(preparation.views))): raise ValueError('unaligned scope views')
        self.views = nn.ModuleList([ScopeTopologyView({k[len(f'views.{i}.'):]: v for k, v in state.items()
                                   if k.startswith(f'views.{i}.')}) for i in range(len(ids))])
        if set(state) != set(self.state_dict()): raise ValueError('unknown scope topology state')
        for v, p in zip(self.views, preparation.views):
            if len(v.neighborhood) != len(p.valid): raise ValueError('unaligned scope rays')


def compile_scope_topology_view(masks, semantic, retained, pixels):
    """Construction only: discard raster coordinates after binding neighbourhoods.

    Input masks are already the chosen canonical binary discretization. Border
    padding repeats the center, matching max_pool2d's ignored outside domain.
    Unlike the older partial-label compiler, erosion occurs after query union.
    """
    if masks.dtype != torch.bool or masks.ndim != 3: raise ValueError('expected binary proposals')
    regions, height, width = masks.shape
    packed = np.packbits(masks.flatten(1).T.cpu().numpy(), axis=1)
    _, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    incidence = masks.flatten(1).T.cpu()[torch.from_numpy(first)].contiguous()
    pixels = pixels.cpu().long()
    if pixels.ndim != 1 or bool(((pixels < 0) | (pixels >= height * width)).any()):
        raise ValueError('invalid observation ray indices')
    y, x = pixels // width, pixels % width
    neighborhood = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = y + dy, x + dx
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            rows = torch.where(valid, yy * width + xx, pixels)
            neighborhood.append(torch.from_numpy(inverse)[rows])
    return dict(incidence=incidence, neighborhood=torch.stack(neighborhood, 1).long(),
                semantic=semantic.detach().cpu(), retained=retained.detach().cpu())


@torch.no_grad()
def assemble_scope(view, preparation_view, prior, accepted, confidence, *, unknown=None):
    """Complete union, morphology, visibility, then native-positive protection.

    This implements the canonical no-track, scalar-confidence prior permission.
    It does not assert that native admission has detector-equivalent identity.
    """
    target, positive, negative = view.compose(accepted, unknown=unknown)
    valid = preparation_view.valid.to(target.device)
    native_positive = torch.sparse.mm(preparation_view.operator(), prior[:, None])[:, 0] > .5
    positive &= valid
    negative &= valid & ~native_positive
    return dict(target=target, positive=positive, negative=negative, confidence=confidence.float())
