"""Query-free observation lifting and support measure for extent decisions.

The lift uses FULL-scene visibility before restricting available members. It
is intentionally different from the selected-set optics in ExtentFactors.
Only incidence weights, validity and analytic costs persist; source cameras,
images, masks, query targets and query initializations do not.
"""
from types import SimpleNamespace
import torch
from torch import nn


class ExtentPreparationView(nn.Module):
    def __init__(self, state, count):
        super().__init__()
        if set(state) != {'index', 'weight', 'valid', 'footprint'}:
            raise ValueError('invalid extent preparation state')
        index, weight, valid, footprint = (state[k] for k in ('index', 'weight', 'valid', 'footprint'))
        if (index.dtype != torch.int64 or index.ndim != 2 or index.shape[0] != 2
                or weight.dtype != torch.float32 or weight.shape != (index.shape[1],)
                or valid.dtype != torch.bool or valid.ndim != 1
                or footprint.dtype != torch.float32 or footprint.shape != (count,)):
            raise ValueError('invalid extent preparation dimensions')
        if index.numel() and (bool((index < 0).any()) or int(index[0].max()) >= len(valid) or int(index[1].max()) >= count):
            raise ValueError('extent preparation index outside domain')
        if any(not torch.isfinite(x).all() or bool((x < 0).any()) for x in (weight, footprint)):
            raise ValueError('invalid extent preparation measure')
        # Saved COO order is the canonical coalesced row/Gaussian order.
        key = index[0] * count + index[1]
        if len(key) > 1 and bool((key[1:] <= key[:-1]).any()):
            raise ValueError('extent preparation indices must be unique and sorted')
        self.count = count
        for k, value in state.items(): self.register_buffer(k, value.detach())

    def operator(self):
        return torch.sparse_coo_tensor(self.index, self.weight, (len(self.valid), self.count)).coalesce()


class ExtentPreparation(nn.Module):
    def __init__(self, state, count):
        super().__init__()
        ids = {int(k.split('.')[1]) for k in state if k.startswith('views.')}
        if not ids or ids != set(range(len(ids))):
            raise ValueError('noncontiguous extent preparation views')
        self.count = count
        self.views = nn.ModuleList([ExtentPreparationView(
            {k[len(f'views.{i}.'):]: v for k, v in state.items() if k.startswith(f'views.{i}.')}, count)
            for i in range(len(ids))])
        if set(state) != set(self.state_dict()): raise ValueError('unknown extent preparation state')

    @torch.no_grad()
    def forward(self, prior, scopes):
        """Read one query's supplied scope; do not infer scope or treat unknown as negative.

        This matches the canonical scalar-confidence condition. Per-pixel local
        reliability is a different contract and must not be silently substituted.
        Unsupported Gaussian rows retain their continuous native prior. The cost
        uses exactly those views with admitted positive evidence, weighted by
        confidence and normalized against this query's reconstructed initializer.
        """
        if (prior.shape != (self.count,) or prior.dtype != torch.float32
                or prior.device != self.views[0].weight.device or not torch.isfinite(prior).all()
                or bool(((prior < 0) | (prior > 1)).any()) or len(scopes) != len(self.views)):
            raise ValueError('invalid extent preparation query')
        positive_mass = torch.zeros_like(prior); known_mass = torch.zeros_like(prior)
        cost = torch.zeros_like(prior); active = []
        for vi, (view, scope) in enumerate(zip(self.views, scopes)):
            if set(scope) != {'target', 'positive', 'negative', 'confidence'}:
                raise ValueError('unsupported scope contract')
            y, p, n, c = (scope[k].to(prior.device) for k in ('target', 'positive', 'negative', 'confidence'))
            if (y.shape != view.valid.shape or y.dtype != torch.float32 or p.shape != y.shape or n.shape != y.shape
                    or p.dtype != torch.bool or n.dtype != torch.bool or c.numel() != 1 or c.dtype != torch.float32
                    or not torch.isfinite(y).all() or bool(((y < 0) | (y > 1)).any())
                    or not torch.isfinite(c).all() or bool(c < 0) or bool(c > 1)
                    or bool((p & n).any()) or bool(((p | n) & ~view.valid).any())):
                raise ValueError('invalid extent scope evidence')
            if not bool(c > 0 and p.any()): continue
            active.append(vi)
            known = p * c + n * c
            op = view.operator()
            known_mass += torch.sparse.mm(op.T, known[:, None])[:, 0]
            positive_mass += torch.sparse.mm(op.T, (known * y)[:, None])[:, 0]
            cost += view.footprint * c
        initial = torch.where(known_mass > 1e-8, positive_mass / known_mass.clamp_min(1e-8), prior)
        normalizer = torch.dot(cost, initial).clamp_min(1e-8)
        cost /= normalizer
        return initial, cost, dict(active_views=active, fallback=not active,
                                   supported_elements=int((known_mass > 1e-8).sum()),
                                   footprint_normalizer=float(normalizer))


@torch.no_grad()
def compile_extent_preparation_view(field, camera, height, width, pixels, *, member_domain):
    """Compile the retained observation function with all occluders present."""
    from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
    from radio_gs.querying.typed_evidence import visible_operator
    from radio_gs.pipeline.subset import sample_operator_rows
    from .support_footprint import compile_support_footprint
    if member_domain.dtype != torch.bool or member_domain.shape != (len(field.xyz),):
        raise ValueError('invalid extent preparation member domain')
    k = camera.intrinsic.to(field.xyz).clone()
    k[0] *= width / camera.width; k[1] *= height / camera.height
    renderer = SimpleNamespace(scaled_intrinsics=lambda w, h: k,
        near_plane=.001 * field.world_units_per_metre, far_plane=100 * field.world_units_per_metre)
    geo = SimpleNamespace(get_xyz=lambda: field.xyz, get_rotation=lambda: field.quaternions,
                          get_scaling=lambda: field.scales, get_opacity=lambda: field.opacities)
    hit = rasterize_single_view_contributions(geo, renderer, camera.world_to_camera.to(field.xyz),
                                             height=height, width=width, include_depths=False)
    hit['base_weights'] = hit['weights']
    op, _, valid = visible_operator(hit, member_domain, height * width, False)
    pixels = pixels.to(field.xyz.device)
    op = sample_operator_rows(op, pixels)
    return dict(index=op.indices().cpu(), weight=op.values().cpu(), valid=valid[pixels, 0].cpu(),
                footprint=compile_support_footprint(field, camera, height, width,
                                                    member_domain=member_domain).cpu())
