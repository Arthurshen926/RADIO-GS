"""Same-view partial-region union before cross-view evidence integration.

Max positive / min negative are unions of field marginals. They do NOT commute
with source rasterization and are not exact v10 pixel-union reconstruction.
"""
import torch


@torch.no_grad()
def prepare_union_parts(evidence,prior,available,region_probability):
    if available.dtype!=torch.bool or prior.shape!=(evidence.count,) or available.shape!=prior.shape:
        raise ValueError('unaligned field query')
    if region_probability.shape!=evidence.views.shape or not torch.isfinite(region_probability).all() or bool(((region_probability<0)|(region_probability>1)).any()):
        raise ValueError('invalid region acceptance')
    if not torch.isfinite(prior).all() or bool(((prior<0)|(prior>1)).any()):raise ValueError('invalid prior')
    positive=evidence.matrix('positive');negative=evidence.matrix('negative')
    mass=torch.sparse.mm(positive.transpose(0,1),available.to(prior)[:,None]).flatten()
    preference=torch.where(mass>0,(2*region_probability.to(prior)-1).clamp(0,1),0)
    p=prior.new_zeros(evidence.count,evidence.view_count);n=torch.zeros_like(p)
    confidence=prior.new_zeros(evidence.view_count)
    for view in range(evidence.view_count):
        ids=torch.where((evidence.views==view)&(preference>0))[0]
        if not len(ids):continue
        confidence[view]=preference[ids].max()
        pv=prior.new_zeros(evidence.count);nv=prior.new_ones(evidence.count)
        # Bound temporary memory without changing candidate membership or union.
        for chunk in ids.split(16):
            weights=prior.new_zeros(len(evidence.views),len(chunk))
            weights[chunk,torch.arange(len(chunk),device=prior.device)]=1
            pv=torch.maximum(pv,torch.sparse.mm(positive,weights).max(1).values)
            nv=torch.minimum(nv,torch.sparse.mm(negative,weights).min(1).values)
        p[:,view]=pv*confidence[view];n[:,view]=nv*confidence[view]
    return p,n,dict(admitted_regions=int((preference>0).sum()),active_views=int((confidence>0).sum()),
        confidence_sum=float(confidence.sum()),native_region_acceptance=True,
        same_view_union=True,negative_requires_all_accepted_regions=True,
        field_marginal_union_not_exact_source_raster_union=True)
