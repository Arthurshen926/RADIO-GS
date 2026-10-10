"""Query-conditioned partial regions with leave-one-view-out exclusion protection.

Field-owned Gaussian coordinates only. This is a source-only hypothesis, not
v10 track identity or a calibrated complete-instance posterior.
"""
import torch


def integrate_views(prior, positive, negative, *, protect=False):
    if positive.ndim!=2 or negative.shape!=positive.shape or prior.shape!=(positive.shape[0],):
        raise ValueError('unaligned per-view member evidence')
    for x in [prior,positive,negative]:
        if not bool(torch.isfinite(x).all()) or bool(((x<0)|(x>1+1e-5)).any()):
            raise ValueError('invalid member evidence')
    exclusion=negative
    if protect:
        other_positive=(positive.sum(1,keepdim=True)-positive).clamp_min(0)
        known=positive+negative
        other_known=(known.sum(1,keepdim=True)-known).clamp_min(0)
        other=torch.where(other_known>0,other_positive/other_known.clamp_min(1e-12),0).clamp(0,1)
        exclusion=negative*(1-other)
    scores=(prior+positive.sum(1))/(1+positive.sum(1)+exclusion.sum(1))
    return scores,exclusion


@torch.no_grad()
def condition_parts(evidence, prior, available, *, protect=False):
    if prior.shape!=(evidence.count,) or available.shape!=prior.shape:
        raise ValueError('query and field rows differ')
    if available.dtype!=torch.bool or not torch.isfinite(prior).all() or bool(((prior<0)|(prior>1)).any()):
        raise ValueError('invalid query evidence')
    b=evidence.matrix('positive');device=prior.device
    mass=torch.sparse.mm(b.transpose(0,1),available.to(prior)[:,None]).flatten()
    foreground=torch.sparse.mm(b.transpose(0,1),(prior*available)[:,None]).flatten()
    mean=foreground/mass.clamp_min(1e-12)
    preference=torch.where(mass>0,(2*mean-1).clamp(0,1),0)
    views=evidence.views;count=evidence.view_count
    total=prior.new_zeros(count).scatter_add_(0,views,preference)
    confidence=prior.new_zeros(count).scatter_reduce_(0,views,preference,reduce='amax',include_self=True)
    coefficients=preference/total[views].clamp_min(1e-12)*confidence[views]
    ids=torch.where(coefficients>0)[0]
    weights=torch.sparse_coo_tensor(torch.stack([ids,views[ids]]),coefficients[ids],
        (len(views),count),device=device).to_dense()
    positive=torch.sparse.mm(b,weights)
    negative=torch.sparse.mm(evidence.matrix('negative'),weights)
    scores,exclusion=integrate_views(prior,positive,negative,protect=protect)
    scores=scores.masked_fill(~available,0)
    return scores,dict(admitted_regions=int((preference>0).sum()),active_views=int((total>0).sum()),
        confidence_sum=float(confidence.sum()),positive_mass=float(positive.sum()),
        negative_mass_before=float(negative.sum()),negative_mass_after=float(exclusion.sum()),
        other_view_protection=protect,source_access=False,new_training=False,
        input_condition='full query foreground probabilities, not inferred single-anchor positives',
        partial_regions_are_not_complete_instances=True,scores_calibrated=False,
        newly_selected=int(((scores>.5)&(prior<=.5)&available).sum()),
        newly_rejected=int(((scores<=.5)&(prior>.5)&available).sum()))
