"""Joint source inference for an explicitly mutually exclusive class ontology.

Only positive source detections identify a class. Outside their union stays
unknown. Overlapping positive detections contribute a soft class distribution,
not mutually contradictory binary labels. Not applicable to nested text sets.
"""
import torch


def simplex_projection(value):
    ordered=value.sort(dim=1,descending=True).values
    excess=ordered.cumsum(1)-1
    rank=torch.arange(1,value.shape[1]+1,device=value.device,dtype=value.dtype)[None]
    count=(ordered-excess/rank>0).sum(1).clamp_min(1)
    offset=excess.gather(1,(count-1)[:,None])/count[:,None]
    return (value-offset).clamp_min(0)


@torch.inference_mode()
def solve_categorical(prior_logits,views,steps=60,prior_strength=.1):
    prior=prior_logits.softmax(dim=1)
    examples=[];degree=torch.zeros(len(prior),1,device=prior.device)
    for e in views:
        evidence=e.positive.float()*e.confidence
        mass=evidence.sum(1,keepdim=True)
        target=evidence/mass.clamp_min(1e-8)
        # A mask certifies class evidence only inside its eroded visible support.
        weight=evidence.max(1,keepdim=True).values
        examples.append((e,target,weight))
        degree+=torch.sparse.mm(e.adjoint,weight)
    value=prior.clone();denominator=degree+prior_strength
    for _ in range(steps):
        gradient=prior_strength*(prior-value)
        for e,target,weight in examples:
            residual=weight*(target-torch.sparse.mm(e.operator,value))
            gradient+=torch.sparse.mm(e.adjoint,residual)
        value=simplex_projection(value+.95*gradient/denominator)
    value=torch.where(degree>0,value,prior)
    return value,dict(steps=steps,prior_strength=prior_strength,source_views=len(views),active_elements=int((degree>0).sum()),outside_positive_union='unknown',ontology='mutually exclusive declared classes',output='joint categorical distribution, not independent object membership')
