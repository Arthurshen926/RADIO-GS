"""Native-coordinate category priors without historical scene parameters.

These heads apply to an explicitly mutually exclusive category vocabulary,
not arbitrary overlapping text sets or reference-instance queries.
"""
import torch
from torch.nn import functional as F


def categorical_prior(cosine, temperature=100.):
    z=temperature*cosine
    # Exact one-vs-rest log odds preserve low-score rankings even when the
    # evaluator selects a smaller category subset. Probability clipping would
    # collapse these ranks and create artificial ties.
    empty=torch.full_like(z[:,:1],-torch.inf)
    left=torch.cat([empty,torch.logcumsumexp(z,dim=1)[:,:-1]],dim=1)
    right=torch.cat([torch.logcumsumexp(z.flip(1),dim=1).flip(1)[:,1:],empty],dim=1)
    return z-torch.logaddexp(left,right)


def fit_exclusive_source_head(cosine,views,steps=200):
    positive=torch.zeros_like(cosine);negative=torch.zeros_like(cosine)
    for e in views:
        single=(e.positive.sum(1,keepdim=True)==1)
        weight=(e.positive.float()*e.confidence).sum(1,keepdim=True)*single
        positive+=torch.sparse.mm(e.adjoint,e.positive.float()*weight)
        negative+=torch.sparse.mm(e.adjoint,(~e.positive).float()*weight)
    pos_mass=positive.sum(0);neg_mass=negative.sum(0)
    active=(pos_mass>0)&(neg_mass>0)
    wp=positive/pos_mass.clamp_min(1e-8);wn=negative/neg_mass.clamp_min(1e-8)
    threshold=torch.full((cosine.shape[1],),.1,device=cosine.device,requires_grad=True)
    logscale=torch.tensor(4.605170186,device=cosine.device,requires_grad=True)
    opt=torch.optim.Adam([dict(params=[threshold],lr=.001),dict(params=[logscale],lr=.01)])
    for _ in range(steps):
        opt.zero_grad()
        z=logscale.clamp(0,6).exp()*(cosine-threshold)
        terms=(wp*F.softplus(-z)+wn*F.softplus(z)).sum(0)
        loss=(terms*active).sum()/active.sum().clamp_min(1)
        loss.backward();opt.step()
        with torch.no_grad():threshold.clamp_(-1,1)
    with torch.no_grad():
        learned=logscale.clamp(0,6).exp()*(cosine-threshold)
        logits=torch.where(active[None],learned,categorical_prior(cosine))
    return logits,dict(threshold=threshold.detach().cpu(),logscale=logscale.detach().cpu(),active=active.cpu(),positive_mass=pos_mass.cpu(),negative_mass=neg_mass.cpu(),steps=steps,loss=float(loss.detach()),outside_source_positive_union='unknown',overlapping_source_positives='unknown',negative_evidence='exclusive other-category eroded positive support',historical_parameters_read=False)


def exclusive_source_views(views):
    """Other confirmed category cores exclude a class; uncovered pixels do not."""
    from radio_gs.querying.typed_evidence import ViewEvidence
    result=[]
    for e in views:
        single=(e.target>.5).sum(1,keepdim=True)==1
        positive=e.positive&single
        known=positive.any(1,keepdim=True)
        negative=(e.target<=.5)&known
        confidence=(e.positive.float()*e.confidence).amax(1,keepdim=True)*known
        visible=e.visible.any(1,keepdim=True).expand_as(e.visible)
        result.append(ViewEvidence(e.operator,e.adjoint,e.evidence_operator,
            positive.float(),positive,negative,visible,e.frame,confidence))
    return result
