"""Query-conditioned set evidence, separate from crop-level semantic features.

A detection supports the requested set in one source view. It is not a verified
cross-view identity and does not relabel every Gaussian intersecting its crop.
This module consumes no query strings or GT; it solves element membership.
"""
from dataclasses import dataclass
from enum import Enum
import torch
from torch.nn import functional as F


class SetType(str, Enum):
    TEXT_UNION = 'text_union'
    REFERENCE_INSTANCE = 'reference_instance'


@dataclass
class ViewEvidence:
    operator: torch.Tensor
    adjoint: torch.Tensor
    evidence_operator: torch.Tensor
    target: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    visible: torch.Tensor
    frame: int
    confidence: torch.Tensor


def visible_operator(hit, observed, pixels, ownership):
    pid=hit['pixel_ids'].long().to(observed.device)
    gid=hit['gaussian_ids'].long().to(observed.device)
    weight=hit['base_weights'].float().to(observed.device)
    total=torch.zeros(pixels,device=weight.device).scatter_add_(0,pid,weight)
    weight=weight*observed[gid]
    mass=torch.zeros_like(total).scatter_add_(0,pid,weight)
    visible=(mass>1e-8)&(mass/total.clamp_min(1e-8)>.5)
    forward=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,len(observed))).coalesce()
    if ownership:
        peak=torch.zeros_like(total).scatter_reduce_(0,pid,weight,reduce='amax',include_self=True)
        weight=weight*(weight>=.5*peak[pid])
        mass=torch.zeros_like(total).scatter_add_(0,pid,weight)
    op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,len(observed))).coalesce()
    return forward,op,visible[:,None]


def view_evidence(op, evidence_op, visible, probability, detected, frame, height, width, confidence=None, boundary_radius=1):
    binary=(probability.T.reshape(-1,1,height,width)>.5).float()
    kernel=2*boundary_radius+1
    positive=(-F.max_pool2d(-binary,kernel,1,boundary_radius)>.5).flatten(1).T
    negative=(F.max_pool2d(binary,kernel,1,boundary_radius)<.5).flatten(1).T
    active=visible&detected[None]
    return ViewEvidence(op,evidence_op.T.coalesce(),evidence_op,probability,positive&active,negative&active,active,frame,torch.ones(1,probability.shape[1],device=probability.device) if confidence is None else confidence[None])


@torch.inference_mode()
def solve_set(prior, views, task: SetType, corroborate=True, steps=60, consistency=False,
              prior_strength=.1, censor_positive=False, evidence_mass_consistency=False):
    """Bounded robust inverse rendering; unsupported membership keeps the prior.

    Cross-view negative admission uses OTHER views only. Strong contradictory
    positives turn evidence into unknown instead of forcing one name/instance.
    Same-view positive and negative labels are eroded teacher masks, not GT.
    Forward rendering keeps the full operator; gated adjoints restrict evidence
    assignment and residual updates, not the deployment renderer. This is a
    visibility-preconditioned fixed-point update, not an exact least-squares solve.
    evidence_mass_consistency weights other-view estimates by their actual
    confidence-weighted contribution mass rather than binary view presence.
    This remains agreement between pseudo-labels, not calibrated correctness.
    """
    task=SetType(task)
    positive_votes=[];negative_votes=[];denominators=[]
    for e in views:
        visibility=torch.sparse.mm(e.adjoint,e.visible.float())
        p=torch.sparse.mm(e.adjoint,e.positive.float())/visibility.clamp_min(1e-8)
        n=torch.sparse.mm(e.adjoint,e.negative.float())/visibility.clamp_min(1e-8)
        positive_votes.append((p>.6).float());negative_votes.append((n>.8).float())
        denominators.append(visibility)
    ps=torch.stack(positive_votes).sum(0);ns=torch.stack(negative_votes).sum(0)
    estimates=[];supports=[]
    if consistency:
        for e in views:
            k=(e.positive|e.negative).float()*e.confidence
            mass=torch.sparse.mm(e.adjoint,k)
            estimates.append(torch.sparse.mm(e.adjoint,e.target*k)/mass.clamp_min(1e-8))
            supports.append(mass if evidence_mass_consistency else (mass>1e-8).float())
        estimate_sum=sum(x*v for x,v in zip(estimates,supports));support_sum=sum(supports)
    # One-sided censoring: an independent positive observation makes exclusion
    # uncertain. Repeated absence is NOT treated as proof of complete extent.
    positive_extent=[]
    if censor_positive:
        for e,visibility in zip(views,denominators):
            positive_extent.append(torch.sparse.mm(e.adjoint,e.positive.float())/visibility.clamp_min(1e-8))
        ranked=torch.stack(positive_extent).topk(min(2,len(views)),dim=0)
    known=[];den=torch.zeros_like(prior);vis=torch.zeros_like(prior)
    for index,(e,p,n,visibility) in enumerate(zip(views,positive_votes,negative_votes,denominators)):
        if consistency:
            other_mass=support_sum-supports[index]
            other=(estimate_sum-estimates[index]*supports[index])/other_mass.clamp_min(1e-8 if evidence_mass_consistency else 1)
            coverage=torch.sparse.mm(e.operator,(other_mass>0).float())
            probability=(torch.sparse.mm(e.operator,other)/coverage.clamp_min(1e-8)).clamp(0,1)
            agreement=e.target*probability+(1-e.target)*(1-probability)
            reliability=torch.where(coverage>.5,agreement,torch.ones_like(agreement))
            k=(e.positive|e.negative).float()*reliability
        elif corroborate:
            other_p=ps-p;other_n=ns-n
            independent_negative=(other_n>=1)&(other_p==0)
            negative_support=torch.sparse.mm(e.evidence_operator,independent_negative.float())>.5
            k=e.positive|(e.negative&negative_support)
        else:k=e.positive|e.negative
        k=k.float()*e.confidence
        if censor_positive and len(views)>1:
            other=torch.where(ranked.indices[0]==index,ranked.values[1],ranked.values[0])
            protected=torch.sparse.mm(e.operator,other).clamp(0,1)
            k=k*torch.where(e.negative,1-protected,torch.ones_like(protected))
        known.append(k);den+=torch.sparse.mm(e.adjoint,k.float());vis+=visibility
    value=prior.clone()
    for _ in range(steps):
        gradient=torch.zeros_like(value)
        for e,k in zip(views,known):
            residual=(e.target-torch.sparse.mm(e.operator,value)).clamp(-.25,.25)*k
            gradient+=torch.sparse.mm(e.adjoint,residual)
        value.add_(.95*(gradient-prior_strength*vis*(value-prior))/(den+prior_strength*vis).clamp_min(1e-8)).clamp_(0,1)
    return value,dict(task=task.value,views=len(views),steps=steps,soft_consistency=consistency,
        conflicting_elements=int(((ps>0)&(ns>0)).sum()),
        positive_entries=sum(int(e.positive.sum()) for e in views),
        admitted_negative_entries=sum(float((k.float()*e.negative).sum()) for e,k in zip(views,known)),
        available_negative_entries=sum(int(e.negative.sum()) for e in views),
        active_elements=int((den>0).sum()),prior_strength=prior_strength,
        censor_positive=censor_positive,evidence_mass_consistency=evidence_mass_consistency)


@torch.no_grad()
def solve_rendered_evidence(initial, observations, steps=120):
    """Fit one element field to source-view evidence from several providers.

    Each observation is (visible operator, target, known mask, weight).
    A native/structural prior may be rendered from a different carrier: its
    pixel-space evidence uses the NEW carrier operator, never old element IDs.
    Unknown pixels contribute neither positive nor negative evidence. Weights
    specify provider strength, not probabilities of annotation correctness.
    This objective is shared by text, image and reference-prompt providers;
    it does not establish correspondence or identity on their behalf.

    Full visibility-normalized forward operators and their exact transposes
    define a bounded Huber inverse-rendering objective. Unsupported elements
    keep their supplied initialization; callers decide the abstention policy.
    """
    from .response_reconstruction import solve_rendered_evidence as reconstruct
    return reconstruct(initial, observations, steps=steps)
