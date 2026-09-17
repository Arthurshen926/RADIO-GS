"""Query-transient signed evidence fitting; scene and geometry remain frozen."""
import torch
from torch.nn import functional as F


class SignedEvidenceUnavailable(ValueError):
    """A valid registered point has no observed semantic support in this map."""
    def __init__(self, seed_index):
        self.seed_index = seed_index
        super().__init__(f'signed pixel lacks observed feature support (seed {seed_index})')


def observed_seed_contributions(field, rows, weight, seed_index, *, device, dtype):
    """Use the same support gate as full-scene feature rendering.

    Weights must be all front-to-back contributions of this pixel, including
    unobserved elements. Passing an observed-only subset changes the contract.
    """
    rows = torch.as_tensor(rows, device=device)
    weight = torch.as_tensor(weight, device=device, dtype=dtype)
    if rows.ndim != 1 or (rows.numel() and rows.dtype not in (torch.int32, torch.int64)):
        raise ValueError('integer source-pixel row indices required')
    rows = rows.long()
    if weight.shape != rows.shape or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('invalid source-pixel contributions')
    if (rows < 0).any() or (rows >= len(field.latent)).any():
        raise ValueError('seed row outside field')
    total = weight.sum()
    keep = field.observed[rows]
    selected = weight[keep]
    mass = selected.sum()
    if total <= 1e-8 or mass / total.clamp_min(1e-8) <= .5:
        raise SignedEvidenceUnavailable(seed_index)
    return rows[keep], (selected / mass.clamp_min(1e-8)).detach()


def refine_signed_query(field, query, seed_rows, seed_weights, labels, *,
                        steps=64, learning_rate=.05, prior_weight=.01, only_if_violated=True,
                        stop_when_satisfied=False, feasibility_margin=.001):
    """Fit one prompt direction using authorized source-pixel mixtures only.

    The caller must bind pixel contributions to the field's frozen geometry.
    Pixels require majority observed support before unknown rows are excluded;
    the remaining contribution is normalized,
    exactly as in feature-supported rendering. No field parameter is optimized.
    Inconsistent or unrepresentable constraints are reported, never clamped.
    """
    if query.ndim != 2 or len(query) != 1 or steps < 0:
        raise ValueError('one prompt query and nonnegative steps required')
    if not torch.isfinite(query).all():
        raise ValueError('prompt query must be finite')
    labels=torch.as_tensor(labels,device=query.device,dtype=query.dtype)
    if len(seed_rows)!=len(seed_weights) or labels.shape!=(len(seed_rows),) or len(labels)<2:
        raise ValueError('signed seed axes differ')
    if not torch.isfinite(labels).all() or not ((labels==0)|(labels==1)).all() or labels.min()==labels.max():
        raise ValueError('explicit positive and negative labels required')
    if learning_rate<=0 or prior_weight<0 or not 0<=feasibility_margin<.5:
        raise ValueError('invalid refinement parameters')
    features=[];weights=[]
    with torch.no_grad():
        initial=F.normalize(field.query_adapters['prompt'](query),dim=-1).detach()
        if initial.norm() == 0:
            raise ValueError('prompt query has no direction')
        for seed_index,(rows,weight) in enumerate(zip(seed_rows,seed_weights)):
            rows, weight = observed_seed_contributions(
                field, rows, weight, seed_index, device=query.device, dtype=query.dtype)
            features.append(F.normalize(field.latent[rows].detach(),dim=-1))
            weights.append(weight)
    def response(direction):
        direction=F.normalize(direction,dim=-1)
        return torch.stack([(torch.sigmoid(field.cosine_logits((z@direction.T)[:,0],'prompt',detach_calibration=True))*w).sum() for z,w in zip(features,weights)])
    def objective(direction):
        probabilities=response(direction)
        return F.binary_cross_entropy(probabilities.clamp(1e-7,1-1e-7),labels)+prior_weight*(1-(F.normalize(direction,dim=-1)*initial).sum())
    before=response(initial).detach();best=initial.clone();best_loss=float(objective(initial).detach())
    before_satisfied=torch.where(labels.bool(),before>.5,before<=.5)
    # Match the actual retrieval decision. Already-correct user evidence does
    # not justify rotating the query just to raise its confidence.
    applied=not (only_if_violated and bool(before_satisfied.all()))
    actual_steps=steps if applied else 0
    with torch.enable_grad():
        direction=initial.clone().requires_grad_(True)
        optimizer=torch.optim.Adam([direction],lr=learning_rate)
        for iteration in range(actual_steps):
            optimizer.zero_grad(set_to_none=True);loss=objective(direction)
            if not torch.isfinite(loss):raise ValueError('nonfinite signed-query objective')
            loss.backward();optimizer.step()
            value=float(objective(direction).detach())
            if value<best_loss:
                best_loss=value;best=F.normalize(direction.detach(),dim=-1).clone()
            if stop_when_satisfied:
                with torch.no_grad():
                    candidate=F.normalize(direction.detach(),dim=-1)
                    def feasible(q):
                        p=response(q)
                        return bool(torch.where(labels.bool(),p>.5+feasibility_margin,p<=.5-feasibility_margin).all())
                    if feasible(candidate):
                        # Backtrack toward the original direction on this
                        # segment. This is not a global closest-point solver.
                        low,high=0.,1.
                        for _ in range(24):
                            middle=(low+high)/2
                            trial=F.normalize((1-middle)*initial+middle*candidate,dim=-1)
                            if feasible(trial):high=middle
                            else:low=middle
                        best=F.normalize((1-high)*initial+high*candidate,dim=-1).clone()
                        actual_steps=iteration+1
                        break
    after=response(best).detach()
    satisfied=torch.where(labels.bool(),after>.5,after<=.5)
    confidence=torch.where(labels.bool(),after>=.9,after<=.1)
    return {'latent_query':best,'seed_before':before,'seed_after':after,
            'constraints_satisfied':satisfied,'confidence_targets_satisfied':confidence,
            'before_constraints_satisfied':before_satisfied,'refinement_applied':applied and actual_steps>0,
            'steps':actual_steps,'requested_steps':steps,'field_updated':False,
            'stop_when_satisfied':stop_when_satisfied,'feasibility_margin':feasibility_margin,
            'cosine_to_initial':float((best*initial).sum())}


def retrieve_latent_query(field, latent_query, modality=None):
    """Same element posterior as ordinary retrieval, with a conditioned query."""
    if latent_query.ndim!=2 or latent_query.shape[1]!=field.latent.shape[1] or not torch.isfinite(latent_query).all():
        raise ValueError('invalid latent query')
    if modality is None:
        if 'prompt' in field.query_adapters:modality='prompt'
        elif len(field.query_adapters)==1:modality=next(iter(field.query_adapters))
        else:raise ValueError('latent query requires an explicit readout modality')
    if getattr(field,'native_decoder',None) is not None and modality in ('text','image'):
        raise ValueError('native decoder readout requires the original native query, not a normalized latent direction')
    scores=torch.sigmoid(field.cosine_logits(F.normalize(field.latent,dim=-1)@F.normalize(latent_query,dim=-1).T,modality))
    return {'scores':torch.where(field.observed[:,None],scores,torch.zeros_like(scores)),
            'valid':field.observed}


def condition_signed_query(field, query, seed_rows, seed_weights, labels, **options):
    """Keep the original query when valid signed evidence is unobservable.

    Registration/shape errors still fail closed. Only missing observed map
    features produce an explicit unavailable result, rather than removing the
    query from evaluation or silently inventing point evidence.
    """
    try:
        result=refine_signed_query(field,query,seed_rows,seed_weights,labels,**options)
        result['available']=True
        return result
    except SignedEvidenceUnavailable as error:
        with torch.no_grad():
            initial=F.normalize(field.query_adapters['prompt'](query),dim=-1)
        return {'latent_query':initial,'available':False,'reason':str(error),
                'missing_seed_index':error.seed_index,'refinement_applied':False,
                'constraints_satisfied':None,'confidence_targets_satisfied':None,
                'before_constraints_satisfied':None,'seed_before':None,'seed_after':None,
                'steps':0,'field_updated':False}
