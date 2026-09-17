"""Source-prompt-only feasibility admission and stable signed query fitting.

A fixed logit cap can make a supplied sign impossible regardless of optimizer.
Select a bounded cap using the authorized reference operator, not target data.
Negative probabilities are rendered directly to avoid 1-sigmoid cancellation.
"""
import torch
from torch.nn import functional as F


def condition_signed_native_image_adaptive(field,query,render_reference,positive,negative,*,steps=240,bounds=(8.,16.,32.,64.),objective='bce'):
    if query.ndim!=2 or len(query)!=1 or not torch.isfinite(query).all() or field.native_decoder is None:
        raise ValueError('one finite native image query required')
    if positive.shape!=negative.shape or positive.dtype!=torch.bool or negative.dtype!=torch.bool or (positive&negative).any():
        raise ValueError('disjoint signed Boolean masks required')
    if objective not in ('bce','minimal_hinge'):raise ValueError('unknown signed objective')
    if steps<0 or not bounds or any(not 0<float(b)<=64 for b in bounds) or list(bounds)!=sorted(set(bounds)):
        raise ValueError('invalid bounded signed policy')
    with torch.no_grad():
        base=field.logits(query,'image').detach();z=field.latent.detach();n=field.native_semantic_dim
        features=F.normalize(z,dim=-1) if n is None else torch.cat([F.normalize(z[:,:n],dim=-1),F.normalize(z[:,n:],dim=-1)],-1)
        features=torch.cat([features,torch.ones(len(z),1,device=z.device,dtype=z.dtype)],-1)
        observed=field.observed_for('image').float()[:,None]
        initial,valid=render_reference(base.sigmoid()*observed)
        if initial.shape!=positive.shape or valid.shape!=positive.shape or valid.dtype!=torch.bool:raise ValueError('reference domain differs')
        pos,neg=positive&valid,negative&valid
        if not pos.any() or not neg.any():raise ValueError('both signs need supported reference pixels')
        trials=[]
        for bound in bounds:
            ceiling,v=render_reference((base+bound).sigmoid()*observed);floor,v2=render_reference((base-bound).sigmoid()*observed)
            if not torch.equal(v,valid) or not torch.equal(v2,valid):raise ValueError('query-dependent support')
            ip=int((ceiling[pos]<.5).sum());ineg=int((floor[neg]>=.5).sum())
            trials.append(dict(bound=float(bound),impossible_positive=ip,impossible_negative=ineg))
            if ip+ineg==0:break
        bound=trials[-1]['bound']
    coefficient=torch.zeros(features.shape[1],1,device=z.device,requires_grad=True)
    def probabilities():
        logits=base+bound*torch.tanh(features@coefficient)
        return logits.sigmoid()*observed,(-logits).sigmoid()*observed
    def evaluate():
        yes,no=probabilities();p,v=render_reference(yes);n,v2=render_reference(no)
        if not torch.equal(v,valid) or not torch.equal(v2,valid):raise ValueError('query-dependent support')
        tiny=torch.finfo(p.dtype).tiny
        if objective=='bce':
            loss=-.5*(p[pos].clamp_min(tiny).log().mean()+n[neg].clamp_min(tiny).log().mean())
        else:
            margin=torch.tensor(.52,device=p.device).log()
            hinge=.5*(F.relu(margin-p[pos].clamp_min(tiny).log()).mean()+F.relu(margin-n[neg].clamp_min(tiny).log()).mean())
            logits=base+bound*torch.tanh(features@coefficient)
            prior=(F.binary_cross_entropy_with_logits(logits,base.sigmoid(),reduction='none')-F.binary_cross_entropy_with_logits(base,base.sigmoid(),reduction='none'))[observed.bool()].mean()
            loss=hinge+.1*prior
        error=.5*((p[pos]<.5).float().mean()+(p[neg]>=.5).float().mean())
        return loss,error,p
    with torch.no_grad():
        loss,error,_=evaluate();best=(float(error),float(loss));initial_error=best[0];state=coefficient.detach().clone();chosen=0
    optimizer=torch.optim.Adam([coefficient],lr=.01)
    with torch.enable_grad():
        for step in range(steps):
            loss,error,_=evaluate()
            if not torch.isfinite(loss):raise ValueError('nonfinite signed objective')
            key=(float(error.detach()),float(loss.detach()))
            if key<best:best=key;state=coefficient.detach().clone();chosen=step
            if key[0]==0:break
            optimizer.zero_grad();(loss+.001*coefficient.square().mean()).backward();optimizer.step()
        with torch.no_grad():
            loss,error,_=evaluate();key=(float(error),float(loss))
            if key<best:best=key;state=coefficient.detach().clone();chosen=step+1 if steps else 0
            coefficient.copy_(state);scores,_=probabilities();_,_,reference=evaluate()
    unsupported_positive=int((positive&~valid).sum());supported_success=best[0]==0
    return dict(scores=scores.detach(),coefficient=state,report=dict(schema='adaptive_signed_native_image.v1',objective=objective,native_prediction_prior_weight=.1 if objective=='minimal_hinge' else 0.,requested_steps=steps,selected_step=chosen,maximum_logit_correction=bound,feasibility_trials=trials,initial_balanced_error=initial_error,final_balanced_error=best[0],supported_positive_pixels=int(pos.sum()),supported_negative_pixels=int(neg.sum()),unsupported_positive_pixels=unsupported_positive,unsupported_negative_pixels=int((negative&~valid).sum()),all_supported_signs_satisfied=supported_success,all_input_signs_satisfied=supported_success and unsupported_positive==0,positive_agreement=float((reference[positive]>=.5).float().mean()),negative_agreement=float((reference[negative]<.5).float().mean()),map_and_encoder_updated=False,target_supervision_used=False,policy='Smallest necessary-feasible cap in fixed [8,16,32,64], then supported signed error/BCE; direct rendered complement probability; no guaranteed joint feasibility'))
