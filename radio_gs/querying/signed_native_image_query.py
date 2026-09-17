"""Temporary signed-reference conditioning over a frozen native image field.

The caller supplies an authorized reference rendering operator, never target
labels. This changes a single interactive query, not map or encoder parameters.
"""
import math
import torch
from torch.nn import functional as F


def condition_signed_native_image(field, query, render_reference, positive, negative,
                                  *, steps=120, maximum_logit_correction=8.):
    if query.shape[0]!=1 or field.native_decoder is None:
        raise ValueError('one native image query required')
    if positive.shape!=negative.shape or positive.dtype!=torch.bool or negative.dtype!=torch.bool or (positive&negative).any():
        raise ValueError('disjoint signed Boolean reference masks required')
    if steps<0 or not math.isfinite(maximum_logit_correction) or maximum_logit_correction<=0:
        raise ValueError('invalid signed-query budget')
    with torch.no_grad():
        base=field.logits(query,'image').detach()
        z=field.latent.detach()
        if field.native_semantic_dim is not None:
            n=field.native_semantic_dim
            features=torch.cat([F.normalize(z[:,:n],dim=-1),F.normalize(z[:,n:],dim=-1)],-1)
        else:features=F.normalize(z,dim=-1)
        features=torch.cat([features,torch.ones(len(z),1,device=z.device,dtype=z.dtype)],-1)
    coefficient=torch.zeros(features.shape[1],1,device=z.device,requires_grad=True)
    def response():
        correction=maximum_logit_correction*torch.tanh(features@coefficient)
        return (base+correction).sigmoid()*field.observed_for('image')[:,None]
    with torch.no_grad():
        initial,valid=render_reference(response())
        if initial.shape!=positive.shape or valid.shape!=positive.shape or valid.dtype!=torch.bool:
            raise ValueError('signed masks and reference renderer domain differ')
        pos=positive&valid;neg=negative&valid
        if not pos.any() or not neg.any():raise ValueError('both signs need supported reference pixels')
    def loss_and_error(probability):
        loss=-.5*(probability[pos].clamp_min(1e-6).log().mean()+(1-probability[neg]).clamp_min(1e-6).log().mean())
        error=.5*((probability[pos]<.5).float().mean()+(probability[neg]>=.5).float().mean())
        return loss,error
    initial_loss,initial_error=loss_and_error(initial)
    best=(float(initial_error),float(initial_loss));best_coeff=coefficient.detach().clone();chosen=0
    optimizer=torch.optim.Adam([coefficient],lr=.03)
    with torch.enable_grad():
        for step in range(steps):
            probability,current_valid=render_reference(response())
            if not torch.equal(current_valid,valid):raise ValueError('query-dependent geometry support')
            loss,error=loss_and_error(probability)
            key=(float(error.detach()),float(loss.detach()))
            if key<best:best=key;best_coeff=coefficient.detach().clone();chosen=step
            if key[0]==0:break
            optimizer.zero_grad();(loss+.001*coefficient.square().mean()).backward();optimizer.step()
        with torch.no_grad():
            probability,_=render_reference(response());loss,error=loss_and_error(probability);key=(float(error),float(loss))
            if key<best:best=key;best_coeff=coefficient.detach().clone();chosen=step+1 if steps else 0
            coefficient.copy_(best_coeff);final=response().detach();reference,_=render_reference(final)
    return dict(scores=final,coefficient=best_coeff,report=dict(
        schema='signed_native_image_query_conditioning.v1',requested_steps=steps,selected_step=chosen,
        maximum_logit_correction=maximum_logit_correction,initial_balanced_error=float(initial_error),final_balanced_error=best[0],
        supported_positive_pixels=int(pos.sum()),unsupported_positive_pixels=int((positive&~valid).sum()),
        supported_negative_pixels=int(neg.sum()),unsupported_negative_pixels=int((negative&~valid).sum()),
        all_supported_signs_satisfied=best[0]==0,
        positive_agreement=float((reference[positive]>=.5).float().mean()),negative_agreement=float((reference[negative]<.5).float().mean()),
        map_and_encoder_updated=False,target_supervision_used=False,
        policy='minimum balanced supported sign error then BCE; unsupported positive pixels remain failures in full agreement'))
