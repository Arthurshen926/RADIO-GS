"""Current legal reference raster -> shared finite actual-set decisions.

The reference camera is a query input, unlike historical source cameras. Optical
functions are instantiated from this camera and the one field; no source loader,
target camera, target labels, query parameter fitting or new teacher is involved.
This full-mask contract does not stand in for sparse positive/negative scribbles.
"""
from dataclasses import replace
import numpy as np
import torch
from .extent_factors import ExtentFactors, compile_extent_view
from .extent_preparation import ExtentPreparation, compile_extent_preparation_view
from .extent_decision_linked import LinkedExtentDecision
from .reference_scope import query as compose_reference


def reference_scope(reference, valid):
    y=reference.flatten().float()
    if (y.shape!=valid.shape or valid.dtype!=torch.bool or not torch.isfinite(y).all()
            or bool(((y<0)|(y>1)).any()) or not bool((y>0).any())):
        raise ValueError('a complete nonempty current reference raster is required')
    # Preserve fractional area targets; background permission comes from the
    # explicitly supplied complete 2D mask, not lack of a sparse positive mark.
    return dict(target=y,positive=(y>0)&valid,negative=(y==0)&valid,
                confidence=y.new_ones(()))


def prepare_reference(preparation, prior, scope, initializer):
    if initializer not in {'prior','lifted'}:raise ValueError('unknown reference initializer')
    lifted,cost,info=preparation(prior,[scope])
    if initializer=='lifted':return lifted,cost,info
    if not info['active_views']:return prior,torch.zeros_like(prior),info
    raw=preparation.views[0].footprint
    normalizer=torch.dot(raw,prior).clamp_min(1e-8)
    return prior,raw/normalizer,dict(info,footprint_normalizer=float(normalizer))


@torch.no_grad()
def query(field,packet,camera,reference,*,initializer='prior',decision_cutoff=.5,observation_weight=0.,audit_state=None):
    reference=torch.as_tensor(reference,device=field.xyz.device,dtype=torch.float32)
    if reference.ndim!=2:raise ValueError('reference mask must be a two-dimensional raster')
    base=compose_reference(field,packet,camera,reference,world_scale=field.world_units_per_metre,
                           observation='scene',member_conditioning=True,member_readout='admission')
    h,w=reference.shape;pixels=torch.arange(h*w);domain=field.member_available
    optical=compile_extent_view(field,camera,h,w,pixels,member_domain=domain)
    lift=compile_extent_preparation_view(field,camera,h,w,pixels,member_domain=domain)
    factors=ExtentFactors({'views.0.'+k:v for k,v in optical.items()},len(field.xyz),field.opacities.cpu())
    preparation=ExtentPreparation({'views.0.'+k:v for k,v in lift.items()},len(field.xyz)).to(field.xyz.device)
    scope=reference_scope(reference,preparation.views[0].valid)
    prior=base.selected.float();initial,cost,info=prepare_reference(preparation,prior,scope,initializer)
    engine=LinkedExtentDecision(factors,len(field.xyz))
    confidence=np.array([1. if info['active_views'] else 0.])
    extra={}
    if observation_weight:
        from .extent_decision_coupled import CoupledExtentDecision
        engine=CoupledExtentDecision(factors,preparation,len(field.xyz))
        extra['observation_weight']=observation_weight
    selected,report=engine.solve((initial>.5).cpu().numpy(),scope['target'].cpu().numpy(),
        scope['positive'].cpu().numpy(),scope['negative'].cpu().numpy(),confidence,cost.cpu().numpy(),
        max_passes=4,decision_cutoff=decision_cutoff,**extra)
    selected=torch.from_numpy(selected).to(field.xyz.device)
    if bool(selected[~domain].any()):raise RuntimeError('reference extent selected an unavailable member')
    if audit_state is not None:
        audit_state.update(optical=optical,lift=lift,scope={k:v.cpu() for k,v in scope.items()},
            prior=prior.cpu(),initial=initial.cpu(),cost=cost.cpu(),confidence=torch.from_numpy(confidence),
            selected=selected.cpu(),reference=reference.cpu(),initializer=initializer,
            decision_cutoff=decision_cutoff,observation_weight=float(observation_weight),domain=domain.cpu())
    return replace(base,membership_scores=selected.float(),selected=selected,
        diagnostics=dict(base.diagnostics,shared_extent_decision=True,extent_initializer=initializer,
            extent_preparation=info,extent_decision=report,decision_alpha_cutoff=float(decision_cutoff),current_query_geometry=True,
            construction_source_geometry_read=False,reference_composition_diagnostics_are_initializer=True,
            source_access=False,query_parameter_fitting=False,new_field_state_bytes=0))
