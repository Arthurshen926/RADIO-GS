"""One-pass accumulation of the existing vote at full declared view coverage.

Detector-budget normalization is a per-query scalar; applying it after the
linear adjoint sums avoids keeping every view's sparse operator on the GPU.
The formula and original confidence budget are unchanged.
"""
from pathlib import Path
import torch
from radio_gs.querying.typed_evidence import visible_operator,view_evidence

@torch.inference_mode()
def streamed_category_vote(prior_logits,observed,scene,texts):
    device=observed.device;spec=scene['source_evidence'];h,w=spec['raster_hw']
    records=[torch.load(v['identity'],map_location='cpu') for v in spec['views']]
    ids=[records[0]['queries'].index(t) for t in texts]
    square=torch.zeros(len(texts));count=torch.zeros_like(square)
    for data in records:
        for j,qi in enumerate(ids):
            row=data['instances'][qi];square[j]+=row['foreground_margin'].square().sum();count[j]+=len(row['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6)
    prior=prior_logits.sigmoid();numerator=torch.zeros_like(prior);mass=torch.zeros_like(prior);visibility=torch.zeros_like(prior)
    reliability=None
    if scene.get('source_reliability_checkpoint'):
        from radio_gs.querying.supervised_scope_reliability import ScopeReliability,scope_features,native_identity_features
        reliability=ScopeReliability(scene['source_reliability_checkpoint'],device)
    budget_old=torch.zeros(len(texts),device=device);budget_detector=torch.zeros_like(budget_old)
    positive=negative=0;negative_mass=torch.zeros_like(budget_old)
    for view,data in zip(spec['views'],records):
        rows=[data['instances'][qi] for qi in ids]
        op,evop,visible=visible_operator(torch.load(view['contributions'],map_location='cpu'),observed,h*w,False)
        probability=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in rows],dim=-1).reshape(h*w,-1).to(device)
        detected=torch.tensor([len(r['scores'])>0 for r in rows],device=device)
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(rows)],device=device)
        detector=torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0. for r in rows],device=device)
        evidence=view_evidence(op,evop,visible,probability,detected,view['frame'],h,w,confidence,boundary_radius=data.get('boundary_radius',1))
        known=(evidence.positive|evidence.negative).float()
        budget_old+=(known*confidence).sum(0);budget_detector+=(known*detector).sum(0)
        weight=known*detector
        if reliability is not None:
            features=scope_features(evidence,probability,detector,confidence,
                torch.tensor([len(r['scores']) for r in rows],device=device,dtype=torch.float32),torch.sparse.mm(op,prior))
            if reliability.state.get('native_region_identity',False):
                identity=torch.load(Path(scene['source_region_identity'])/f"frame_{view['frame']:05d}.pt",map_location='cpu')
                native_ids=[identity['queries'].index(t) for t in texts]
                features=torch.cat([features,native_identity_features(identity,native_ids,device)],1)
            quality=reliability(features)
            if scene.get('source_reliability_mode')=='shared':
                quality=quality.prod(1,keepdim=True).sqrt().expand(-1,2)
            elif scene.get('source_reliability_mode')=='shared_matched_mass':
                positive_count=evidence.positive.sum(0)
                negative_count=evidence.negative.sum(0)
                shared=(positive_count*quality[:,0]+negative_count*quality[:,1])/(positive_count+negative_count).clamp_min(1)
                quality=shared[:,None].expand(-1,2)
            weight=detector*(evidence.positive*quality[:,0]+evidence.negative*quality[:,1])
        numerator+=torch.sparse.mm(evidence.adjoint,evidence.target*weight)
        mass+=torch.sparse.mm(evidence.adjoint,weight)
        visibility+=torch.sparse.mm(evidence.adjoint,evidence.visible.float())
        positive+=int(evidence.positive.sum());negative+=int(evidence.negative.sum())
        negative_mass+=(evidence.negative*weight).sum(0)
    ratio=budget_old/budget_detector.clamp_min(1e-8)
    numerator*=ratio;mass*=ratio
    value=(numerator+.1*visibility*prior)/(mass+.1*visibility).clamp_min(1e-8)
    logits=torch.where(mass>0,torch.logit(value.clamp(1e-6,1-1e-6)),prior_logits)
    return logits,dict(method='visibility_weighted_category_vote',identity_weighting='detector_matched_mass',
        views=len(records),steps=0,passes=1,prior_strength=.1,positive_entries=positive,
        available_negative_entries=negative,admitted_negative_entries=float((negative_mass*ratio).sum()),
        active_elements=int((mass>0).sum()),streamed_view_operators=True,normalization_commuted_after_linear_sum=True,
        source_reliability_checkpoint=scene.get('source_reliability_checkpoint'),
        source_reliability_mode=scene.get('source_reliability_mode'),
        supervised_quality_fit_scenes=reliability.state['fit_scenes'] if reliability is not None else [],
        quality_targets=reliability.state['targets'] if reliability is not None else None)
