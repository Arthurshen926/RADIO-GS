"""Local, class-shared residual fusion of frozen native and scope evidence.

Unlike view/category quality scalars, each visible element receives its own
decision. Query encoders and map coordinates remain frozen. This supervised
category-partition reader does not apply to overlapping parts or instances.
"""
import json,math,time
from pathlib import Path
import torch
from radio_gs.querying.typed_evidence import visible_operator,view_evidence
from radio_gs.querying.instance_reliability import instance_reliability

FEATURE_NAMES=['native_probability','native_log_odds','native_competitor','native_entropy',
    'source_probability','source_log_odds','source_competitor','source_entropy',
    'base_probability','base_competitor','positive_mass_fraction','positive_target_mean',
    'negative_target_mean','known_confidence_per_visibility','log_visibility','source_variance']
AUXILIARY_NAMES=['auxiliary_probability','auxiliary_log_odds','auxiliary_competitor','auxiliary_entropy']

def competition(p):
    top=p.topk(2,dim=1);own=torch.arange(p.shape[1],device=p.device)[None]
    competitor=torch.where(own==top.indices[:,:1],top.values[:,1:2],top.values[:,:1])
    q=p/p.sum(1,keepdim=True).clamp_min(1e-8)
    entropy=-(q*q.clamp_min(1e-8).log()).sum(1,keepdim=True)/math.log(p.shape[1])
    return competitor,entropy.expand_as(p)

@torch.no_grad()
def compile_local_evidence(prior_logits,observed,scene,texts,auxiliary_logits=None, *, capture_observations=False):
    captured=[] if capture_observations else None
    device=prior_logits.device;spec=scene['source_evidence'];h,w=spec['raster_hw']
    records=[torch.load(v['identity'],map_location='cpu') for v in spec['views']]
    ids=[records[0]['queries'].index(t) for t in texts];c=len(ids)
    square=torch.zeros(c);count=torch.zeros(c)
    for record in records:
        for j,qi in enumerate(ids):
            row=record['instances'][qi];square[j]+=row['foreground_margin'].square().sum();count[j]+=len(row['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6)
    pn,pd,nn,nd,second,visibility=[torch.zeros_like(prior_logits) for _ in range(6)]
    old=torch.zeros(c,device=device);det=torch.zeros_like(old)
    reliability_mode=scene.get('instance_evidence_reliability')
    local_reliability=reliability_mode in ('local','positive_local')
    for view,record in zip(spec['views'],records):
        rows=[record['instances'][qi] for qi in ids]
        op,evop,visible=visible_operator(torch.load(view['contributions'],map_location='cpu'),observed,h*w,False)
        probability=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in rows],-1).reshape(h*w,c).to(device)
        detector=torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0 for r in rows],device=device)
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0 for j,r in enumerate(rows)],device=device)
        e=view_evidence(op,evop,visible,probability,detector>0,view['frame'],h,w,confidence,boundary_radius=record.get('boundary_radius',1))
        positive=e.positive.float()*detector;negative=e.negative.float()*detector;known=(e.positive|e.negative).float()
        if local_reliability:
            reliability=[instance_reliability(r,(h,w),device,
                r['scores'] if reliability_mode=='positive_local' else r['scores']*(r['foreground_margin']/scale[j]).sigmoid(),
                complement='view' if reliability_mode=='positive_local' else 'minimum') for j,r in enumerate(rows)]
            positive=e.positive*torch.stack([r[0] for r in reliability],-1).reshape(h*w,c)
            negative=e.negative*torch.stack([r[1] for r in reliability])[None]
            for j,r in enumerate(rows):
                unknown=r.get('unknown_masks')
                if unknown is not None and len(unknown):
                    from torch.nn import functional as F
                    excluded=F.interpolate(unknown.float().amax(0)[None,None].to(device),size=(h,w),mode='bilinear',align_corners=False)[0,0].flatten()>.5
                    negative[:,j]*=~excluded
        if captured is not None:captured.append((op,probability,positive+negative))
        pn+=torch.sparse.mm(e.adjoint,probability*positive);pd+=torch.sparse.mm(e.adjoint,positive)
        nn+=torch.sparse.mm(e.adjoint,probability*negative);nd+=torch.sparse.mm(e.adjoint,negative)
        second+=torch.sparse.mm(e.adjoint,probability.square()*(positive+negative))
        visibility+=torch.sparse.mm(e.adjoint,e.visible.float());old+=(known*confidence).sum(0)
        det+=(positive+negative).sum(0) if local_reliability else (known*detector).sum(0)
    ratio=(old/det.clamp_min(1e-8) if not local_reliability or scene.get('instance_evidence_mass')=='matched'
           else torch.ones_like(old))
    pn*=ratio;pd*=ratio;nn*=ratio;nd*=ratio;second*=ratio
    mass=pd+nd;num=pn+nn;prior=prior_logits.sigmoid();strength=.1*visibility
    value=(num+strength*prior)/(mass+strength).clamp_min(1e-8)
    base=torch.where(mass>0,value.clamp(1e-6,1-1e-6).logit(),prior_logits)
    source=num/mass.clamp_min(1e-8);source=torch.where(mass>0,source,prior)
    pc,pe=competition(prior);sc,se=competition(source);bc,_=competition(base.sigmoid())
    features=[prior,prior_logits/10,pc,pe,source,source.clamp(1e-6,1-1e-6).logit()/10,sc,se,
        base.sigmoid(),bc,pd/mass.clamp_min(1e-8),pn/pd.clamp_min(1e-8),nn/nd.clamp_min(1e-8),
        mass/visibility.clamp_min(1e-8),visibility.log1p()/10,(second/mass.clamp_min(1e-8)-source.square()).clamp_min(0)]
    average=base
    if auxiliary_logits is not None:
        auxiliary=auxiliary_logits.sigmoid();ac,ae=competition(auxiliary)
        features.extend([auxiliary,auxiliary_logits/10,ac,ae])
        p=(num+strength*(prior+auxiliary)*.5)/(mass+strength).clamp_min(1e-8)
        average=torch.where(mass>0,p.clamp(1e-6,1-1e-6).logit(),prior_logits)
    result=dict(features=torch.stack(features,-1),base_logits=base,average_logits=average,
        active=(mass>0)&observed[:,None],source_views=len(records))
    if captured is not None:
        # Opt-in experiment view of the SAME production admission and weights.
        result['reconstruction']=dict(observations=[(op,y,k,ratio[None]) for op,y,k in captured],
            prior=prior if auxiliary_logits is None else (prior+auxiliary_logits.sigmoid())*.5,
            strength=strength,initial=prior,mass=mass)
    return result

def make_local_head(dimension):
    return torch.nn.Sequential(torch.nn.Linear(dimension,32),torch.nn.SiLU(),torch.nn.Linear(32,1))

def local_logits(network,x,base,active,mode='local'):
    if mode=='global':
        summary=(x*active[:,:,None]).sum(0)/active.sum(0).clamp_min(1)[:,None]
        delta=network(summary).squeeze(-1)[None]
    else:delta=network(x).squeeze(-1)
    return base+delta*active

def point_retrieval_loss(logits,counts,observed):
    p=logits.softmax(1)*observed[:,None];size=counts.sum(0);present=size>0
    ce=-(counts*p.clamp_min(1e-8).log()).sum(0)/size.clamp_min(1)
    intersection=(counts*p).sum(0);union=(counts.sum(1,keepdim=True)*p).sum(0)+size-intersection
    return .5*ce[present].mean()+.5*(1-intersection[present]/union[present].clamp_min(1)).mean()

@torch.inference_mode()
def fuse_local_elements(prior_logits,observed,scene,texts):
    started=time.monotonic();device=prior_logits.device
    state=torch.load(scene['local_element_checkpoint'],map_location=device) if scene.get('local_element_checkpoint') else None
    auxiliary=None
    if state is None or state['dual_teacher']:
        from radio_gs.models.native_query_encoders import encode_category_text
        from radio_gs.querying.native_observation_prior import compiled_observation_prior
        aux=json.loads(Path(scene['local_auxiliary_scene']).read_text())
        # Both native spaces use the declared readout policy. Raw adapter map
        # configs predate the entropy-weighted runtime and are not fit recipes.
        aux['native_observation_reliability']=scene['native_observation_reliability']
        aux.pop('observation_weighting',None)
        query=encode_category_text(aux,texts,device=device)
        auxiliary,_=compiled_observation_prior(aux,query,observed,prior_logits)
    data=compile_local_evidence(prior_logits,observed,scene,texts,auxiliary)
    if state is None:result=data['average_logits']
    else:
        model=make_local_head(len(state['mean'])).to(device).eval();model.load_state_dict(state['network'])
        x=(data['features']-state['mean'])/state['std']
        result=local_logits(model,x,data['base_logits'],data['active'],state['mode'])
    return result,dict(method='local_element_evidence_residual' if state else 'equal_native_prior_control',
        views=data['source_views'],complete_menu_before_local_features=True,native_query_coordinates_frozen=True,
        active_element_class_pairs=int(data['active'].sum()),source_evidence_required=True,
        fit_scenes=state['fit_scenes'] if state else [],ground_truth_training=state is not None,
        local_element_checkpoint=scene.get('local_element_checkpoint'),local_auxiliary_scene=scene.get('local_auxiliary_scene'),
        granularity=state['mode'] if state else 'uniform',seconds=time.monotonic()-started)
