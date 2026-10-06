"""Lift partial categorical observations without treating uncovered RGB as negative.

Only for a complete, mutually exclusive vocabulary. Overlapping proposals keep
soft category competition; missing or boundary evidence does not assert a class.
"""
import torch
from torch.nn import functional as F
from radio_gs.querying.typed_evidence import visible_operator


@torch.inference_mode()
def fuse_categorical_observations(prior_logits, observed, scene, texts):
    spec=scene['source_evidence']; h,w=spec['raster_hw']; device=prior_logits.device
    records=[torch.load(v['identity'],map_location='cpu',weights_only=False) for v in spec['views']]
    ids=[records[0]['queries'].index(t) for t in texts]
    square=torch.zeros(len(texts));count=torch.zeros_like(square)
    for data in records:
        for j,qi in enumerate(ids):
            row=data['instances'][qi]
            square[j]+=row['foreground_margin'].square().sum();count[j]+=len(row['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6)
    prior=prior_logits.sigmoid()
    numerator=torch.zeros_like(prior)
    mass=torch.zeros(len(observed),1,device=device);visibility=torch.zeros_like(mass)
    covered_pixels=ambiguous_pixels=0
    for view,data in zip(spec['views'],records):
        strengths=[]
        for j,qi in enumerate(ids):
            row=data['instances'][qi]
            if not len(row['scores']):
                strengths.append(torch.zeros(h,w));continue
            probability=row['masks'].float()
            core=(-F.max_pool2d(-(probability>.5).float()[:,None],3,1,1)[:,0]>.5)
            identity=row['scores']*(row['foreground_margin']/scale[j]).sigmoid()
            strengths.append((probability*core*identity[:,None,None]).amax(0))
        strength=torch.stack(strengths,dim=-1).reshape(h*w,-1).to(device)
        target=strength/strength.sum(-1,keepdim=True).clamp_min(1e-8)
        # Repetition and overlapping class proposals cannot increase confidence
        # beyond the strongest individual observation.
        reliability=strength.amax(-1,keepdim=True)
        hit=torch.load(view['contributions'],map_location='cpu',weights_only=False)
        op,_,visible=visible_operator(hit,observed,h*w,False)
        weight=reliability*visible
        adjoint=op.T.coalesce()
        numerator+=torch.sparse.mm(adjoint,target*weight)
        mass+=torch.sparse.mm(adjoint,weight)
        visibility+=torch.sparse.mm(adjoint,visible.float())
        covered_pixels+=int((weight>0).sum())
        ambiguous_pixels+=int(((strength>0).sum(-1)>1).sum())
    value=(numerator+.1*visibility*prior)/(mass+.1*visibility).clamp_min(1e-8)
    logits=torch.where(mass>0,torch.logit(value.clamp(1e-6,1-1e-6)),prior_logits)
    return logits,dict(method='partial_categorical_observation_vote',views=len(records),passes=1,
        prior_strength=.1,covered_pixels=covered_pixels,ambiguous_pixels=ambiguous_pixels,
        active_elements=int((mass>0).sum()),uncovered_source_pixels='unknown',
        category_competition='complete vocabulary before geometric lift')
