"""Categorical source evidence with identities retained until projection.

Geometry-only region links transport semantic evidence, not masks or extent.
Uncovered pixels are unknown. The matched emitter/native/union controls share
the decoder, geometry, views and dual-native prior.
"""
import json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.native_query_encoders import encode_category_text
from radio_gs.querying.native_observation_prior import compiled_observation_prior
from radio_gs.querying.typed_evidence import visible_operator


@torch.inference_mode()
def fuse_region_observations(prior_logits,observed,scene,texts):
    started=time.monotonic();device=prior_logits.device;root=Path(scene['source_region_observations']);mode=scene['source_region_mode']
    auxiliary=json.loads(Path(scene['local_auxiliary_scene']).read_text())
    auxiliary['native_observation_reliability']=scene['native_observation_reliability'];auxiliary.pop('observation_weighting',None)
    query=encode_category_text(auxiliary,texts,device=device)
    native,_=compiled_observation_prior(auxiliary,query,observed,prior_logits);prior=(native.sigmoid()+prior_logits.sigmoid())*.5
    views=sorted(scene['source_evidence']['views'],key=lambda v:v['frame']);records=[torch.load(root/f"frame_{v['frame']:05d}.pt",map_location='cpu') for v in views]
    kind='union' if mode=='verified_union' else 'instance';probability=[];emitting=[];confidence=[];offsets=[0]
    for record in records:
        row=record[kind];ids=[record['queries'].index(t) for t in texts]
        descriptor=F.normalize(row['descriptors'].float().to(device),dim=-1)
        probability.append((100*descriptor@query.T).softmax(1));emitting.append(row['emitting'][:,ids].to(device))
        confidence.append(row['confidence'].to(device));offsets.append(offsets[-1]+len(descriptor))
    probability=torch.cat(probability);emitting=torch.cat(emitting);confidence=torch.cat(confidence)
    edge_count=0
    if mode=='linked_verified':
        graph=torch.load(root/'graph.pt',map_location=device);edges=graph['edges'];weight=graph['weights'];edge_count=len(edges)
        numerator=probability.clone();mass=torch.ones(len(probability),1,device=device)
        for a,b in [(edges[:,0],edges[:,1]),(edges[:,1],edges[:,0])]:
            numerator.index_add_(0,a,weight[:,None]*probability[b]);mass.index_add_(0,a,weight[:,None])
        probability=numerator/mass
    if mode=='emitter_instance':identity=emitting
    elif mode=='native_instance':identity=probability
    else:identity=emitting*probability
    numerator=torch.zeros_like(prior_logits);mass=torch.zeros_like(prior[:,:1]);visibility=torch.zeros_like(mass);pixels_known=0
    for i,(view,record) in enumerate(zip(views,records)):
        masks=record[kind]['masks'].to(device);h,w=masks.shape[-2:];lo,hi=offsets[i:i+2]
        op,_,visible=visible_operator(torch.load(view['contributions'],map_location='cpu'),observed,h*w,False)
        visibility+=torch.sparse.mm(op.T,visible.float())
        if not len(masks):continue
        radius=record['boundary_radius'];core=(-F.max_pool2d(-masks[:,None].float(),2*radius+1,1,radius)>.5)[:,0].flatten(1).T.float()
        score=identity[lo:hi]*confidence[lo:hi,None];strength=core@score
        distribution=strength/strength.sum(1,keepdim=True).clamp_min(1e-12)
        reliability=(core*score.sum(1)[None]).amax(1,keepdim=True)*visible
        numerator+=torch.sparse.mm(op.T,distribution*reliability);mass+=torch.sparse.mm(op.T,reliability)
        pixels_known+=int((reliability>0).sum())
    value=(numerator+.1*visibility*prior)/(mass+.1*visibility).clamp_min(1e-8)
    logits=torch.where(mass>0,value.clamp(1e-6,1-1e-6).logit(),prior_logits)
    return logits,dict(method='region_observation_vote',mode=mode,views=len(views),nodes=len(probability),edges=edge_count,
        complete_menu_before_projection=True,uncovered_pixels='unknown',known_pixels=pixels_known,
        instance_observations_separate=kind=='instance',geometry_only_correspondences=mode=='linked_verified',
        mask_extent_transported=False,transitive_merging=False,ground_truth_training=False,
        identity_scores_are_calibrated=False,prior_strength=.1,seconds=time.monotonic()-started)
