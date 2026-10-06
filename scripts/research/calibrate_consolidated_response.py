"""Current-query scalar calibration from source masks with held-out source views.

Changes semantic probabilities only. Source pseudo-label risk is not true
object accuracy; detector complements can still miss parts of the target.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',required=True)
    p.add_argument('--readout',required=True)
    p.add_argument('--evidence',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
    scene=json.loads(Path(a.scene).read_text());data=load(a.readout)
    binding=data['semantic_geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
    cameras=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    observed=data['semantic_observed'].cuda();prior=data['scores'].cuda();views=[];h,w=96,128
    with torch.no_grad():
        for i,frame in enumerate(scene['source_frames']):
            evidence=load(Path(a.evidence)/f'identity_{frame}.pt')
            if evidence['queries']!=data['texts']:raise ValueError('Evidence queries differ from current response')
            cam=cameras.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
            renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*cameras.scale,far_plane=100*cameras.scale)
            hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights']
            op,_,valid=visible_operator(hit,observed,h*w,False)
            target=torch.stack([F.interpolate(r['masks'].float().amax(0)[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0] if len(r['scores']) else torch.zeros(h,w) for r in evidence['instances']],-1).reshape(h*w,-1).cuda()
            binary=(target.T.reshape(-1,1,h,w)>.5).float()
            positive=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid
            negative=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
            confidence=torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0. for r in evidence['instances']],device='cuda')
            views.append(dict(op=op,y=target,pos=positive,neg=negative,confidence=confidence,held=i%5==0,frame=frame))
    scores=prior.clone();rows=[]
    for qi,text in enumerate(data['texts']):
        train=[v for v in views if not v['held'] and v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        held=[v for v in views if v['held'] and v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not train or not held:
            rows.append(dict(query=text,accepted=False,reason='missing_train_or_heldout_source_positive'));continue
        logits=torch.logit(prior[:,qi].clamp(1e-5,1-1e-5)).detach()
        delta=torch.nn.Parameter(torch.zeros((),device='cuda'));opt=torch.optim.Adam([delta],lr=.05)
        def loss(v,bias):
            value=(logits+bias).sigmoid()*observed
            rendered=torch.sparse.mm(v['op'],value[:,None])[:,0]
            error=F.binary_cross_entropy(rendered.clamp(1e-5,1-1e-5),v['y'][:,qi],reduction='none')
            pos,neg=v['pos'][:,qi],v['neg'][:,qi]
            return .5*((error*pos).sum()/pos.sum().clamp_min(1)+(error*neg).sum()/neg.sum().clamp_min(1))
        for step in range(max(240,len(train))):
            v=train[step%len(train)];objective=loss(v,delta)*v['confidence'][qi]
            opt.zero_grad();objective.backward();opt.step()
        with torch.no_grad():
            weights=torch.stack([v['confidence'][qi] for v in held]);before=torch.stack([loss(v,0.) for v in held]);after=torch.stack([loss(v,delta) for v in held])
            baseline=float((weights*before).sum()/weights.sum());candidate=float((weights*after).sum()/weights.sum());accepted=candidate<baseline
            if accepted:scores[:,qi]=(logits+delta).sigmoid()*observed
        rows.append(dict(query=text,accepted=accepted,delta=float(delta),train_views=len(train),held_views=len(held),held_source_bce_before=baseline,held_source_bce_after=candidate))
        print(scene['scene'],text,accepted,round(float(delta),3),flush=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(data,scores=scores.cpu(),calibration_parent=a.readout),out/'typed_readout.pt')
    (out/'calibration.json').write_text(json.dumps(dict(scene=a.scene,readout=a.readout,evidence=a.evidence,
        policy='one scalar logit offset per runtime text;240steps;every fifth source view held out;accept only lower held-source balanced BCE',
        query_rows=rows,seconds=time.monotonic()-start,selection_unchanged=True,
        target_rgb_or_gt_used=False,source_complements_are_fallible=True),indent=2)+'\n')


if __name__=='__main__':main()
