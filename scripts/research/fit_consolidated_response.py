"""Reconstruct a typed text response from admitted source evidence.

Uses the same rendered-evidence solver as reference queries. Native probabilities
initialize unsupported elements; source instances stay identity-filtered, rejected
masks remain unknown. Membership head is an independent unchanged output.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.querying.typed_evidence import visible_operator,solve_rendered_evidence
from scripts.research.decode_expanded_extent_next72 import sample_operator_rows
from radio_gs.querying.finite_evidence import combine_with_native,source_strength

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--finite-evidence',action='store_true');p.add_argument('--evidence-strength',choices=['confidence','positive_logit'],default='confidence');p.add_argument('--inputs',required=True);p.add_argument('--readout',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=120);p.add_argument('--solver',choices=['rendered','backprojection'],default='rendered');p.add_argument('--policies',nargs='+',choices=['native_protected','source_complement'],default=['native_protected','source_complement']);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
    inputs=json.loads(Path(a.inputs).read_text());scene=json.loads(Path(inputs['scene']).read_text());data=torch.load(a.readout,map_location='cpu',weights_only=False);b=data['semantic_geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);observed=data['semantic_observed'].cuda();prior=data['scores'].cuda();h,w=192,256;views=[];active=torch.zeros(prior.shape[1],dtype=torch.long);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for i,frame in enumerate(scene['source_frames']):
        evidence=torch.load(Path(inputs['detections'])/f'identity_{frame}.pt',map_location='cpu',weights_only=False)
        if evidence['queries']!=data['texts']:raise ValueError('Evidence and explicit query order differ')
        cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights'];op,_,valid=visible_operator(hit,observed,h*w,False)
        target=torch.stack([F.interpolate(r['masks'].float().amax(0)[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0] if len(r['scores']) else torch.zeros(h,w) for r in evidence['instances']],-1).reshape(h*w,-1).cuda();binary=(target.T.reshape(-1,1,h,w)>.5).float();pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
        for qi,r in enumerate(evidence['instances']):
            rejected=r.get('unknown_masks')
            if rejected is not None and len(rejected):neg[:,qi]&=~(F.interpolate(rejected.float().amax(0)[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False)[0,0].flatten()>.5)
        confidence=torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0. for r in evidence['instances']],device='cuda');active+=(confidence>0).cpu()
        phase=i%4;pixels=torch.arange(h*w,device='cuda').reshape(h,w)[phase//2::2,phase%2::2].flatten();op=sample_operator_rows(op,pixels);pos=pos[pixels];neg=neg[pixels];target=target[pixels];native_positive=torch.sparse.mm(op,prior)>.5
        views.append((op,target,pos,neg,native_positive,confidence))
    strength=source_strength([v[5]*(v[2].any(0)) for v in views],a.evidence_strength)
    results={}
    for policy in a.policies:
        observations=[]
        for op,target,pos,neg,native_positive,confidence in views:
            admitted=neg&~native_positive if policy=='native_protected' else neg
            known=(pos|admitted).float()*confidence[None];observations.append((op,target,known,1.))
        if a.solver=='rendered':values,mass=solve_rendered_evidence(prior,observations,steps=a.steps)
        else:
            mass=torch.zeros_like(prior);positive=torch.zeros_like(prior)
            for op,target,known,weight in observations:
                mass+=torch.sparse.mm(op.T,known);positive+=torch.sparse.mm(op.T,known*target)
            values=torch.where(mass>1e-8,positive/mass.clamp_min(1e-8),prior)
        if a.finite_evidence:values=combine_with_native(values,prior,strength[None])
        values[~observed]=0
        dest=out/policy;dest.mkdir(exist_ok=True);torch.save(dict(data,scores=values.cpu(),response_evidence=a.inputs,response_policy=policy),dest/'typed_readout.pt')
        results[policy]=dict(updated_entries=int((mass>1e-8).sum()),maximum_change=float((values-prior).abs().max()),unsupported_prior_retained=True)
        print(scene['scene'],policy,'fit',flush=True)
        del observations,values,mass
    (out/'fit.json').write_text(json.dumps(dict(evidence_strength=a.evidence_strength,finite_evidence=a.finite_evidence,source_strength=strength.cpu().tolist(),native_reference_strength=1 if a.finite_evidence else 0,inputs=a.inputs,readout=a.readout,source_frames=scene['source_frames'],active_views_per_query=active.tolist(),query_texts=data['texts'],steps=a.steps if a.solver=='rendered' else 0,solver=a.solver,raster=[h,w],sample='one of four interleaved pixel phases per view; same phase as subset solver',results=results,selection_unchanged=True,new_vfm_calls=0,target_rgb_used=False,target_gt_used=False,seconds=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
