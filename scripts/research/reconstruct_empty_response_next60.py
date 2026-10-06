"""Reconstruct a separate semantic response for eligible empty-query bootstraps.

Nonempty observed memberships do not trigger this path. Source evidence is pseudo-
supervision, not verified identity. No benchmark query-name exceptions are used.
"""
import argparse,json,time,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator,ViewEvidence,solve_set,SetType


def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def balanced(error,pos,neg):
    return .5*((error*pos).sum()/pos.sum().clamp_min(1)+(error*neg).sum()/neg.sum().clamp_min(1))


def main():
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--arm',choices=['response'],default='response');a.add_argument('--geometry-root',default='output/carrier_depth_next52_recovered');a.add_argument('--output-root',default='output/empty_bootstrap_next60');args=a.parse_args();torch.set_num_threads(4);start=time.monotonic()
    base=Path('output/factorized_scope_next59/hierarchy')/args.key;data=load(base/'field.pt');root=Path(data['root']);state=load(data['base_field'])['state_dict'];observed=state['observed'].cuda()
    initial=load(base/'typed_readout.pt');prior=initial['selection_scores'].cuda();info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    exclusion_reference=load(Path('output/taskwise_scope_composition_next38')/args.key/'anchored/pipeline_calibrated/scores.pt')['scores']['hierarchical_pipeline'].cuda()
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');geometry_prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    old_observed=observed
    geometry_path=Path(args.geometry_root)/args.key.removeprefix('lerf_')/'geometry.pt'
    from radio_gs.utils.immutable_artifacts import sha256_file
    geo=FrozenGaussianGeometry.load(geometry_path,expected_sha256=sha256_file(geometry_path),device='cuda')
    from radio_gs.utils.immutable_artifacts import sha256_file
    binding=dict(path=str(geometry_path),sha256=sha256_file(geometry_path))
    from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
    from types import SimpleNamespace
    records=[];square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
    for frame in info['train_frames']:
        d=load(Path('output/typed_evidence_next44')/args.key/f'identity_{frame}.pt');records.append((frame,d))
        for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[]
    for frame,d in records:
        hit=load(root/auth['views'][info['frames'].index(frame)]['relative_path']);op_old,_,valid=visible_operator(hit,old_observed,h*w,False)
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale)
        hit=rasterize_single_view_contributions(geo,renderer,camera.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights']
        op,_,_=visible_operator(hit,observed,h*w,False)
        target=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in d['instances']],-1).reshape(h*w,-1).cuda();binary=(target.T.reshape(-1,1,h,w)>.5).float()
        pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
        # A contradictory retained semantic response does not certify a
        # positive label. It makes the detector-complement exclusion unknown.
        # Use the existing .5 retrieval decision, no benchmark calibration.
        raw_positive=pos.clone();raw_negative=neg.clone()
        native_positive=torch.sparse.mm(op_old,exclusion_reference)>.5
        neg=neg&~native_positive
        pos=pos&native_positive
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(raw_positive=raw_positive,raw_negative=raw_negative,identity_positive=torch.tensor([bool((r['foreground_margin']>0).any()) for r in d['instances']],device='cuda'),op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    result=load(Path('output/empty_bootstrap_next60/bootstrap')/args.key/'typed_readout.pt')['selection_scores'].cuda();response_values=initial['scores'].clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    steps=60
    for qi,text in enumerate(initial['texts']):
        if ((prior[:,qi]>.5)&observed).any():continue
        active=[v for v in views if v['confidence'][qi]>0 and v['identity_positive'][qi] and v['raw_positive'][:,qi].any()]
        for v in active:
            v['pos'][:,qi]=v['raw_positive'][:,qi]
            v['neg'][:,qi]=v['raw_negative'][:,qi]
        if not active:continue
        visibility=sum(v['mass'] for v in active);support=(visibility>1e-8)&observed
        numerator=torch.zeros(len(prior),device='cuda');denominator=numerator.clone()
        for v in active:
            adjoint=v['op'].T.coalesce();confidence=v['confidence'][qi]
            numerator+=confidence*torch.sparse.mm(adjoint,v['y'][:,qi,None])[:,0]
            denominator+=confidence*v['mass']
        seed=(numerator/denominator.clamp_min(1e-8)).clamp(.01,.99)
        if initial['scores'][:,qi].max()>.5:continue
        examples=[]
        for vi,v in enumerate(active):
            valid=(torch.sparse.sum(v['op'],dim=1).to_dense()>0)[:,None]
            examples.append(ViewEvidence(v['op'],v['op'].T.coalesce(),v['op'],v['y'][:,qi,None],v['pos'][:,qi,None],v['neg'][:,qi,None],valid,vi,v['confidence'][qi].reshape(1,1)))
        value,stats=solve_set(seed[:,None],examples,SetType.TEXT_UNION,corroborate=False,consistency=True,steps=60)
        response_values[:,qi]=torch.where(support,value[:,0],initial['scores'][:,qi].cuda()).cpu()
        history.append(dict(query=text,views=len(active),response_fit=stats))
        print(args.key,text,'response reconstructed',flush=True)
    out=Path(args.output_root)/args.arm/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=response_values,selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=response_values,selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(key=args.key,arm=args.arm,steps=steps,lr=None,prior_weight=.1,inherited_selection_steps=120,semantic_prior_role='source projection seeds only globally empty memberships; nonempty queries unchanged',probability_readout='separate source-only linear inverse rendering for eligible empty queries',seconds=time.monotonic()-start,geometry_frozen=True,selection_carrier_multiplier=1,original_semantic_geometry_unchanged=True,selection_visibility_recomputed=True,subset_temperature='0.5 to 0.1',deployment_cutoff=10.5/255,target_rgb_used=False,source_raster=[h,w],history=history),indent=2)+'\n')

if __name__=='__main__':main()
