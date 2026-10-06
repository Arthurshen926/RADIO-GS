"""Compare 240-step source set fitting with selected-output distillation.

Both arms freeze the rebuilt next66 carrier and semantic scores. The output
arm first fits the retained hard-subset source render for 120 steps, then fits
source pseudo-labels for 120 steps; continuation uses pseudo-labels throughout.
"""
import argparse,json,time,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.querying.typed_evidence import visible_operator


def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def balanced(error,pos,neg):
    return .5*((error*pos).sum()/pos.sum().clamp_min(1)+(error*neg).sum()/neg.sum().clamp_min(1))


def main():
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--carrier-arm',required=True);a.add_argument('--seed-mode',choices=['continuation','output'],required=True);a.add_argument('--transfer-seed',action='store_true');a.add_argument('--scope-negatives',action='store_true');a.add_argument('--arm',choices=['decision'],default='decision');a.add_argument('--geometry-root',default='output/carrier_depth_next52_recovered');a.add_argument('--output-root',default='output/text_video_next63');args=a.parse_args();torch.set_num_threads(4);start=time.monotonic()
    base=Path('output/joint_semantic_next66')/args.carrier_arm/'calibrated'/args.key;data=load(base/'field.pt');root=Path(data['root']);state=load(data['base_field'])['state_dict'];observed=state['observed'].cuda()
    initial=load(base/'typed_readout.pt');old_observed=observed;observed=initial['semantic_observed'].cuda();prior=initial['selection_scores'].cuda();info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    exclusion_reference=load(Path('output/taskwise_scope_composition_next38')/args.key/'anchored/pipeline_calibrated/scores.pt')['scores']['hierarchical_pipeline'].cuda()
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');geometry_prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    geometry_path=Path('output/joint_carrier_next66')/args.carrier_arm/args.key.removeprefix('lerf_')/'geometry.pt'
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
    tracks=[load(Path('output/text_video_next63/tracks')/args.key/f'{qi}.pt') for qi in range(len(initial['texts']))]
    byframe=dict(records)
    seed_confidence=[]
    for qi,t in enumerate(tracks):
        r=byframe[t['seed_frame']]['instances'][qi] if t['seed_frame'] is not None else None
        seed_confidence.append(float((r['scores']*(r['foreground_margin']/scale[qi]).sigmoid()).max()) if r is not None else 0.)
    for frame,d in records:
        hit=load(root/auth['views'][info['frames'].index(frame)]['relative_path']);op_old,_,valid=visible_operator(hit,old_observed,h*w,False)
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale)
        hit=rasterize_single_view_contributions(geo,renderer,camera.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights']
        op,_,_=visible_operator(hit,observed,h*w,False)
        target=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in d['instances']],-1).reshape(h*w,-1).cuda()
        present=torch.tensor([bool(t['frames'].get(frame,{}).get('present',False)) and frame!=t['seed_frame'] for t in tracks],device='cuda')
        temporal=torch.stack([t['frames'][frame]['probability'].float() if bool(present[qi]) else torch.zeros(h,w) for qi,t in enumerate(tracks)],-1).reshape(h*w,-1).cuda()
        target=torch.maximum(target,temporal)
        binary=(target.T.reshape(-1,1,h,w)>.5).float()
        pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
        # A contradictory retained semantic response does not certify a
        # positive label. It makes the detector-complement exclusion unknown.
        # Use the existing .5 retrieval decision, no benchmark calibration.
        native_positive=torch.sparse.mm(op_old,exclusion_reference)>.5
        if args.arm in ('agreement','decision','anneal'):
            neg=neg if args.scope_negatives else neg&~native_positive
            pos=pos&native_positive
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        confidence=torch.maximum(confidence,torch.tensor(seed_confidence,device='cuda')*present)
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    if args.arm=='crossview':
        weighted=[];masses=[]
        for v in views:
            known=(v['pos']|v['neg']).float()*v['confidence'][None]
            adjoint=v['op'].T.coalesce()
            masses.append(torch.sparse.mm(adjoint,known))
            weighted.append(torch.sparse.mm(adjoint,known*v['y']))
        total_mass=sum(masses);total_positive=sum(weighted)
        for v,m,n in zip(views,masses,weighted):
            other_mass=total_mass-m
            other=(total_positive-n)/other_mass.clamp_min(1e-8)
            coverage=torch.sparse.mm(v['op'],(other_mass>1e-8).float())
            response=torch.sparse.mm(v['op'],other)/coverage.clamp_min(1e-8)
            v['pos']=v['pos']&(coverage>.5)&(response>.5)
            v['neg']=v['neg']&(coverage>.5)&(response<.5)
    transfer_stats=None
    if args.transfer_seed:
        retained=load(Path('output/text_video_next63/decision')/args.key/'typed_readout.pt')
        old_geo=FrozenGaussianGeometry.load(retained['selection_geometry']['path'],expected_sha256=retained['selection_geometry']['sha256'],device='cuda')
        old_membership=retained['selection_scores'].cuda();old_support=retained['selection_observed'].cuda();targets=[];den=torch.zeros(len(prior),1,device='cuda');num=torch.zeros_like(prior)
        for v,(frame,_) in zip(views,records):
            camera=geometry_prior.cameras[frame];k=v['k'][0]
            renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale)
            hit=rasterize_single_view_contributions(old_geo,renderer,camera.world_to_camera.cuda(),height=h,width=w,include_depths=False);hit['base_weights']=hit['weights']
            old_op,_,valid=visible_operator(hit,old_support,h*w,False)
            y=torch.sparse.mm(old_op,old_membership);targets.append((v,y,valid));den+=torch.sparse.mm(v['op'].T,valid.float());num+=torch.sparse.mm(v['op'].T,y*valid)
        transferred=num/den.clamp_min(1e-8)
        for _ in range(60):
            gradient=sum(torch.sparse.mm(v['op'].T,(y-torch.sparse.mm(v['op'],transferred))*valid) for v,y,valid in targets)
            transferred=(transferred+.95*gradient/den.clamp_min(1e-8)).clamp(0,1)
        prior=torch.where(den>1e-8,transferred,prior)
        transfer_stats=dict(steps=60,source_views=len(targets),source='next63 soft membership rendered on original carrier; no target views',supported_elements=int((den>1e-8).sum()))
    old_geo=None
    if args.seed_mode=='output':
        retained=load(Path('output/text_video_next63/decision')/args.key/'typed_readout.pt')
        old_geo=FrozenGaussianGeometry.load(retained['selection_geometry']['path'],expected_sha256=retained['selection_geometry']['sha256'],device='cuda')
        old_membership=retained['selection_scores'].cuda();old_support=retained['selection_observed'].cuda()
        with torch.no_grad():
            for v in views:
                full,alpha,_=rasterization(means=old_geo.xyz,quats=old_geo.quaternions,scales=old_geo.scales,opacities=old_geo.opacities.flatten(),colors=old_support.float()[:,None],viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
                v['old_visible']=((full[0,...,0]/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)).flatten()
    result=prior.clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    steps=240
    for qi,text in enumerate(initial['texts']):
        active=[v for v in views if v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not active:continue
        if args.seed_mode=='output':
            selected=(old_membership[:,qi]>.5)&old_support
            with torch.no_grad():
                for v in active:
                    if selected.any():
                        _,alpha,_=rasterization(means=old_geo.xyz[selected],quats=old_geo.quaternions[selected],scales=old_geo.scales[selected],opacities=old_geo.opacities.flatten()[selected],colors=torch.ones(int(selected.sum()),1,device='cuda'),viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
                        v['old_target']=((alpha[0,...,0]*255+.5).floor()>10).float().flatten()
                    else:v['old_target']=torch.zeros(h*w,device='cuda')
        visibility=sum(v['mass'] for v in active);support=(visibility>1e-8)&observed
        parameter=torch.nn.Parameter(torch.logit(prior[:,qi].clamp(.001,.999)));optimizer=torch.optim.Adam([parameter],lr=.05)
        for step in range(steps):
            v=active[step%len(active)];p=torch.where(support,parameter.sigmoid(),prior[:,qi]);response=torch.sparse.mm(v['op'],p[:,None])[:,0]
            member=p*observed
            if args.arm in ('decision','anneal'):
                temperature=.5*(.1/.5)**(step/max(steps-1,1))
                member=torch.where(support,(parameter/temperature).sigmoid(),(prior[:,qi]>.5).float())*observed
            _,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*member,colors=ones,viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
            silhouette=alpha[0,...,0].flatten()
            if args.arm=='decision':
                # Frozen scorer rounds alpha to a byte and accepts values >10.
                cutoff=10.5/255
                silhouette=torch.sigmoid(torch.logit(silhouette.clamp(1e-6,1-1e-6))-torch.logit(torch.tensor(cutoff,device='cuda')))
            known=v['pos'][:,qi]|v['neg'][:,qi];target=v['y'][:,qi]
            distilling=args.seed_mode=='output' and step<120
            if distilling:known=v['old_visible'];target=v['old_target']
            bce=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),target,reduction='none')
            overlap=1-(2*(silhouette*target*known).sum()+1)/((silhouette*known).sum()+(target*known).sum()+1)
            objective=(1. if distilling else v['confidence'][qi])*((bce*known).sum()/known.sum().clamp_min(1)+overlap)
            loss_probability=balanced((response-target).square(),v['pos'][:,qi],v['neg'][:,qi])
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path('output/joint_semantic_next66')/args.carrier_arm/('decision_'+args.seed_mode)/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),semantic_geometry=binding,semantic_observed=observed.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),semantic_geometry=binding,semantic_observed=observed.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(seed_mode=args.seed_mode,output_distillation_steps=120 if args.seed_mode=='output' else 0,scope_complement_negatives=args.scope_negatives,transfer_initialization=transfer_stats,source_evidence='detector union with source-only text-seeded temporal tracking; old native qualification retained',key=args.key,arm=args.arm,steps=steps,lr=.05,prior_weight=0,semantic_prior_role='new-carrier semantic-score initialization',probability_readout='rebuilt next66 calibrated native field, separate from set decoding',seconds=time.monotonic()-start,geometry_frozen=True,selection_carrier_multiplier=1,new_semantic_geometry_frozen_during_selection=True,selection_visibility_recomputed=True,subset_temperature='0.5 to 0.1' if args.arm in ('decision','anneal') else 1.,deployment_cutoff=10.5/255 if args.arm=='decision' else None,target_rgb_used=False,source_raster=[h,w],history=history),indent=2)+'\n')

if __name__=='__main__':main()
