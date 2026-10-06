"""Preserve local identity confidence and optional partial-extent uncertainty.

All arms share the next56 decision decoder and its next54 membership start.
Local identity weights do not merge detections into a verified physical identity;
source containment only censors exclusions and does not assert new positives.
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
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--arm',choices=['local_identity','instance_balance','extent_unknown','combined','clip_identity'],default='local_identity');a.add_argument('--geometry-root',default='output/carrier_depth_next52_recovered');a.add_argument('--evidence-base',default='output/typed_evidence_next44');a.add_argument('--output-root',default='output/instance_evidence_next58');args=a.parse_args();torch.set_num_threads(4);start=time.monotonic()
    base=Path('output/native_rebuild_next54/mixture_masscal')/args.key;data=load(base/'field.pt');root=Path(data['root']);state=load(data['base_field'])['state_dict'];observed=state['observed'].cuda()
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
        d=load(Path(args.evidence_base)/args.key/f'identity_{frame}.pt');records.append((frame,d))
        for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[]
    relations=[]
    if args.arm in ('extent_unknown','combined'):
        relations=json.loads((Path('output/decision_sets_next49/proposal_unknown')/args.key/'fit.json').read_text())['proposal_relations']
        episode=load(root/'episodes.pt');region_masks=episode['masks'];del episode
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
        native_positive=torch.sparse.mm(op_old,exclusion_reference)>.5
        neg=neg&~native_positive
        pos=pos&native_positive
        if args.arm in ('extent_unknown','combined'):
            for relation in relations:
                if relation['frame']==frame:
                    qi=initial['texts'].index(relation['query'])
                    neg[:,qi]&=~(region_masks[relation['region']].flatten().cuda()>.5)
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        positive_weight=torch.ones_like(target)
        for j,r in enumerate(d['instances']):
            if not len(r['scores']):continue
            masks=r['masks'].float().cuda();ci=(r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).cuda()
            # Each detected region keeps its own identity confidence; a good
            # detection cannot lend its confidence to an unrelated union member.
            union=masks.amax(0)
            local=(masks*ci[:,None,None]).amax(0)/union.clamp_min(1e-8)
            local=local/confidence[j].clamp_min(1e-8)
            if args.arm=='instance_balance':
                area=(masks>.5).sum((1,2)).clamp_min(1)
                inverse=((masks>.5).float()/area[:,None,None]).amax(0)
                occupied=union>.5
                inverse=inverse/inverse[occupied].mean().clamp_min(1e-8) if occupied.any() else inverse
                local=local*inverse
            if args.arm!='extent_unknown':positive_weight[:,j]=local.flatten()
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(positive_weight=positive_weight,op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    result=prior.clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    steps=120
    for qi,text in enumerate(initial['texts']):
        active=[v for v in views if v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not active:continue
        visibility=sum(v['mass'] for v in active);support=(visibility>1e-8)&observed
        parameter=torch.nn.Parameter(torch.logit(prior[:,qi].clamp(.001,.999)));optimizer=torch.optim.Adam([parameter],lr=.05)
        for step in range(steps):
            v=active[step%len(active)];p=torch.where(support,parameter.sigmoid(),prior[:,qi]);response=torch.sparse.mm(v['op'],p[:,None])[:,0]
            member=p*observed
            temperature=.5*(.1/.5)**(step/max(steps-1,1))
            member=torch.where(support,(parameter/temperature).sigmoid(),(prior[:,qi]>.5).float())*observed
            _,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*member,colors=ones,viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
            silhouette=alpha[0,...,0].flatten()
            # Frozen scorer rounds alpha to a byte and accepts values >10.
            cutoff=10.5/255
            silhouette=torch.sigmoid(torch.logit(silhouette.clamp(1e-6,1-1e-6))-torch.logit(torch.tensor(cutoff,device='cuda')))
            known=v['pos'][:,qi]|v['neg'][:,qi];target=v['y'][:,qi]
            weighted_known=known.float()*torch.where(v['pos'][:,qi],v['positive_weight'][:,qi],torch.ones_like(target))
            bce=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),target,reduction='none')
            overlap=1-(2*(silhouette*target*weighted_known).sum()+1)/((silhouette*weighted_known).sum()+(target*weighted_known).sum()+1)
            objective=v['confidence'][qi]*((bce*weighted_known).sum()/known.sum().clamp_min(1)+overlap)
            loss_probability=balanced((response-target).square(),v['pos'][:,qi],v['neg'][:,qi])
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path(args.output_root)/args.arm/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(key=args.key,arm=args.arm,steps=steps,lr=.05,prior_weight=0,semantic_prior_role='retained independent selection initialization',probability_readout='retained next54, separate from set decoding',seconds=time.monotonic()-start,geometry_frozen=True,selection_carrier_multiplier=1,original_semantic_geometry_unchanged=True,selection_visibility_recomputed=True,subset_temperature='0.5 to 0.1',deployment_cutoff=10.5/255,target_rgb_used=False,evidence_base=args.evidence_base,source_raster=[h,w],proposal_relations=relations,history=history),indent=2)+'\n')

if __name__=='__main__':main()
