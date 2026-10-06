"""Source-native candidate extent selection in a common image window.

Frozen SigLIP contrasts foreground and complement before geometric fusion.
No target images or query-list competition; geometry and decoder stay fixed.
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
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--arm',choices=['source'],default='source');a.add_argument('--geometry-root',default='output/carrier_depth_next52_recovered');a.add_argument('--output-root',default='output/native_extent_next61');args=a.parse_args();torch.set_num_threads(4);start=time.monotonic()
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
    evidence_out=Path(args.output_root)/'source_evidence'/args.key
    records=[];square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
    for frame in info['train_frames']:
        d=load(Path('output/typed_evidence_next44')/args.key/f'identity_{frame}.pt');records.append((frame,d))
        for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[]
    relations=json.loads((Path('output/decision_sets_next49/proposal_unknown')/args.key/'fit.json').read_text())['proposal_relations']
    ep=load(root/'episodes.pt');region_masks=ep['masks'];region_views=ep['proposal_views'];del ep
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
        frame_ids=torch.where(region_views==info['frames'].index(frame))[0]
        frame_masks=region_masks[frame_ids].flatten(1).float().cuda()
        hypotheses=[];native_choices=[];native_margins=[]
        cached=load(evidence_out/f'{frame}.pt')
        for qi,record in enumerate(d['instances']):
            groups=[];choices=[];margins=[]
            related=[r for r in relations if r['frame']==frame and r['query']==initial['texts'][qi]]
            for mask in record['masks']:
                original=mask.float().flatten().cuda();candidates=[original]
                for relation in related:
                    enclosure=region_masks[relation['region']].float().flatten().cuda()
                    if (enclosure*original).sum()>=.9*original.sum().clamp_min(1):candidates.append(torch.maximum(original,enclosure))
                overlap=frame_masks@original;areas=frame_masks.sum(1)
                contains=(overlap>=.9*original.sum().clamp_min(1))&(areas>original.sum())
                indices=torch.where(contains)[0]
                if len(indices):
                    indices=indices[torch.argsort(areas[indices])[:3]]
                    candidates=[original]+[torch.maximum(original,frame_masks[i]) for i in indices]
                group=torch.stack(candidates)
                gi=len(groups)
                choices.append(cached['choices'][qi][gi]);margins.append(cached['margins'][qi][gi])
                groups.append(group)
            if groups and args.arm=='joint_view':
                literal=torch.stack([g[0] for g in groups]).amax(0)
                enclosing=torch.stack([g.amax(0) for g in groups]).amax(0)
                groups=[torch.stack([literal,enclosing])]
            hypotheses.append(groups);native_choices.append(choices);native_margins.append(margins)
        print(args.key,'source evidence',frame,flush=True)
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(native_margins=native_margins,native_choices=native_choices,hypotheses=hypotheses,native=native_positive,valid=valid,op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    # Materialize independent per-instance evidence before view aggregation.
    for v in views:
        targets=[];positives=[];negatives=[]
        for qi,groups in enumerate(v['hypotheses']):
            accepted=[g[v['native_choices'][qi][gi]] for gi,g in enumerate(groups)
                      if v['native_margins'][qi][gi][v['native_choices'][qi][gi]]>0]
            target=torch.stack(accepted).amax(0) if accepted else torch.zeros(h*w,device='cuda')
            universe=torch.stack([g.amax(0) for g in groups]).amax(0) if groups else target
            binary=(target.reshape(1,1,h,w)>.5).float()
            positive=(-F.max_pool2d(-binary,3,1,1)>.5).flatten()&v['valid'][:,0]
            # Rejected identities and alternative extents are unknown, not negatives.
            negative=(F.max_pool2d((universe.reshape(1,1,h,w)>.5).float(),3,1,1)<.5).flatten()&~v['native'][:,qi]&v['valid'][:,0]
            targets.append(target);positives.append(positive);negatives.append(negative)
        v['source_target']=torch.stack(targets,1);v['pos']=torch.stack(positives,1);v['neg']=torch.stack(negatives,1)

    result=prior.clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    steps=120
    for qi,text in enumerate(initial['texts']):
        active=[v for v in views if v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not active:continue
        visibility=sum(v['mass'] for v in active);support=(visibility>1e-8)&observed
        seed=prior[:,qi]
        if not ((seed>.5)&observed).any():
            numerator=sum(v['confidence'][qi]*torch.sparse.mm(v['op'].T,v['source_target'][:,qi,None])[:,0] for v in active)
            denominator=sum(v['confidence'][qi]*v['mass'] for v in active)
            seed=torch.where(support,numerator/denominator.clamp_min(1e-8),seed)
        parameter=torch.nn.Parameter(torch.logit(seed.clamp(.001,.999)));optimizer=torch.optim.Adam([parameter],lr=.05)
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
            target=v['source_target'][:,qi]
            known=v['pos'][:,qi]|v['neg'][:,qi]
            bce=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),target,reduction='none')
            overlap=1-(2*(silhouette*target*known).sum()+1)/((silhouette*known).sum()+(target*known).sum()+1)
            objective=v['confidence'][qi]*((bce*known).sum()/known.sum().clamp_min(1)+overlap)
            loss_probability=balanced((response-target).square(),v['pos'][:,qi],v['neg'][:,qi])
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path(args.output_root)/args.arm/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(candidate_selection='native foreground minus complement; per-instance positive admission; alternative extent unknown',key=args.key,arm=args.arm,steps=steps,lr=.05,prior_weight=0,semantic_prior_role='retained independent selection initialization',probability_readout='retained next54, separate from set decoding',seconds=time.monotonic()-start,geometry_frozen=True,selection_carrier_multiplier=1,original_semantic_geometry_unchanged=True,selection_visibility_recomputed=True,subset_temperature='0.5 to 0.1',deployment_cutoff=10.5/255,target_rgb_used=False,source_raster=[h,w],history=history),indent=2)+'\n')

if __name__=='__main__':main()
