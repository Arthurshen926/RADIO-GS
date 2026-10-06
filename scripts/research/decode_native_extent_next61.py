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
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--arm',choices=['native'],default='native');a.add_argument('--geometry-root',default='output/carrier_depth_next52_recovered');a.add_argument('--output-root',default='output/native_extent_next61');args=a.parse_args();torch.set_num_threads(4);start=time.monotonic()
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
    from PIL import Image
    import numpy as np
    from radio_gs.querying.query_region_evidence import same_window
    from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
    src=json.loads((root/'source_frames.json').read_text());source_frames={r['frame_idx']:r for r in src['frames']}
    teacher=load(root/info['teacher_files']['siglip']['path'])
    runtime=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'))
    query=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float().cuda(),dim=-1)
    evidence_out=Path(args.output_root)/'source_evidence'/args.key;evidence_out.mkdir(parents=True,exist_ok=True)
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
        rgb=torch.from_numpy(np.array(Image.open(Path(src['image_dir'])/source_frames[frame]['source_file']).convert('RGB'),copy=True)).permute(2,0,1).float()/255
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
                full_masks=F.interpolate((group.reshape(-1,1,h,w)>.5).float().cpu(),size=rgb.shape[1:],mode='nearest-exact')[:,0].bool()
                # All competing extents see exactly the same source window.
                _,box=same_window(rgb,full_masks.any(0));x0,y0,x1,y1=box
                crop=rgb[:,y0:y1,x0:x1];encoded=[]
                for candidate_mask in full_masks:
                    local=candidate_mask[y0:y1,x0:x1][None]
                    pair=torch.stack([torch.where(local,crop,.5),torch.where(local,.5,crop)])
                    pair=F.interpolate(pair,(384,384),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                    encoded.append(runtime.encode(pair).float())
                features=torch.stack(encoded)
                margin=(features[:,0]-features[:,1])@query[qi]
                choices.append(int(margin.argmax()));margins.append(margin.cpu())
                groups.append(group)
            if groups and args.arm=='joint_view':
                literal=torch.stack([g[0] for g in groups]).amax(0)
                enclosing=torch.stack([g.amax(0) for g in groups]).amax(0)
                groups=[torch.stack([literal,enclosing])]
            hypotheses.append(groups);native_choices.append(choices);native_margins.append(margins)
        torch.save(dict(texts=initial['texts'],choices=native_choices,margins=native_margins),evidence_out/f'{frame}.pt')
        print(args.key,'source evidence',frame,flush=True)
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(native_choices=native_choices,hypotheses=hypotheses,native=native_positive,valid=valid,op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    del runtime;torch.cuda.empty_cache()
    result=prior.clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    steps=120
    for qi,text in enumerate(initial['texts']):
        active=[v for v in views if v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not active:continue
        visibility=sum(v['mass'] for v in active);support=(visibility>1e-8)&observed
        choice_counts=[0,0]
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
            groups=v['hypotheses'][qi];chosen=[]
            with torch.no_grad():
                for gi,candidates in enumerate(groups):
                    index=v['native_choices'][qi][gi]
                    chosen.append(candidates[index]);choice_counts[int(index>0)]+=1
                target=torch.stack(chosen).amax(0) if chosen else v['y'][:,qi]
                binary=(target.reshape(1,1,h,w)>.5).float()
                positive=(-F.max_pool2d(-binary,3,1,1)>.5).flatten()&v['native'][:,qi]&v['valid'][:,0]
                negative=(F.max_pool2d(binary,3,1,1)<.5).flatten()&~v['native'][:,qi]&v['valid'][:,0]
                known=positive|negative
            bce=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),target,reduction='none')
            overlap=1-(2*(silhouette*target*known).sum()+1)/((silhouette*known).sum()+(target*known).sum()+1)
            objective=v['confidence'][qi]*((bce*known).sum()/known.sum().clamp_min(1)+overlap)
            loss_probability=balanced((response-target).square(),v['pos'][:,qi],v['neg'][:,qi])
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),literal_choices=choice_counts[0],enclosure_choices=choice_counts[1],final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path(args.output_root)/args.arm/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(candidate_selection='native foreground minus complement in shared source window',key=args.key,arm=args.arm,steps=steps,lr=.05,prior_weight=0,semantic_prior_role='retained independent selection initialization',probability_readout='retained next54, separate from set decoding',seconds=time.monotonic()-start,geometry_frozen=True,selection_carrier_multiplier=1,original_semantic_geometry_unchanged=True,selection_visibility_recomputed=True,subset_temperature='0.5 to 0.1',deployment_cutoff=10.5/255,target_rgb_used=False,source_raster=[h,w],history=history),indent=2)+'\n')

if __name__=='__main__':main()
