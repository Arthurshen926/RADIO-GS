"""Reconstruct membership using supplemental native source-region recognition.

Same next66 geometry, frozen semantic field and240steps as next68. Compare
supplementing only queries lacking fit views versus all runtime queries.
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
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--mode',choices=['cold','augment','replace','local_exclusion'],required=True);args=a.parse_args();args.carrier_arm='adaptive_allrgb';args.arm='decision';args.scope_negatives=False;torch.set_num_threads(4);start=time.monotonic()
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
    from PIL import Image
    src=json.loads((root/'source_frames.json').read_text());source_paths={r['frame_idx']:Path(src['image_dir'])/r['source_file'] for r in src['frames']}
    trained={r['query'] for r in json.loads((Path('output/subset_consistency_next67/exposed')/args.key/'fit.json').read_text())['history']}
    for frame in info['train_frames']:
        d=load(Path('output/typed_evidence_next44')/args.key/f'identity_{frame}.pt')
        extra=load(Path('output/zoom_grounding_next70/evidence')/args.key/f'identity_{frame}.pt')
        for qi,text in enumerate(initial['texts']):
            if args.mode=='cold' and text in trained:continue
            for name in ['masks','scores','foreground_margin']:
                if args.mode in ('replace','local_exclusion') and len(extra['instances'][qi]['scores']):d['instances'][qi][name]=extra['instances'][qi][name].clone()
                else:d['instances'][qi][name]=torch.cat([d['instances'][qi][name],extra['instances'][qi][name]],dim=0)
        d['local_windows']=extra['windows'];d['image_size']=Image.open(source_paths[frame]).size
        d['local_present']=torch.tensor([bool(len(r['scores'])) for r in extra['instances']])
        records.append((frame,d))
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
        detector=target.clone()
        present=torch.tensor([bool(t['frames'].get(frame,{}).get('present',False)) and frame!=t['seed_frame'] for t in tracks],device='cuda')
        temporal=torch.stack([t['frames'][frame]['probability'].float() if bool(present[qi]) else torch.zeros(h,w) for qi,t in enumerate(tracks)],-1).reshape(h*w,-1).cuda()
        target=torch.where(d['local_present'].cuda()[None],target,torch.maximum(target,temporal)) if args.mode in ('replace','local_exclusion') else torch.maximum(target,temporal)
        binary=(target.T.reshape(-1,1,h,w)>.5).float()
        pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
        # A contradictory retained semantic response does not certify a
        # positive label. It makes the detector-complement exclusion unknown.
        # Use the existing .5 retrieval decision, no benchmark calibration.
        native_positive=torch.sparse.mm(op_old,exclusion_reference)>.5
        if args.arm in ('agreement','decision','anneal'):
            local_window=torch.zeros(h,w,len(initial['texts']),device='cuda',dtype=torch.bool)
            if args.mode=='local_exclusion':
                W,H=d['image_size']
                for qi,t in enumerate(initial['texts']):
                    if d['local_present'][qi]:
                        x0,y0,x1,y1=d['local_windows'][t]
                        local_window[int(y0*h/H):int(y1*h/H),int(x0*w/W):int(x1*w/W),qi]=True
            neg=neg&(~native_positive|local_window.reshape(h*w,-1))
            pos=pos&native_positive
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        detector_confidence=confidence.clone()
        confidence=torch.maximum(confidence,torch.tensor(seed_confidence,device='cuda')*present)
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,detector=detector,detector_confidence=detector_confidence,valid=valid,view=camera.world_to_camera.cuda()[None],k=k[None]))
    previously_active=torch.tensor([t in trained for t in initial['texts']],device='cuda')
    masses=[];positives=[]
    for v in views:
        detector_binary=(v['detector'].T.reshape(-1,1,h,w)>.5).float()
        detector_pos=(-F.max_pool2d(-detector_binary,3,1,1)>.5).flatten(1).T&v['valid']
        detector_neg=(F.max_pool2d(detector_binary,3,1,1)<.5).flatten(1).T&v['valid']
        weight=v['detector_confidence'][None]*(detector_pos|detector_neg)
        v['detector_pos']=detector_pos
        masses.append(torch.sparse.mm(v['op'].T,weight))
        positives.append(torch.sparse.mm(v['op'].T,weight*detector_pos))
    total_mass=sum(masses);total_positive=sum(positives);admission=[]
    for v,m,n in zip(views,masses,positives):
        other_mass=total_mass-m;other=(total_positive-n)/other_mass.clamp_min(1e-8)
        coverage=torch.sparse.mm(v['op'],(other_mass>1e-8).float())
        corroboration=torch.sparse.mm(v['op'],other)/coverage.clamp_min(1e-8)
        extra=v['detector_pos']&(coverage>.5)&(corroboration>.5)&~v['pos']
        v['pos']|=extra;v['neg']&=~extra;admission.append(extra.sum(0).cpu())
    newly_active=(torch.stack([v['pos'].any(0)&(v['confidence']>0) for v in views]).any(0))&~previously_active
    if newly_active.any():
        seed=total_positive/total_mass.clamp_min(1e-8)
        prior[:,newly_active]=torch.maximum(prior[:,newly_active],seed[:,newly_active])
    admission_counts=torch.stack(admission).sum(0).tolist()
    del masses,positives,total_mass,total_positive
    result=prior.clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    steps=240
    for qi,text in enumerate(initial['texts']):
        active=[v for v in views if v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not active:continue
        visibility=sum(v['mass'] for v in active);support=observed
        parameter=torch.nn.Parameter(torch.logit(prior[:,qi].clamp(.001,.999)));optimizer=torch.optim.Adam([parameter],lr=.05)
        for step in range(steps):
            v=active[step%len(active)];p=torch.where(support,parameter.sigmoid(),prior[:,qi]);response=torch.sparse.mm(v['op'],p[:,None])[:,0]
            member=p*observed
            if args.arm in ('decision','anneal'):
                temperature=.5*(.1/.5)**(step/max(steps-1,1))
                member=torch.where(support,(parameter/temperature).sigmoid(),(prior[:,qi]>.5).float())*observed
            if args.mode in ('hard','hard_exposed'):
                member=(member>.5).float()+member-member.detach()
            _,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*member,colors=ones,viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
            silhouette=alpha[0,...,0].flatten()
            if args.arm=='decision':
                # Frozen scorer rounds alpha to a byte and accepts values >10.
                cutoff=10.5/255
                silhouette=torch.sigmoid(torch.logit(silhouette.clamp(1e-6,1-1e-6))-torch.logit(torch.tensor(cutoff,device='cuda')))
            known=v['pos'][:,qi]|v['neg'][:,qi];target=v['y'][:,qi]
            bce=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),target,reduction='none')
            overlap=1-(2*(silhouette*target*known).sum()+1)/((silhouette*known).sum()+(target*known).sum()+1)
            objective=v['confidence'][qi]*((bce*known).sum()/known.sum().clamp_min(1)+overlap)
            loss_probability=balanced((response-target).square(),v['pos'][:,qi],v['neg'][:,qi])
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path('output/zoom_grounding_next70')/args.mode/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),semantic_geometry=binding,semantic_observed=observed.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),selection_geometry=binding,selection_observed=observed.cpu(),semantic_geometry=binding,semantic_observed=observed.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(mode=args.mode,steps=steps,source_raster=[h,w],seconds=time.monotonic()-start,geometry=str(geometry_path),source_evidence='native region searches -> cropped SAM3 exact-query segmentation -> native foreground qualification; replace arm suppresses stale detector/temporal union when a crop mask exists',hard_forward=args.mode in ('hard','hard_exposed'),optimize_all_observed=True,admitted_source_pixels=dict(zip(initial['texts'],admission_counts)),newly_active=[t for t,b in zip(initial['texts'],newly_active.tolist()) if b],semantic_scores_unchanged=True,target_rgb_used=False,history=history),indent=2)+'\n')

if __name__=='__main__':main()
