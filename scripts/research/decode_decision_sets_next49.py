"""Deployment-matched set inference on fixed geometry and source evidence.

Optimize transient membership only. The separate subset renderer changes
transmittance when nonmembers are removed; geometry/opacities are not saved
or trained. Soft membership is an optimization relaxation, not exact hard
selection. The next44 semantic posterior initializes a separate set decoder;
the decoder is not constrained to reproduce that posterior as a probability
image. It uses source masks only, 120 updates per query, BCE plus overlap loss.
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
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--arm',choices=['baseline','decision','bernoulli','both','proposal_unknown','proposal_positive','proposal_decision','enclosure_required'],default='baseline');args=a.parse_args();torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
    base=Path('output/typed_evidence_next44/soft_identity')/args.key;data=load(base/'field.pt');root=Path(data['root']);state=load(data['base_field'])['state_dict'];observed=state['observed'].cuda()
    initial=load(base/'elements.pt');prior=initial['scores'].cuda();info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    exclusion_reference=load(Path('output/taskwise_scope_composition_next38')/args.key/'anchored/pipeline_calibrated/scores.pt')['scores']['hierarchical_pipeline'].cuda()
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');geometry_prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    records=[];square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
    for frame in info['train_frames']:
        d=load(Path('output/typed_evidence_next44')/args.key/f'identity_{frame}.pt');records.append((frame,d))
        for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    proposal_records=[]
    if args.arm in ['proposal_unknown','proposal_positive','proposal_decision','enclosure_required']:
        episodes=load(root/'episodes.pt');region_masks=episodes['masks'];region_views=episodes['proposal_views'];del episodes
        descriptors=F.normalize(load(root/'teacher_siglip.pt')['descriptors'].float().cuda(),dim=1)
        queries=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].half().float().cuda(),dim=1)
        proposal_scores=descriptors@queries.T
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[]
    for frame,d in records:
        hit=load(root/auth['views'][info['frames'].index(frame)]['relative_path']);op,_,valid=visible_operator(hit,observed,h*w,False)
        target=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in d['instances']],-1).reshape(h*w,-1).cuda();binary=(target.T.reshape(-1,1,h,w)>.5).float()
        pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
        # A contradictory retained semantic response does not certify a
        # positive label. It makes the detector-complement exclusion unknown.
        # Use the existing .5 retrieval decision, no benchmark calibration.
        native_positive=torch.sparse.mm(op,exclusion_reference)>.5
        neg=neg&~native_positive
        pos=pos&native_positive
        if args.arm in ['proposal_unknown','proposal_positive','proposal_decision','enclosure_required']:
            ids=torch.where(region_views==info['frames'].index(frame))[0]
            masks=region_masks[ids].flatten(1).float().cuda();scores=proposal_scores[ids.cuda()]
            for qi,record in enumerate(d['instances']):
                for seed in record['masks']:
                    seed=seed.flatten().float().cuda();overlap=masks@seed
                    # Retain distinct detections. Containment proposes a possible
                    # scope, not equality of whole/part or a new positive label.
                    contains=overlap>=.9*seed.sum().clamp_min(1)
                    intersects=overlap>.5*seed.sum().clamp_min(1)
                    if not contains.any():continue
                    best=int(scores[:,qi].masked_fill(~contains,-float('inf')).argmax())
                    # A larger scope must not reduce native query identity
                    # compared with the best intersecting source proposal.
                    if scores[best,qi]<scores[:,qi].masked_fill(~intersects,-float('inf')).max():continue
                    enclosure=masks[best]>.5
                    neg[:,qi]&=~enclosure
                    if args.arm=='proposal_positive':
                        pos[:,qi]|=enclosure&native_positive[:,qi]&valid[:,0]
                        target[:,qi]=torch.maximum(target[:,qi],enclosure.float())
                    proposal_records.append(dict(frame=frame,query=initial['texts'][qi],region=int(ids[best]),seed_pixels=float(seed.sum()),scope_pixels=int(enclosure.sum()),native_score=float(scores[best,qi])))
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    result=prior.clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    steps=120
    for qi,text in enumerate(initial['texts']):
        if args.arm=='enclosure_required' and len({r['frame'] for r in proposal_records if r['query']==text})<2:
            history.append(dict(query=text,abstained=True,reason='fewer than two source views provide a native-compatible enclosure'));continue
        active=[v for v in views if v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not active:continue
        visibility=sum(v['mass'] for v in active);support=(visibility>1e-8)&observed
        parameter=torch.nn.Parameter(torch.logit(prior[:,qi].clamp(.001,.999)));optimizer=torch.optim.Adam([parameter],lr=.05)
        for step in range(steps):
            v=active[step%len(active)];p=torch.where(support,parameter.sigmoid(),prior[:,qi]);response=torch.sparse.mm(v['op'],p[:,None])[:,0]
            gate=(torch.rand_like(p)<p).float()+p-p.detach() if args.arm in ['bernoulli','both'] else p
            member=gate*observed
            _,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*member,colors=ones,viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
            silhouette=alpha[0,...,0].flatten();known=v['pos'][:,qi]|v['neg'][:,qi];target=v['y'][:,qi]
            if args.arm in ['decision','both','proposal_decision']:
                # Existing deployment: round(alpha*255)>10, boundary10.5/255.
                cutoff=10.5/255
                silhouette=torch.sigmoid(torch.logit(silhouette.clamp(1e-5,1-1e-5))-torch.logit(torch.tensor(cutoff,device=silhouette.device)))
            bce=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),target,reduction='none')
            overlap=1-(2*(silhouette*target*known).sum()+1)/((silhouette*known).sum()+(target*known).sum()+1)
            objective=v['confidence'][qi]*((bce*known).sum()/known.sum().clamp_min(1)+overlap)
            loss_probability=balanced((response-target).square(),v['pos'][:,qi],v['neg'][:,qi])
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path('output/decision_sets_next49')/args.arm/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=result.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(key=args.key,arm=args.arm,steps=steps,lr=.05,prior_weight=0,semantic_prior_role='initialization only',probability_readout='retained next44, separate from set decoding',seconds=time.monotonic()-start,geometry_frozen=True,stochastic_membership=args.arm in ['bernoulli','both'],gradient_estimator='biased straight-through Bernoulli' if args.arm in ['bernoulli','both'] else 'continuous',deployment_alpha_boundary=10.5/255,decision_matched=args.arm in ['decision','both','proposal_decision'],seed=0,subset_temperature=1.,target_rgb_used=False,proposal_relations=proposal_records,proposal_role='unknown enclosure' if args.arm in ['proposal_unknown','proposal_decision','enclosure_required'] else 'positive expansion' if args.arm=='proposal_positive' else None,source_raster=[h,w],history=history),indent=2)+'\n')

if __name__=='__main__':main()
