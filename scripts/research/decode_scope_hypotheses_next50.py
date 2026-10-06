"""Competing partial-scope hypotheses coupled through frozen 3D geometry.

Equal-prior soft-min or hard-min of qualified detector/enclosure losses.
This is a latent energy objective, not a calibrated likelihood of completeness.
Only transient membership changes; native semantic output stays frozen.
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
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--arm',choices=['latent','hard','unknown','crossview','corroborated'],default='latent');args=a.parse_args();torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
    base=Path('output/typed_evidence_next44/soft_identity')/args.key;data=load(base/'field.pt');root=Path(data['root']);state=load(data['base_field'])['state_dict'];observed=state['observed'].cuda()
    initial=load(base/'elements.pt');prior=initial['scores'].cuda();info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    exclusion_reference=load(Path('output/taskwise_scope_composition_next38')/args.key/'anchored/pipeline_calibrated/scores.pt')['scores']['hierarchical_pipeline'].cuda()
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');geometry_prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    records=[];square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
    for frame in info['train_frames']:
        d=load(Path('output/typed_evidence_next44')/args.key/f'identity_{frame}.pt');records.append((frame,d))
        for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    proposal_records=[]
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
        raw_positive=pos.clone()
        pos=pos&native_positive
        alternate=target.clone()
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
                alternate[:,qi]=torch.maximum(alternate[:,qi],enclosure.float())
                if args.arm=='unknown':neg[:,qi]&=~enclosure
                proposal_records.append(dict(frame=frame,query=initial['texts'][qi],region=int(ids[best]),seed_pixels=float(seed.sum()),scope_pixels=int(enclosure.sum()),native_score=float(scores[best,qi])))
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(raw_positive=raw_positive,alternate=alternate,native=native_positive,valid=valid,op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    if args.arm=='corroborated':
        votes=[]
        for v in views:
            adj=v['op'].T.coalesce()
            visibility=(v['mass'][:,None]>1e-8).float()*v['confidence'][None]
            pv=torch.sparse.mm(adj,v['raw_positive'].float())/v['mass'][:,None].clamp_min(1e-8)*v['confidence'][None]
            votes.append((pv,visibility))
        ps=sum(p for p,n in votes);vs=sum(n for p,n in votes)
        for v,(pv,visibility) in zip(views,votes):
            other_mass=(vs-visibility).clamp_min(0)
            other=torch.sparse.mm(v['op'],(ps-pv).clamp_min(0)/other_mass.clamp_min(1e-8))
            corroborated=other>.5
            v['pos']|=v['raw_positive']&corroborated
            v['neg']&=~corroborated
        del votes,ps,vs
    if args.arm=='crossview':
        # Independently qualify each hypothesis using all OTHER source views.
        # Per-element votes are visibility normalized before view aggregation.
        votes=[]
        for v in views:
            adj=v['op'].T.coalesce()
            den=v['mass'][:,None].clamp_min(1e-8)
            pv=torch.sparse.mm(adj,v['pos'].float())/den*v['confidence'][None]
            nv=torch.sparse.mm(adj,v['neg'].float())/den*v['confidence'][None]
            votes.append((pv,nv))
        ps=sum(p for p,n in votes);ns=sum(n for p,n in votes)
        for v,(pv,nv) in zip(views,votes):
            other_p=torch.sparse.mm(v['op'],ps-pv).clamp_min(0)
            other_n=torch.sparse.mm(v['op'],ns-nv).clamp_min(0)
            extension=(v['alternate']-v['y']).clamp_min(0)
            pp=(extension*other_p).sum(0);nn=(extension*other_n).sum(0)
            v['enclosure_prior']=torch.where(pp+nn>1e-8,pp/(pp+nn).clamp_min(1e-8),torch.full_like(pp,.5)).clamp(.05,.95)
        del votes,ps,ns,pv,nv,other_p,other_n
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
            _,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*member,colors=ones,viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
            silhouette=alpha[0,...,0].flatten();known=v['pos'][:,qi]|v['neg'][:,qi];target=v['y'][:,qi]
            bce=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),target,reduction='none')
            overlap=1-(2*(silhouette*target*known).sum()+1)/((silhouette*known).sum()+(target*known).sum()+1)
            objective=v['confidence'][qi]*((bce*known).sum()/known.sum().clamp_min(1)+overlap)
            if args.arm in ['latent','hard','crossview']:
                # Competing view-level scope hypotheses share one 3D membership.
                # Both retain native-field conflict-to-unknown qualification.
                alternative=v['alternate'][:,qi]
                ab=(alternative.reshape(1,1,h,w)>.5).float()
                ap=(-F.max_pool2d(-ab,3,1,1)>.5).flatten()&v['native'][:,qi]&v['valid'][:,0]
                an=(F.max_pool2d(ab,3,1,1)<.5).flatten()&~v['native'][:,qi]&v['valid'][:,0]
                ak=ap|an
                ae=F.binary_cross_entropy(silhouette.clamp(1e-5,1-1e-5),alternative,reduction='none')
                al=(ae*ak).sum()/ak.sum().clamp_min(1)+1-(2*(silhouette*alternative*ak).sum()+1)/((silhouette*ak).sum()+(alternative*ak).sum()+1)
                losses=torch.stack([objective/v['confidence'][qi],al])
                # Equal-prior mixture; no target GT or query-specific selection.
                if args.arm=='crossview':
                    ep=v['enclosure_prior'][qi]
                    log_prior=torch.stack([(1-ep).log(),ep.log()])
                else:log_prior=torch.full_like(losses,-0.6931471805599453)
                objective=v['confidence'][qi]*(losses.min() if args.arm=='hard' else -.1*torch.logsumexp(log_prior-losses/.1,0))

            loss_probability=balanced((response-target).square(),v['pos'][:,qi],v['neg'][:,qi])
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path('output/scope_hypotheses_next50')/args.arm/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=result.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    torch.save(dict(texts=initial['texts'],scores=initial['scores'],selection_scores=result.cpu(),readout_contract='semantic_probability_and_independent_selected_set',semantic_parent=str(base/'elements.pt')),out/'typed_readout.pt')
    (out/'fit.json').write_text(json.dumps(dict(hypothesis_temperature=.1,scope_hypotheses='detector and native-compatible enclosure, shared 3D membership',key=args.key,arm=args.arm,steps=steps,lr=.05,prior_weight=0,semantic_prior_role='initialization only',probability_readout='retained next44, separate from set decoding',seconds=time.monotonic()-start,geometry_frozen=True,stochastic_membership=False,gradient_estimator='continuous',deployment_alpha_boundary=10.5/255,decision_matched=False,seed=0,subset_temperature=1.,target_rgb_used=False,proposal_relations=proposal_records,proposal_role='competing extent hypothesis' if args.arm!='unknown' else 'unknown enclosure',source_raster=[h,w],crossview_priors=[v['enclosure_prior'].cpu().tolist() for v in views] if args.arm=='crossview' else None,history=history),indent=2)+'\n')

if __name__=='__main__':main()
