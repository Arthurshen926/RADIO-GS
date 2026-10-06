"""Joint probability/soft-subset rendering on fixed geometry and source evidence.

Optimize transient membership only. The separate subset renderer changes
transmittance when nonmembers are removed; geometry/opacities are not saved
or trained. Soft membership is an optimization relaxation, not exact hard
selection. Both arms use identical source evidence, schedule and prior.
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
    a=argparse.ArgumentParser();a.add_argument('--key',required=True);a.add_argument('--arm',choices=['probability','joint'],required=True);args=a.parse_args();torch.set_num_threads(4);start=time.monotonic()
    base=Path('output/typed_evidence_next44/soft_identity')/args.key;data=load(base/'field.pt');root=Path(data['root']);state=load(data['base_field'])['state_dict'];observed=state['observed'].cuda()
    initial=load(base/'elements.pt');prior=initial['scores'].cuda();info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');geometry_prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    records=[];square=torch.zeros(prior.shape[1]);count=torch.zeros_like(square)
    for frame in info['train_frames']:
        d=load(Path('output/typed_evidence_next44')/args.key/f'identity_{frame}.pt');records.append((frame,d))
        for j,r in enumerate(d['instances']):square[j]+=r['foreground_margin'].square().sum();count[j]+=len(r['scores'])
    scale=(square/count.clamp_min(1)).sqrt().clamp_min(1e-6);views=[]
    for frame,d in records:
        hit=load(root/auth['views'][info['frames'].index(frame)]['relative_path']);op,_,valid=visible_operator(hit,observed,h*w,False)
        target=torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w) for r in d['instances']],-1).reshape(h*w,-1).cuda();binary=(target.T.reshape(-1,1,h,w)>.5).float()
        pos=(-F.max_pool2d(-binary,3,1,1)>.5).flatten(1).T&valid;neg=(F.max_pool2d(binary,3,1,1)<.5).flatten(1).T&valid
        confidence=torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max()) if len(r['scores']) else 0. for j,r in enumerate(d['instances'])],device='cuda')
        camera=geometry_prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
        mass=torch.sparse.sum(op,dim=0).to_dense();views.append(dict(op=op,y=target,pos=pos,neg=neg,confidence=confidence,mass=mass,view=camera.world_to_camera.cuda()[None],k=k[None]))
    result=prior.clone();history=[];ones=torch.ones(len(prior),1,device='cuda')
    for qi,text in enumerate(initial['texts']):
        active=[v for v in views if v['confidence'][qi]>0 and v['pos'][:,qi].any()]
        if not active:continue
        visibility=sum(v['mass'] for v in active);support=(visibility>1e-8)&observed
        parameter=torch.nn.Parameter(torch.logit(prior[:,qi].clamp(.001,.999)));optimizer=torch.optim.Adam([parameter],lr=.05)
        for step in range(60):
            v=active[step%len(active)];p=torch.where(support,parameter.sigmoid(),prior[:,qi]);response=torch.sparse.mm(v['op'],p[:,None])[:,0]
            loss_probability=balanced((response-v['y'][:,qi]).square(),v['pos'][:,qi],v['neg'][:,qi]);loss=loss_probability
            if args.arm=='joint':
                temperature=1.-.8*step/59
                member=torch.sigmoid(torch.logit(p.clamp(.001,.999))/temperature)*observed
                _,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten()*member,colors=ones,viewmats=v['view'],Ks=v['k'],width=w,height=h,near_plane=.001*geometry_prior.scale,far_plane=100*geometry_prior.scale,packed=False)
                silhouette=alpha[0,...,0].flatten()
                loss=loss+balanced((silhouette-v['y'][:,qi]).square(),v['pos'][:,qi],v['neg'][:,qi])
            regularizer=((p-prior[:,qi]).square()*visibility).sum()/visibility.sum().clamp_min(1e-8)
            objective=v['confidence'][qi]*loss+.1*regularizer
            optimizer.zero_grad();objective.backward();optimizer.step()
        result[:,qi]=torch.where(support,parameter.sigmoid().detach(),prior[:,qi]);history.append(dict(query=text,views=len(active),final_probability_loss=float(loss_probability.detach()),final_objective=float(objective.detach())))
        print(args.key,args.arm,qi,text,'done',flush=True)
    out=Path('output/subset_readout_next45')/args.arm/args.key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=initial['texts'],scores=result.cpu()),out/'elements.pt');shutil.copyfile(base/'field.pt',out/'field.pt');shutil.copyfile(base/'calibration.pt',out/'calibration.pt')
    (out/'fit.json').write_text(json.dumps(dict(key=args.key,arm=args.arm,steps=60,lr=.05,prior_weight=.1,seconds=time.monotonic()-start,geometry_frozen=True,subset_temperature=[1.,.2],target_rgb_used=False,source_raster=[h,w],history=history),indent=2)+'\n')

if __name__=='__main__':main()
