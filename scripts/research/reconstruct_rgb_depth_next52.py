"""Source RGB reconstruction constrained by calibrated MoGe expected depth."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization
from radio_gs.scripts.repair_moge_geometry import load_views
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior


def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--steps',type=int,default=1500);p.add_argument('--ray-distribution',action='store_true');p.add_argument('--recover-calibration',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
    field=load(Path('output/taskwise_scope_composition_next38')/('lerf_'+a.scene)/'field.pt')
    report=json.loads(Path(field['geometry_report']).read_text());base=load(report['results']['independent_scale']['geometry'])
    state={k:v.cuda() for k,v in base['state_dict'].items()};info=json.loads((Path(field['root'])/'teachers.json').read_text())
    cameras=MogeGeometryPrior(report['source_report'],report['source_report_sha256']);views={}
    for f in info['train_frames']+info['validation_frames']:
        im=Image.open(f'/mnt/pool/sqy/3d_understanding/lerf_ovs/{a.scene}/images/frame_{f:05d}.jpg').convert('RGB')
        width=512;height=round(im.height*width/im.width);im=im.resize((width,height),Image.Resampling.LANCZOS)
        cam=cameras.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
        views[f]=(torch.from_numpy(np.asarray(im).copy()).cuda().float()/255,cam.world_to_camera.cuda()[None],k[None])
    recovered=[]
    if a.recover_calibration:
        for r in report.get('recovered_scales',[]):
            f=r['frame']
            if r['accepted'] and f in cameras.calibration and not cameras.calibration[f]['accepted']:
                cameras.calibration[f]=dict(cameras.calibration[f],accepted=True,scale=r['scale'],offset=0.)
                recovered.append(f)
    cameras.records={f:r for f,r in cameras.records.items() if f in views}
    depth_views={v['frame']:v for v in load_views(cameras,512,'cuda')}
    color=torch.nn.Parameter(torch.zeros_like(state['xyz']));opacity=torch.nn.Parameter(torch.logit(state['opacities'].flatten()))
    log_size=torch.nn.Parameter(torch.zeros_like(state['scales']))
    offset=torch.nn.Parameter(torch.zeros_like(state['xyz']))
    radius=state['scales'].amax(1,keepdim=True).detach()
    optimizer=torch.optim.Adam([dict(params=[color],lr=.03),dict(params=[opacity],lr=.01),dict(params=[log_size],lr=.003),dict(params=[offset],lr=.003)])
    out=Path('output/carrier_ray_next52' if a.ray_distribution else 'output/carrier_depth_next52_recovered' if a.recover_calibration else 'output/carrier_depth_next52')/a.scene;out.mkdir(parents=True,exist_ok=True)
    def render(f):
        target,pose,k=views[f];h,w=target.shape[:2]
        xyz=state['xyz']+radius*offset.tanh()
        z=xyz@pose[0,2,:3]+pose[0,2,3]
        colors=torch.cat([color.sigmoid(),z.square()[:,None]],1) if a.ray_distribution else color.sigmoid()
        rgb,alpha,_=rasterization(means=state['xyz']+radius*offset.tanh(),quats=state['quaternions'],scales=state['scales']*log_size.exp(),opacities=opacity.sigmoid(),colors=colors,viewmats=pose,Ks=k,width=w,height=h,near_plane=.001*cameras.scale,far_plane=100*cameras.scale,render_mode="RGB+ED",packed=False)
        variance=(rgb[0,:,:,3]/alpha[0,:,:,0].clamp_min(1e-8)-rgb[0,:,:,-1].square()).clamp_min(0) if a.ray_distribution else torch.zeros_like(alpha[0,:,:,0])
        return rgb[0,:,:,:3],target,rgb[0,:,:,-1],alpha[0,:,:,0],variance
    @torch.no_grad()
    def evaluate(tag):
        result={}
        for split,frames in [('train',info['train_frames']),('validation',info['validation_frames'])]:
            mse=[];depth_errors=[];coverage=[]
            for f in frames:
                rgb,target,depth,alpha,variance=render(f);mse.append(float((rgb-target).square().mean()))
                if f in depth_views:
                    dv=depth_views[f];valid=dv['valid']&(alpha>.1)&torch.isfinite(depth)&(depth>0)
                    depth_errors.append(float(((depth[valid]-dv['depth'][valid]).abs()/dv['depth'][valid]).mean()))
                    coverage.append(float(valid.sum()/dv['valid'].sum().clamp_min(1)))
                if split=='validation':Image.fromarray((rgb.clamp(0,1)*255).byte().cpu().numpy()).save(out/f'{tag}_{f}.png')
            result[split]=dict(moge_absrel=float(np.mean(depth_errors)) if depth_errors else None,moge_coverage=float(np.mean(coverage)) if coverage else None,mse=float(np.mean(mse)),psnr=float(np.mean([-10*np.log10(x) for x in mse])))
        return result
    before=evaluate('before');history=[]
    # Fixed source ordering, no semantic labels and no benchmark selection.
    frames=info['train_frames'];order=torch.randperm(len(frames)).tolist()
    for step in range(a.steps):
        if step%len(frames)==0:order=torch.randperm(len(frames)).tolist()
        f=frames[order[step%len(frames)]];rgb,target,depth,alpha,variance=render(f)
        photometric=(rgb-target).abs().mean();loss=photometric+.001*log_size.square().mean()+.001*offset.tanh().square().mean()
        depth_loss=depth.sum()*0
        if f in depth_views:
            dv=depth_views[f];valid=dv['valid']&(alpha.detach()>.1)&torch.isfinite(depth)&(depth>0)
            # Relative L1 balances near/far surfaces; calibrated depth is a
            # source prior, not GT. No supervision from validation views.
            residual=(depth[valid]-dv['depth'][valid]).square()
            residual=(residual+variance[valid]+1e-12).sqrt() if a.ray_distribution else (depth[valid]-dv['depth'][valid]).abs()
            depth_loss=(residual/dv['depth'][valid]).clamp_max(1).mean()
            loss=loss+.1*depth_loss
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%100==0:
            record=dict(depth_loss=float(depth_loss),step=step+1,l1=float(photometric),median_size=float(log_size.exp().median()),median_opacity=float(opacity.sigmoid().median()))
            history.append(record);print(record,flush=True)
    after=evaluate('after')
    result={k:v.detach().cpu() for k,v in state.items()};result['xyz']=(state['xyz']+radius*offset.tanh()).detach().cpu();result['scales']=(state['scales']*log_size.exp()).detach().cpu();result['opacities']=opacity.sigmoid().detach().cpu()[:,None]
    payload=dict(base);payload['state_dict']=result;payload['source_rgb_depth_fit_next52']=dict(frames=frames,steps=a.steps,centres_and_normals_fixed=False,rotations_fixed=True,centre_displacement_bound="one maximum parent sigma per axis",query_or_label_access=False)
    torch.save(payload,out/'geometry.pt');torch.save(color.sigmoid().detach().cpu(),out/'rgb.pt')
    meta=dict(ray_distribution=a.ray_distribution,recovered_calibration_frames=recovered,depth_weight=.1,depth_training_frames=[f for f in frames if f in depth_views],depth_reference='calibrated MoGe, not GT',scene=a.scene,steps=a.steps,seconds=time.monotonic()-start,before=before,after=after,history=history,source_frames=frames,validation_frames=info['validation_frames'],centres_fixed=False,normals_fixed=True,geometry_frozen_before_semantic_rebuild=True,benchmark_masks_used=False)
    (out/'fit.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta,flush=True)

if __name__=='__main__':main()
