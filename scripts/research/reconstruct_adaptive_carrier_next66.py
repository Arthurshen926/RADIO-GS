"""Joint source RGB+MoGe reconstruction with a matched fixed-count control."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from radio_gs.scripts.repair_moge_geometry import load_views
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.train_rgb_gs import ssim_loss

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--arm',choices=['fixed','adaptive','adaptive_allrgb'],required=True);p.add_argument('--steps',type=int,default=3000);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
 old=Path('output/carrier_depth_next52_recovered')/a.scene;base=load(old/'geometry.pt');s={k:v.cuda() for k,v in base['state_dict'].items()}
 data=load(Path('output/taskwise_scope_composition_next38')/('lerf_'+a.scene)/'field.pt');root=Path(data['root']);gr=json.loads(Path(data['geometry_report']).read_text());info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());paths={r['frame_idx']:Path(src['image_dir'])/r['source_file'] for r in src['frames']};gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);views={}
 original_train=list(info['train_frames'])
 if a.arm=='adaptive_allrgb':
  forbidden={int(p.stem.split('_')[-1]) for p in (Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene).glob('frame_*.json')}|set(info['validation_frames'])
  paths.update({int(p.stem.split('_')[-1]):p for p in Path(src['image_dir']).glob('frame_*.jpg')})
  info['train_frames']=sorted(set(paths)&set(gp.cameras)-forbidden)
 for f in info['train_frames']+info['validation_frames']:
  im=Image.open(paths[f]).convert('RGB');w=512;h=round(im.height*w/im.width);im=im.resize((w,h),Image.Resampling.LANCZOS);cam=gp.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
  views[f]=(torch.from_numpy(np.array(im,copy=True)).cuda().float()/255,cam.world_to_camera.cuda()[None],k[None])
 for r in gr.get('recovered_scales',[]):
  f=r['frame']
  if r['accepted'] and f in gp.calibration and not gp.calibration[f]['accepted']:gp.calibration[f]=dict(gp.calibration[f],accepted=True,scale=r['scale'],offset=0.)
 gp.records={f:r for f,r in gp.records.items() if f in views};depths={v['frame']:v for v in load_views(gp,512,'cuda')}
 rgb=load(old/'rgb.pt').cuda();n=len(rgb);extent=float(torch.linalg.vector_norm(s['xyz']-s['xyz'].median(0).values,dim=1).quantile(.9))
 params=torch.nn.ParameterDict(dict(means=torch.nn.Parameter(s['xyz'].clone()),quats=torch.nn.Parameter(s['quaternions'].clone()),scales=torch.nn.Parameter(s['scales'].log()),opacities=torch.nn.Parameter(torch.logit(s['opacities'].flatten().clamp(1e-6,1-1e-6))),sh0=torch.nn.Parameter(((rgb-.5)/.28209479177387814)[:,None]),shN=torch.nn.Parameter(torch.zeros(n,8,3,device='cuda'))))
 rates=dict(means=.00016*extent,quats=.001,scales=.005,opacities=.05,sh0=.0025,shN=.000125);opts={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
 strategy=DefaultStrategy(refine_start_iter=300,refine_stop_iter=1800,refine_every=100,reset_every=3000,verbose=True);strategy.check_sanity(params,opts);st=strategy.initialize_state(scene_scale=extent)
 out=Path('output/joint_carrier_next66')/a.arm/a.scene;out.mkdir(parents=True,exist_ok=True)
 def render(f,degree):
  target,pose,k=views[f];h,w=target.shape[:2]
  result,alpha,aux=rasterization(means=params['means'],quats=F.normalize(params['quats'],dim=-1),scales=params['scales'].exp(),opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],1),sh_degree=degree,viewmats=pose,Ks=k,width=w,height=h,near_plane=.001*gp.scale,far_plane=100*gp.scale,render_mode='RGB+ED',packed=False)
  return result[0,:,:,:3],target,result[0,:,:,-1],alpha[0,:,:,0],aux
 @torch.no_grad()
 def evaluate(tag,degree):
  result={}
  for split,frames in [('train',info['train_frames']),('validation',info['validation_frames'])]:
   psnr=[];errors=[];coverage=[]
   for f in frames:
    prediction,target,depth,alpha,_=render(f,degree);psnr.append(float(-10*torch.log10((prediction-target).square().mean().clamp_min(1e-12))))
    if f in depths:
     d=depths[f];valid=d['valid']&(alpha>.1)&(depth>0)&torch.isfinite(depth)
     if valid.any():errors.append(float(((depth[valid]-d['depth'][valid]).abs()/d['depth'][valid]).mean()))
     coverage.append(float(valid.sum()/d['valid'].sum().clamp_min(1)))
    if split=='validation':Image.fromarray((prediction.clamp(0,1)*255).byte().cpu().numpy()).save(out/f'{tag}_{f}.png')
   result[split]=dict(psnr=float(np.mean(psnr)),moge_absrel=float(np.mean(errors)) if errors else None,moge_coverage=float(np.mean(coverage)) if coverage else None)
  return result
 before=evaluate('before',0);history=[];frames=info['train_frames']
 for step in range(1,a.steps+1):
  if (step-1)%len(frames)==0:order=torch.randperm(len(frames)).tolist()
  f=frames[order[(step-1)%len(frames)]];degree=min(2,step//1000);prediction,target,depth,alpha,aux=render(f,degree)
  if a.arm.startswith('adaptive'):strategy.step_pre_backward(params,opts,st,step,aux)
  photo=.8*(prediction-target).abs().mean()+.2*ssim_loss(prediction,target);depth_loss=depth.sum()*0
  depth_frame=f
  depth_prediction,depth_alpha=depth,alpha
  if a.arm=='adaptive_allrgb':
   # Retain the original depth-source exposure budget while adding RGB views.
   depth_frame=original_train[(step-1)%len(original_train)]
   if depth_frame in depths and depth_frame!=f:
    _,_,depth_prediction,depth_alpha,_=render(depth_frame,degree)
  if depth_frame in depths:
   d=depths[depth_frame];valid=d['valid']&(depth_alpha.detach()>.1)&(depth_prediction>0)&torch.isfinite(depth_prediction)
   if valid.any():depth_loss=((depth_prediction[valid]-d['depth'][valid]).abs()/d['depth'][valid]).clamp_max(1).mean()
  loss=photo+.1*depth_loss
  for opt in opts.values():opt.zero_grad(set_to_none=True)
  loss.backward()
  for opt in opts.values():opt.step()
  if a.arm.startswith('adaptive'):strategy.step_post_backward(params,opts,st,step,aux,packed=False)
  opts['means'].param_groups[0]['lr']=rates['means']*(.1**(step/a.steps))
  if step%100==0:
   row=dict(step=step,gaussians=len(params['means']),photometric=float(photo),depth_loss=float(depth_loss));history.append(row);print(a.scene,a.arm,row,flush=True)
 after=evaluate('after',min(2,a.steps//1000));q=F.normalize(params['quats'].detach(),dim=-1);qw,qx,qy,qz=q.unbind(-1);normals=torch.stack([2*(qx*qz+qw*qy),2*(qy*qz-qw*qx),1-2*(qx*qx+qy*qy)],-1)
 state=dict(xyz=params['means'].detach().cpu(),quaternions=q.cpu(),normals=F.normalize(normals,dim=-1).cpu(),scales=params['scales'].detach().exp().cpu(),opacities=params['opacities'].detach().sigmoid().clamp(1e-6,1-1e-6).cpu()[:,None])
 payload=dict(base,state_dict=state,source_joint_reconstruction_next66=dict(arm=a.arm,steps=a.steps,source_frames=frames,warm_start=str(old),geometry_trainable_during_semantics=False));torch.save(payload,out/'geometry.pt');torch.save({k:v.detach().cpu() for k,v in params.items()},out/'rgb_parameters.pt')
 report=dict(scene=a.scene,arm=a.arm,steps=a.steps,initial_gaussians=n,final_gaussians=len(q),seconds=time.monotonic()-start,before=before,after=after,history=history,source_frames=frames,validation_frames=info['validation_frames'],depth_reference='calibrated MoGe, not GT',depth_weight=.1,sh_degree=min(2,a.steps//1000),geometry_frozen_before_semantics=True,target_labels_used=False,warm_start=str(old),source_raster_width=512,original_depth_training_frames=original_train,expanded_rgb_budget=a.arm=='adaptive_allrgb')
 (out/'fit.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,a.arm,'done',report['seconds'],flush=True)
if __name__=='__main__':main()
