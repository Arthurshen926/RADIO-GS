"""Target-label-only diagnostic of attainable carrier detail, NEVER a method score.

Fits a disposable scalar on each fixed carrier using target GT. Saves a raster
and diagnostic receipt only, never an element table or deployable field.
"""
import argparse,json,os,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=160);p.add_argument('--tangent-scale',type=float,choices=[1.,.5],default=1.);a=p.parse_args();torch.set_num_threads(4)
assert not a.output.exists();a.output.mkdir(parents=True)
evalpath=a.root/'evaluation'/f'{a.scene}_signed/report.json';evaluation=json.loads(evalpath.read_text());assert evaluation['target_gt_used_after_prediction_only']
manifest=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json');m=json.loads(manifest.read_text());assert sha256_file(manifest)==evaluation['manifest_sha256'];s=next(s for s in m['scenes'] if s['scene_id']==a.scene);target=next(f for f in s['frames'] if f['frame_id']==evaluation['target_frame'])
forbidden={str(Path(target[k]).resolve()) for k in ['rgb_path','annotation_rgb_path']}
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise RuntimeError('No target RGB in carrier diagnostic')
sys.addaudithook(guard)
gp=a.root/'frozen_geometry/report.json';gr=json.loads(gp.read_text());r=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(r['geometry'],expected_sha256=r['geometry_sha256'],device=a.device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);camera=prior.cameras[int(target['canonical_index'])]
run='image_supervised_native_residual';fp=a.root/'field'/run/'field.pt';assert sha256_file(fp)==evaluation['fields'][run];payload=torch.load(fp,map_location='cpu',weights_only=False);observed=payload['state_dict']['observed'].to(a.device);binding=payload['geometry_binding'];del payload
w,h=evaluation['grid_wh'];k=camera.intrinsic.to(a.device).clone();k[0]*=w/camera.width;k[1]*=h/camera.height
path=Path(target['gt_mask_path']);assert sha256_file(path)==target['ground_truth_sha256'];truth=np.asarray(Image.open(path));truth=np.any(truth>127,axis=2) if truth.ndim==3 else truth>127
assert m['protocol']['resize']=='nearest';index=cv2.resize(np.arange(h*w,dtype=np.float32).reshape(h,w),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int64);index=torch.from_numpy(index).to(a.device);gt=torch.from_numpy(truth).to(a.device)
render_scales=geo.scales.clone();render_scales[:,:2]*=a.tangent_scale
def render(values):
 colors=torch.stack([values*observed,observed.float()],1)
 image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=render_scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
 mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
 response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0)
 return response,valid
values=torch.nn.Parameter(torch.full((len(observed),),.5,device=a.device));optimizer=torch.optim.Adam([values],lr=.1);best=(-1,None);history=[]
for step in range(a.steps+1):
 response,valid=render(values);score=response.flatten()[index];binary=score>=.5;iou=float((binary&gt).sum()/(binary|gt).sum().clamp_min(1))
 if iou>best[0]:best=(iou,response.detach().cpu().clone());beststep=step
 # Fixed raster and membership values -> convex binary cross entropy in values.
 loss=-.5*(score[gt].clamp_min(1e-6).log().mean()+(1-score[~gt]).clamp_min(1e-6).log().mean())
 if step%20==0 or step==a.steps:history.append(dict(step=step,iou=iou,loss=float(loss)));print(history[-1],flush=True)
 if step==a.steps:break
 optimizer.zero_grad();loss.backward();optimizer.step()
 with torch.no_grad():values.clamp_(0,1)
geo.assert_binding(binding)
torch.save(dict(response=best[1],threshold=.5,diagnostic_target_gt_fitted=True,deployable=False),a.output/'oracle_raster.pt')
report=dict(schema='target_carrier_attainability_diagnostic.v1',scene=a.scene,method_score=False,deployable=False,target_gt_used_for_optimization=True,target_rgb_opened=False,field_updated=False,source_geometry_updated=False,render_tangent_scale_factor=a.tangent_scale,scalar_table_saved=False,observed_support_fixed=True,geometry_sha256=r['geometry_sha256'],field_sha256=sha256_file(fp),evaluation_sha256=sha256_file(evalpath),gt_sha256=sha256_file(path),steps=a.steps,history=history,best_step=beststep,attained_iou=best[0],actual_iou=evaluation['results'][run]['iou'],global_optimum_certified=False,output_sha256=sha256_file(a.output/'oracle_raster.pt'),implementation_sha256=sha256_file(__file__))
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
