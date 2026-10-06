"""Diagnostic only: separate alpha support from observed-native support in fixed maps."""
import json
from pathlib import Path
import numpy as np
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4)
out=Path('output/lerf_support_decomposition_next19');out.mkdir(exist_ok=False)
reg={'diagnostic_only':True,'gt_for_analysis_only':True,'geometry_changed':False,'observed_changed':False,'parameter_search':False,'implementation_sha256':sha256_file(__file__)}
(out/'registration.json').write_text(json.dumps(reg,indent=2)+'\n')
allrows=[];bindings={}
with torch.no_grad():
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  gp=Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json';bindings[str(gp)]=sha256_file(gp);gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
  fp=Path('output/depth_opacity_pipeline_20260918')/scene/'reconstruction/field/bounded_fallback_admission/field.pt';bindings[str(fp)]=sha256_file(fp);p=torch.load(fp,map_location='cpu',weights_only=False);geo.assert_binding(p['geometry_binding']);obs=p['state_dict']['observed'].cuda().float();frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/scene,frames=SCENE_GT_FRAMES[scene]);dst=out/scene;dst.mkdir()
  for frame,objects in frames.items():
   camera=prior.cameras[int(frame.split('_')[-1])];h,w=objects[0].mask.shape;k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
   rendered,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=obs[:,None],viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   a=alpha[0,...,0].cpu().numpy();mass=rendered[0,...,0].cpu().numpy();frac=mass/np.maximum(a,1e-8);valid=(a>1e-8)&(frac>.5);torch.save({'alpha':torch.from_numpy(a),'observed_fraction':torch.from_numpy(frac),'valid':torch.from_numpy(valid)},dst/(frame+'.pt'))
   for obj in objects:
    mask=obj.mask;row={'scene':scene,'frame':frame,'query':obj.query,'positive_pixels':int(mask.sum()),'alpha_nonzero_recall':float((a[mask]>1e-8).mean()),'alpha_ge_05_recall_diagnostic':float((a[mask]>=.5).mean()),'valid_semantic_recall':float(valid[mask].mean()),'geometry_absent_fraction':float((a[mask]<=1e-8).mean()),'geometry_present_semantics_rejected_fraction':float(((a[mask]>1e-8)&(frac[mask]<=.5)).mean()),'alpha_median':float(np.median(a[mask])),'observed_fraction_median':float(np.median(frac[mask]))};allrows.append(row)
   print(scene,frame,'done',flush=True)
  del geo,p,obs;torch.cuda.empty_cache()
report={'count':len(allrows),'rows':allrows,'source_bindings':bindings,'registration_sha256':sha256_file(out/'registration.json'),'interpretation':'Nonzero alpha does not certify correct geometry; observed fraction uses the frozen deployed .5 validity gate. GT only diagnoses which failure matters, never admits new elements.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('complete',len(allrows))
