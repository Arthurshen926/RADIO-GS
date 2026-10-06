"""GT-only attribution of selected-mask false positives, never a selection policy.

Projected-center membership is not physical 3D identity. Contributions use the
same selected-only compositing as the frozen evaluator.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,_load_frame_gt_masks,load_binary_mask
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/lerf_selected_ladder_20260919')/a.scene;out=root/'occlusion_attribution.json';assert not out.exists()
rep=json.loads((root/'report.json').read_text());assert sha256_file(root/'selections.pt')==rep['selection_sha256'];selected=torch.load(root/'selections.pt',weights_only=False)['final'].cuda();data=torch.load(Path('output/lerf_semantic_ladder_calibrated_20260919')/a.scene/'scores.pt',weights_only=False)
gr=json.loads((Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json').read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(data['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);rows=[]
with torch.no_grad():
 for frame in SCENE_GT_FRAMES[a.scene]:
  gtroot=Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/a.scene/'gt'
  for obj in _load_frame_gt_masks(gtroot,frame,threshold=10):
   target=torch.from_numpy(obj.mask).cuda();h,w=target.shape;camera=prior.cameras[int(frame.split('_')[-1])];k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height;ids=selected[:,data['texts'].index(obj.query)];xyz=geo.xyz[ids]
   if not len(xyz):continue
   cam=xyz@camera.world_to_camera[:3,:3].cuda().T+camera.world_to_camera[:3,3].cuda();uv=cam@k.T;ij=(uv[:,:2]/uv[:,2:].clamp_min(1e-8)).floor().long();valid=(cam[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h);inside=torch.zeros(len(xyz),device='cuda:0',dtype=torch.bool);inside[valid]=target[ij[valid,1],ij[valid,0]]
   colors=torch.stack([inside.float(),(~inside).float()],1)
   render,alpha,_=rasterization(means=xyz,quats=geo.quaternions[ids],scales=geo.scales[ids],opacities=geo.opacities.flatten()[ids],colors=colors,viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   pred=(alpha[0,...,0]*255+.5).floor()>10;reference=torch.from_numpy(load_binary_mask(root/'final'/(frame+'_'+obj.query+'.png'),grayscale=True)).cuda();assert torch.equal(pred,reference)
   full_render,_,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=ids.float()[:,None],viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   visible=(full_render[0,...,0]*255+.5).floor()>10
   fp=pred&~target;mass=render[0][fp].sum(0);support=render[0,...,0]*255+.5
   rows.append(dict(frame=frame,query=obj.query,selected_elements=len(xyz),fp_exposed_by_removing_nonselected=int((fp&~visible).sum()),full_scene_selected_contribution_iou=float((visible&target).sum()/(visible|target).sum().clamp_min(1)),selected_only_iou=float((pred&target).sum()/(pred|target).sum().clamp_min(1)),centers_project_inside=int(inside.sum()),false_positive_pixels=int(fp.sum()),false_positive_alpha_mass=mass.cpu().tolist(),fp_pixels_with_inside_center_contribution_above_threshold=int((fp&(support.floor()>10)).sum()),predicted_pixels=int(pred.sum()),target_pixels=int(target.sum())))
  print(a.scene,frame,flush=True)
 mass=np.array([r['false_positive_alpha_mass'] for r in rows]).sum(0);count=sum(r['false_positive_pixels'] for r in rows)
 result=dict(diagnostic_only=True,uses_target_gt=True,deployable=False,physical_identity_inferred=False,rule='Per-view projected center inside GT vs outside; alpha contribution decomposition of actual selected-only render; not an additive IoU attribution',observations=rows,fp_pixel_fraction_exposed_by_removing_nonselected=sum(r['fp_exposed_by_removing_nonselected'] for r in rows)/max(count,1),full_scene_selected_contribution_miou=float(np.mean([r['full_scene_selected_contribution_iou'] for r in rows])),selected_only_miou=float(np.mean([r['selected_only_iou'] for r in rows])),fp_alpha_fraction_from_centers_inside=float(mass[0]/max(mass.sum(),1e-8)),fp_pixel_fraction_with_inside_contribution_above_threshold=sum(r['fp_pixels_with_inside_center_contribution_above_threshold'] for r in rows)/max(count,1),implementation_sha256=sha256_file(__file__))
 out.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,{k:v for k,v in result.items() if k!='observations'},flush=True)
