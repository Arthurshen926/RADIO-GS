"""Geometry-only source consensus filtering with frozen parent controls.
Candidate is a new geometry bundle; old semantic fields are incompatible.
"""
import argparse,json,math
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views, build_geometry
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.audit_geometry_all_source_tracks import stats


def run():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 gp=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
 all_records=prior.records;prior.records={f:all_records[f] for f in gr['mapping_frames']}
 views=load_views(prior,gr['width'],a.device,allow_global_scale_fallback=True)
 recovered={r['frame']:r for r in gr['recovered_scales'] if r['accepted']}
 for v in views:
  if not v['local_calibration_accepted']:
   if v['frame'] not in recovered:raise ValueError('mapping view lacks accepted recovery')
   v['depth']*=recovered[v['frame']]['scale']
 prior.records=all_records
 n=len(geo.xyz)
 trusted=[v for v in views if v['local_calibration_accepted']]
 candidate,_=build_geometry(trusted,torch.ones(len(trusted),device=a.device),prior.scale,gr['voxel_metres'],a.device,gr['minimum_voxel_views'])
 cp=a.output_dir/'geometry.pt'
 torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',state_dict={k:v.cpu() for k,v in candidate.state_dict().items()},geometry_trainable_during_semantics=False,parent_geometry_sha256=g['geometry_sha256'],mapping_frames=[v['frame'] for v in trusted],source_authority_sha256=prior.source_authority_sha256),cp)
 auditpath=Path('paper/artifacts')/f'{a.scene}_all_source_track_render_v2_20260916.json';baseline=json.loads(auditpath.read_text());cohort={(r['frame'],r['point_id']) for r in baseline['rows'] if r['split']=='heldout' and r['semantic_support_fraction']>.5}
 inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']};points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
 rows=[]
 for frame in gr['heldout_frames']:
  cam=prior.cameras[frame];w=128;h=round(cam.height*w/cam.width);k=cam.intrinsic.to(a.device).clone();k[0]*=w/cam.width;k[1]*=h/cam.height;pose=cam.world_to_camera.to(a.device)
  predictions={}
  for name,model in [('parent',geo),('trusted_only',candidate)]:
   xyz=model.xyz;z=(xyz@pose[:3,:3].T+pose[:3,3])[:,2]
   with torch.no_grad():im,alpha,_=rasterization(means=xyz,quats=model.quaternions,scales=model.scales,opacities=model.opacities.flatten(),colors=z[:,None],viewmats=pose[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   predictions[name]=((im[0,:,:,0]/alpha[0,:,:,0].clamp_min(1e-8)).cpu(),alpha[0,:,:,0].cpu())
  for pid,xy in zip(cam.point_ids.tolist(),cam.xy):
   if pid not in points:continue
   uv=xy*torch.tensor([w/cam.width,h/cam.height]);x,y=uv.floor().long().tolist();ref=float(points[pid]@cam.world_to_camera[2,:3]+cam.world_to_camera[2,3])
   if not (0<=x<w and 0<=y<h) or ref<=0:continue
   row=dict(frame=frame,point_id=pid,baseline_semantic_supported=(frame,pid) in cohort)
   for name,(depth,alpha) in predictions.items():row[name]=dict(covered=bool(alpha[y,x]>=.5),absrel=abs(float(depth[y,x])-ref)/ref)
   rows.append(row)
 groups={}
 for subset in ['all','baseline_semantic_supported']:
  rr=rows if subset=='all' else [r for r in rows if r['baseline_semantic_supported']]
  both=[r for r in rr if r['parent']['covered'] and r['trusted_only']['covered']]
  groups[subset]={}
  for name in ['parent','trusted_only']:
   covered=[r[name]['absrel'] for r in rr if r[name]['covered']]
   groups[subset][name]=dict(total=len(rr),covered=len(covered),coverage=len(covered)/len(rr),absrel=stats(covered),common_support_absrel=stats([r[name]['absrel'] for r in both]),accurate_5percent_coverage=sum(r[name]['covered'] and r[name]['absrel']<=.05 for r in rr)/len(rr),capped_error_missing_is_one=sum(min(r[name]['absrel'],1.) if r[name]['covered'] else 1. for r in rr)/len(rr))
 result=dict(schema='trusted_source_geometry_ablation.v1',scene=a.scene,source_only=True,benchmark_data_used=False,semantic_used_only_for_fixed_evaluation_stratum=True,parent_geometry_sha256=g['geometry_sha256'],candidate_geometry_sha256=sha256_file(cp),candidate_geometry=str(cp.resolve()),mapping_frames=[v['frame'] for v in trusted],heldout_frames=gr['heldout_frames'],parent_elements=n,candidate_elements=len(candidate.xyz),groups=groups,rows=rows,implementation_sha256=sha256_file(__file__),fusion_implementation_sha256=sha256_file('radio_gs/v4/geometry/tsdf_fusion.py'),geometry_report_sha256=sha256_file(gp),baseline_audit_sha256=sha256_file(auditpath),source_geometry_receipt_inputs=prior.report['geometry_receipt']['inputs'],promoted=False,limitations='geometry source ablation only; heldout reference was not excluded from old calibration; source coverage differs; no old semantic field reuse allowed')
 (a.output_dir/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,'trusted views',len(trusted),'elements',n,len(candidate.xyz),'groups',groups,flush=True)

if __name__=='__main__':run()
