"""Geometry-only source consensus filtering with frozen parent controls.
Candidate is a new geometry bundle; old semantic fields are incompatible.
"""
import argparse,json,math
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views
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
 n=len(geo.xyz);agree=torch.zeros(n,device=a.device,dtype=torch.int32);conflict=torch.zeros_like(agree);old_agree=torch.zeros_like(agree);old_conflict=torch.zeros_like(agree)
 for v in views:
  pose=torch.linalg.inv(v['pose']);xyz=geo.xyz@pose[:3,:3].T+pose[:3,3];z=xyz[:,2];uv=xyz@v['k'].T
  x=(uv[:,0]/z.clamp_min(1e-8)).floor().long();y=(uv[:,1]/z.clamp_min(1e-8)).floor().long();h,w=v['depth'].shape;inside=(z>0)&(x>=0)&(x<w)&(y>=0)&(y<h);ids=inside.nonzero().flatten();xx=x[ids];yy=y[ids];valid=v['valid'][yy,xx];d=v['depth'][yy,xx];res=z[ids]-d
  # Predeclared tighter geometric support, not selected against benchmark labels.
  tol=.5*gr['voxel_metres']*prior.scale+.005*d
  old_tol=math.sqrt(3)*gr['voxel_metres']*prior.scale+.01*d
  agree[ids]+=(valid&(res.abs()<=tol)).int();conflict[ids]+=(valid&(res < -tol)).int()
  old_agree[ids]+=(valid&(res.abs()<=old_tol)).int();old_conflict[ids]+=(valid&(res < -old_tol)).int()
 keep=(agree>=2)&(agree>conflict)
 old_keep=(old_agree>=2)&(old_agree>old_conflict)
 if not bool(keep.any()):raise ValueError('candidate removed all geometry')
 candidate={k:v[keep].cpu() for k,v in geo.state_dict().items()}
 cp=a.output_dir/'geometry.pt';torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',state_dict=candidate,geometry_trainable_during_semantics=False,parent_geometry_sha256=g['geometry_sha256'],mapping_frames=gr['mapping_frames'],source_authority_sha256=prior.source_authority_sha256),cp)
 evidence=agree+conflict
 confidence=torch.where(evidence>0,agree.float()/evidence.clamp_min(1),torch.ones_like(evidence,dtype=torch.float32))
 soft_opacity=(geo.opacities*confidence[:,None]).clamp_min(1e-5)
 soft_path=a.output_dir/'soft_geometry.pt'
 soft_state={k:v.cpu() for k,v in geo.state_dict().items()};soft_state['opacities']=soft_opacity.cpu()
 torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',state_dict=soft_state,geometry_trainable_during_semantics=False,parent_geometry_sha256=g['geometry_sha256'],mapping_frames=gr['mapping_frames'],source_authority_sha256=prior.source_authority_sha256),soft_path)
 torch.save(dict(keep=keep.cpu(),agreement=agree.cpu(),free_space_conflict=conflict.cpu(),parent_geometry_sha256=g['geometry_sha256']),a.output_dir/'selection.pt')
 auditpath=Path('paper/artifacts')/f'{a.scene}_all_source_track_render_v2_20260916.json';baseline=json.loads(auditpath.read_text());cohort={(r['frame'],r['point_id']) for r in baseline['rows'] if r['split']=='heldout' and r['semantic_support_fraction']>.5}
 inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']};points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
 rows=[]
 for frame in gr['heldout_frames']:
  cam=prior.cameras[frame];w=128;h=round(cam.height*w/cam.width);k=cam.intrinsic.to(a.device).clone();k[0]*=w/cam.width;k[1]*=h/cam.height;pose=cam.world_to_camera.to(a.device)
  predictions={}
  for name,mask in [('parent',torch.ones(n,device=a.device,dtype=torch.bool)),('tight_consensus',keep),('soft_consensus',torch.ones(n,device=a.device,dtype=torch.bool))]:
   xyz=geo.xyz[mask];z=(xyz@pose[:3,:3].T+pose[:3,3])[:,2]
   with torch.no_grad():im,alpha,_=rasterization(means=xyz,quats=geo.quaternions[mask],scales=geo.scales[mask],opacities=(soft_opacity if name=='soft_consensus' else geo.opacities)[mask].flatten(),colors=z[:,None],viewmats=pose[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   predictions[name]=((im[0,:, :,0]/alpha[0,:,:,0].clamp_min(1e-8)).cpu(),alpha[0,:,:,0].cpu())
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
  both=[r for r in rr if r['parent']['covered'] and r['tight_consensus']['covered'] and r['soft_consensus']['covered']]
  groups[subset]={}
  for name in ['parent','tight_consensus','soft_consensus']:
   covered=[r[name]['absrel'] for r in rr if r[name]['covered']]
   groups[subset][name]=dict(total=len(rr),covered=len(covered),coverage=len(covered)/len(rr),absrel=stats(covered),common_support_absrel=stats([r[name]['absrel'] for r in both]),accurate_5percent_coverage=sum(r[name]['covered'] and r[name]['absrel']<=.05 for r in rr)/len(rr),capped_error_missing_is_one=sum(min(r[name]['absrel'],1.) if r[name]['covered'] else 1. for r in rr)/len(rr))
 result=dict(schema='tight_source_consensus_geometry.v1',scene=a.scene,source_only=True,benchmark_data_used=False,semantic_used_only_for_fixed_evaluation_stratum=True,parent_geometry_sha256=g['geometry_sha256'],candidate_geometry_sha256=sha256_file(cp),candidate_geometry=str(cp.resolve()),soft_geometry=str(soft_path.resolve()),soft_geometry_sha256=sha256_file(soft_path),mapping_frames=gr['mapping_frames'],heldout_frames=gr['heldout_frames'],parent_elements=n,candidate_elements=int(keep.sum()),parent_consensus_reproduced_fraction=float(old_keep.float().mean()),policy=dict(absolute_voxel_fraction=.5,relative_depth_tolerance=.005,minimum_agreement=2,agreement_must_exceed_conflict=True),groups=groups,rows=rows,implementation_sha256=sha256_file(__file__),geometry_report_sha256=sha256_file(gp),baseline_audit_sha256=sha256_file(auditpath),source_geometry_receipt_inputs=prior.report['geometry_receipt']['inputs'],mapping_predictions=[dict(frame=f,sha256=all_records[f]['prediction_sha256']) for f in gr['mapping_frames']],promoted=False,limitations='sparse-track geometry development control; new geometry requires complete cache/field rebuild; no GT masks or semantic features in filtering; coverage and common-support errors reported; old calibration may use sparse heldout observations')
 (a.output_dir/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,'elements',n,int(keep.sum()),'reproduction',result['parent_consensus_reproduced_fraction'],'groups',groups,flush=True)

if __name__=='__main__':run()
