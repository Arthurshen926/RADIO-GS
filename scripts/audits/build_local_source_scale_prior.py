"""Materialize bounded local geometry with fit/admission/audit point separation."""
import argparse,json,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points,_bilinear
from radio_gs.geometry.local_depth_calibration import local_depth_scales,warp_normals_for_image_scale
from radio_gs.geometry.ray_alignment import align_normals_to_intrinsics
from radio_gs.geometry.independent_view_scale import relative_error_stats
from radio_gs.v4.contracts.geometry_receipt import HashedInput
from radio_gs.utils.immutable_artifacts import sha256_file
from dataclasses import asdict


def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--maximum-ratio',type=float,choices=[1.2,2.],default=1.2);p.add_argument('--width',type=int,default=320);a=p.parse_args();torch.set_num_threads(4);a.output_root.mkdir(parents=True,exist_ok=False)
 prior=MogeGeometryPrior(a.prior,sha256_file(a.prior));inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']}
 points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
 result=copy.deepcopy(prior.report);calibration=[];accepted_count=0
 for frame,rec in sorted(prior.records.items()):
  row=copy.deepcopy(prior.calibration[frame]);cam=prior.cameras[frame];fit=row['source_scale_fit'];factor=fit['scale']
  if factor is None:calibration.append(row);continue
  assert sha256_file(rec['prediction'])==rec['prediction_sha256'];blob=torch.load(rec['prediction'],weights_only=False)
  indices=[i for i,pid in enumerate(cam.point_ids.tolist()) if pid in points];ids=cam.point_ids[indices];xy=cam.xy[indices]
  xyz=torch.stack([points[int(pid)] for pid in ids]);reference=(xyz@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3])[:,2]
  raw=blob['point_map'][...,2].float();sample=_bilinear(raw,xy,cam.width,cam.height,pixel_contract='image_edge_v2')*factor
  legal=(_bilinear(blob['validity'].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2')>.999)&torch.isfinite(sample)&torch.isfinite(reference)&(sample>0)&(reference>0)
  train=legal&(ids%5!=0);gate=legal&(ids%10==0);audit=legal&(ids%10==5)
  w=a.width;h=round(raw.shape[0]*w/raw.shape[1]);depth=F.interpolate(raw[None,None],size=(h,w),mode='nearest-exact')[0,0].to(a.device)*factor
  valid=F.interpolate(blob['validity'].float()[None,None],size=(h,w),mode='nearest-exact')[0,0].to(a.device)>.5
  normals=F.interpolate(blob['normals'].float().permute(2,0,1)[None],size=(h,w),mode='nearest-exact')[0].permute(1,2,0).to(a.device)
  nk=cam.intrinsic.clone();nk[0]/=cam.width;nk[1]/=cam.height
  if blob.get('intrinsics') is not None:normals=align_normals_to_intrinsics(normals,blob['intrinsics'].float().to(a.device),nk.to(a.device))
  valid=valid&torch.isfinite(depth)&(depth>0)&torch.isfinite(normals).all(-1)&(normals.norm(dim=-1)>1e-6)
  depth=torch.where(valid,depth,torch.ones_like(depth));normals=torch.where(valid[...,None],normals,torch.tensor([0.,0.,1.],device=a.device))
  yy,xx=torch.meshgrid(torch.arange(h,device=a.device),torch.arange(w,device=a.device),indexing='ij');uv=torch.stack([(xx.flatten()+.5)/w,(yy.flatten()+.5)/h],1)
  scale,supported=local_depth_scales((xy[train]/torch.tensor([cam.width,cam.height])).to(a.device),sample[train].to(a.device),reference[train].to(a.device),uv,depth.flatten(),fallback_scale=1.,maximum_ratio=a.maximum_ratio)
  scale=scale.reshape(h,w).float();supported=supported.reshape(h,w);k=cam.intrinsic.clone();k[0]*=w/cam.width;k[1]*=h/cam.height;k=k.to(a.device)
  rays=torch.stack([xx+.5,yy+.5,torch.ones_like(xx)],-1).float()@torch.linalg.inv(k).T
  normals=warp_normals_for_image_scale(rays*depth[...,None],normals,scale,k);depth=depth*scale;valid=valid&supported&torch.isfinite(depth)&(depth>0)
  corrected=_bilinear(depth.cpu(),xy,cam.width,cam.height,pixel_contract='image_edge_v2');coverage=_bilinear(valid.float().cpu(),xy,cam.width,cam.height,pixel_contract='image_edge_v2')>.999
  error=(corrected-reference).abs()/reference.clamp_min(1e-12);gs=relative_error_stats(error[gate]);audit_stats=relative_error_stats(error[audit]);gate_coverage=float(coverage[gate].float().mean()) if gate.any() else 0.
  accepted=gs['count']>=16 and gate_coverage>=.9 and gs['median']<=.05 and gs['p90']<=.15
  path=a.output_root/f'frame_{frame:05d}.pt';torch.save(dict(schema='source_local_scale_geometry.v1',frame=frame,prediction_sha256=rec['prediction_sha256'],source_only=True,benchmark_data_used=False,depth=depth.cpu(),normals=normals.cpu(),valid=valid.cpu(),intrinsic=k.cpu(),world_to_camera=cam.world_to_camera,scale_map=scale.cpu(),local_supported=supported.cpu(),fit_point_partition='ID modulo5 !=0',source_scale_prior_sha256=sha256_file(a.prior)),path)
  row.update(accepted=accepted,rejection_reason=None if accepted else 'local_admission_depth_or_coverage_failure',source_geometry_override=dict(path=str(path.resolve()),sha256=sha256_file(path)),local_scale_fit=dict(gate=gs,audit=audit_stats,gate_coverage=gate_coverage,audit_coverage=float(coverage[audit].float().mean()) if audit.any() else None,dense_supported_fraction=float(valid.float().mean()),materialization_hw=[h,w],actual_materialized_depth_used_for_gate=True))
  calibration.append(row);accepted_count+=accepted;print(frame,accepted,gs,gate_coverage,flush=True)
 result.update(schema='radio_gs.local_source_scale_prior.v1',calibration=calibration,accepted_view_count=accepted_count,rejected_view_count=len(calibration)-accepted_count,parent_prior_sha256=sha256_file(a.prior),implementation_sha256=sha256_file(__file__),local_implementation_sha256=sha256_file('radio_gs/geometry/local_depth_calibration.py'))
 result['geometry_receipt']['carrier']='materialized_source_local_scale_depth_prior';result['geometry_receipt']['inputs'].append(asdict(HashedInput.seal('independent_scale_prior',a.prior)));result['geometry_receipt']['calibration'].update(contract='source_local_point_id_scale_v1',local_scale_bounds=[1/a.maximum_ratio,a.maximum_ratio],local_minimum_gate_coverage=.9,materialization_width=a.width,normal_transport='image-scale inverse-transpose Jacobian',unknown_pixels='excluded when fewer than four local anchors; no invented scale confidence')
 (a.output_root/'source_scale_prior.json').write_text(json.dumps(result,indent=2)+'\n');print('accepted',accepted_count,'of',len(calibration),flush=True)
if __name__=='__main__':main()
