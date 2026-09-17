"""Bounded local scale diagnostic on held-out source track IDs, no field/GT."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points,_bilinear
from radio_gs.geometry.local_depth_calibration import local_depth_scales
from radio_gs.geometry.independent_view_scale import relative_error_stats
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
 prior=MogeGeometryPrior(a.prior,sha256_file(a.prior));inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']}
 points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
 rows=[];all_errors={k:[] for k in ['scale','local','local_supported']};counts=0;covered=0
 for frame,rec in sorted(prior.records.items()):
  cam=prior.cameras[frame];indices=[i for i,pid in enumerate(cam.point_ids.tolist()) if pid in points]
  if not indices:continue
  ids=cam.point_ids[indices];xy=cam.xy[indices];xyz=torch.stack([points[int(pid)] for pid in ids]);r=(xyz@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3])[:,2]
  assert sha256_file(rec['prediction'])==rec['prediction_sha256'];blob=torch.load(rec['prediction'],weights_only=False)
  d=_bilinear(blob['point_map'][...,2].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2');valid=_bilinear(blob['validity'].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2')>.999
  valid &= torch.isfinite(d)&torch.isfinite(r)&(d>0)&(r>0)
  train=valid&(ids%5!=0);test=valid&(ids%5==0)
  if train.sum()<32 or test.sum()<16:continue
  factor=float((r[train]/d[train]).median());uv=xy/torch.tensor([cam.width,cam.height])
  local,supported=local_depth_scales(uv[train].to(a.device),d[train].to(a.device),r[train].to(a.device),uv[test].to(a.device),d[test].to(a.device),fallback_scale=factor)
  local=local.cpu();supported=supported.cpu();global_error=(factor*d[test]-r[test]).abs()/r[test];error=(local*d[test]-r[test]).abs()/r[test]
  groups=dict(scale=global_error,local=error,local_supported=error[supported]);row=dict(frame=frame,heldout_points=int(test.sum()),local_support_fraction=float(supported.float().mean()),metrics={k:relative_error_stats(v) for k,v in groups.items()})
  for k,v in groups.items():all_errors[k].append(v)
  counts+=len(error);covered+=int(supported.sum());rows.append(row);print(frame,row['metrics']['local']['median'],row['local_support_fraction'],flush=True)
 out=dict(schema='local_source_scale_diagnostic.v1',prior_sha256=sha256_file(a.prior),source_only=True,benchmark_data_used=False,fit_point_split='id modulo5 !=0',evaluation_point_split='id modulo5 ==0',local_policy=dict(neighbors=16,minimum_neighbors=4,spatial_radius=.1,log_depth_radius=.2,maximum_ratio=1.2),aggregate={k:relative_error_stats(torch.cat(v)) for k,v in all_errors.items()},local_support_fraction=covered/counts,rows=rows,limitations='Prior bounded local policy, sparse source reference only; held-out pixels not independent dense truth; no geometry or field built.',implementation_sha256=sha256_file(__file__),local_implementation_sha256=sha256_file('radio_gs/geometry/local_depth_calibration.py'))
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(out['aggregate'],flush=True)
if __name__=='__main__':main()
