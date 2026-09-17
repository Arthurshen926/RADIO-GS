"""Check whether remaining local depth tails coincide with the fixed scale cap."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points,_bilinear
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
path=Path('/mnt/pool/sqy/results/RADIO-GS/output/nvos_local_scale_repair_20260917')/a.scene/'source_scale_prior.json';prior=MogeGeometryPrior(path,sha256_file(path));inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']};points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
rows=[]
for frame,cal in prior.calibration.items():
 if 'source_geometry_override' not in cal:continue
 r=cal['source_geometry_override'];assert sha256_file(r['path'])==r['sha256'];d=torch.load(r['path'],weights_only=False);cam=prior.cameras[frame]
 ids=[i for i,pid in enumerate(cam.point_ids.tolist()) if pid%10==5 and pid in points]
 if not ids:continue
 xy=cam.xy[ids];xyz=torch.stack([points[int(pid)] for pid in cam.point_ids[ids]]);z=xyz@cam.world_to_camera[2,:3]+cam.world_to_camera[2,3]
 depth=_bilinear(d['depth'],xy,cam.width,cam.height,pixel_contract='image_edge_v2');scale=_bilinear(d['scale_map'],xy,cam.width,cam.height,pixel_contract='image_edge_v2');valid=(_bilinear(d['valid'].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2')>.999)&(z>0)
 error=(depth-z)/z;tail=valid&(error.abs()>.15);bound=valid&(((scale<=1/1.2+.001)&(error>0))|((scale>=1.2-.001)&(error<0)))
 rows.append(dict(frame=frame,audit_points=len(ids),valid=int(valid.sum()),tail=int(tail.sum()),tail_with_correction_blocked_at_bound=int((tail&bound).sum())))
totals={k:sum(r[k] for r in rows) for k in ['audit_points','valid','tail','tail_with_correction_blocked_at_bound']};totals['saturated_tail_fraction']=totals['tail_with_correction_blocked_at_bound']/max(1,totals['tail'])
a.output.write_text(json.dumps(dict(schema='local_scale_saturation_audit.v1',scene=a.scene,source_prior_sha256=sha256_file(path),source_only=True,benchmark_data_used=False,totals=totals,rows=rows,interpretation='Saturation with remaining error of the correction direction is a diagnostic, not proof that relaxing the cap yields accurate geometry.',implementation_sha256=sha256_file(__file__)),indent=2)+'\n');print(a.scene,totals)
