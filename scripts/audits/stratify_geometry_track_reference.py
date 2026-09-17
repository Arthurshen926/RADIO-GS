"""Separate low-parallax reference uncertainty from geometry disagreement."""
import argparse,json,math
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.audit_geometry_all_source_tracks import stats

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(4);r=json.loads(a.input.read_text());scene=r['scene']
gp=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')/scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
inputs={v['role']:v for v in prior.report['geometry_receipt']['inputs']};points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
needed={row['point_id'] for row in r['rows']};directions={pid:[] for pid in needed}
for f in prior.records:
 c=prior.cameras[f]
 for pid in c.point_ids.tolist():
  if pid in needed:directions[pid].append(torch.nn.functional.normalize(points[pid]-c.camera_to_world[:3,3],dim=0))
angles={}
for pid,ds in directions.items():
 if len(ds)<2:angles[pid]=None;continue
 d=torch.stack(ds);angles[pid]=math.degrees(math.acos(float((d@d.T).min().clamp(-1,1))))
groups={}
for split in ['mapping','heldout']:
 for threshold in [0,1,5]:
  for support in [False,True]:
   rows=[v for v in r['rows'] if v['split']==split and (v['semantic_support_fraction']>.5)==support and angles[v['point_id']] is not None and angles[v['point_id']]>=threshold]
   matched=[v for v in rows if v['native_absrel'] is not None]
   groups[f'{split}_angle_ge_{threshold}_semantic_{support}']=dict(unique_points=len({v['point_id'] for v in rows}),gaussian=stats([v['absrel'] for v in rows]),matched_gaussian=stats([v['absrel'] for v in matched]),matched_moge=stats([v['native_absrel'] for v in matched]))
a.output.write_text(json.dumps(dict(schema='source_track_parallax_strata.v1',scene=scene,source_only=True,benchmark_data_used=False,input_sha256=sha256_file(a.input),implementation_sha256=sha256_file(__file__),groups=groups,angles=angles,limitations='maximum baseline angle uses all source-authorized observations of existing COLMAP tracks; angle strata are diagnostic, not a new calibrated geometry or manual GT; shared-source reference uncertainty remains'),indent=2)+'\n')
print(scene,{k:v for k,v in groups.items() if k.startswith('heldout') and k.endswith('True')},flush=True)
