"""Separate missing surface support from disagreement of the visible rendered surface."""
import argparse,json
from pathlib import Path
import torch
from scipy.spatial import cKDTree
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.audit_geometry_all_source_tracks import stats

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(4)
gp=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'])
inputs={v['role']:v for v in prior.report['geometry_receipt']['inputs']};points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
ap=Path('paper/artifacts')/f'{a.scene}_all_source_track_render_v2_20260916.json';audit=json.loads(ap.read_text());ids=sorted({v['point_id'] for v in audit['rows']});dist,idx=cKDTree(geo.xyz.numpy()).query(torch.stack([points[i] for i in ids]).numpy());nearest={pid:(float(d),int(i)) for pid,d,i in zip(ids,dist,idx)}
rows=[]
for r in audit['rows']:
 cam=prior.cameras[r['frame']];ref=points[r['point_id']];z=float(ref@cam.world_to_camera[2,:3]+cam.world_to_camera[2,3]);distance,index=nearest[r['point_id']]
 rows.append(dict(**r,nearest_surface_metres=distance/prior.scale,nearest_surface_relative=distance/z,nearest_element=index))
groups={}
for split in ['mapping','heldout']:
 for support in [False,True]:
  rr=[r for r in rows if r['split']==split and (r['semantic_support_fraction']>.5)==support];bad=[r for r in rr if r['absrel']>.05]
  groups[f'{split}_semantic_{support}']=dict(count=len(rr),nearest_surface_relative=stats([r['nearest_surface_relative'] for r in rr]),nearest_surface_metres=stats([r['nearest_surface_metres'] for r in rr]),render_absrel=stats([r['absrel'] for r in rr]),bad_render_count=len(bad),bad_with_surface_within_1cm=sum(r['nearest_surface_metres']<.01 for r in bad),bad_with_surface_within_1percent_depth=sum(r['nearest_surface_relative']<.01 for r in bad))
a.output.write_text(json.dumps(dict(schema='surface_support_vs_render.v1',scene=a.scene,source_only=True,benchmark_data_used=False,diagnostic_only=True,geometry_sha256=g['geometry_sha256'],input_sha256=sha256_file(ap),implementation_sha256=sha256_file(__file__),groups=groups,rows=rows,limitations='nearest point uses sparse reference xyz only for diagnosis; no target labels; nearest centre may lie on wrong nearby surface and is not a visibility or covariance test'),indent=2)+'\n');print(a.scene,groups,flush=True)
