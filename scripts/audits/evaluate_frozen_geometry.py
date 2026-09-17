"""Reopen saved geometry in a fresh process and evaluate source-only geometry."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views,evaluate_geometry
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);device=torch.device('cuda:0')
report=json.loads((a.root/a.scene/'report.json').read_text())
fusion=Path('/mnt/pool/sqy/results/RADIO-GS/output/v4_fusion_contract_repair_20260914')/(a.scene+'.json')
prior=MogeGeometryPrior(report.get('source_report',str(fusion)),report['source_report_sha256'])
views=load_views(prior,report['width'],device,report.get('allow_global_scale_fallback',False))
ids={r['frame'] for r in report['results']['independent_scale']['views']}
views=[v for v in views if v['frame'] in ids]
inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']}
points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{v.image_id for f,v in prior.cameras.items() if f in prior.records},maximum_reprojection_error=2.,minimum_track_views=3)
meta=json.loads(Path(inputs['source_authority']['path']).read_text())['metadata']
results={}
for name in ('historical_gaussians','independent_scale','joint_scale'):
 if name=='historical_gaussians':
  model,*rest=load_render_pipeline(meta['config'],meta['checkpoint'],device,strict_checkpoint_contract=True,load_ply_rgb_features=False);del rest
 else:
  record=report['results'][name]
  model=FrozenGaussianGeometry.load(record['geometry'],expected_sha256=record['geometry_sha256'],device=device)
  binding=model.fingerprint();model.assert_binding(binding)
 results[name]=evaluate_geometry(model,views,prior.scale,prior,points)
 if name!='historical_gaussians':results[name]['full_geometry_binding']=binding
 del model;torch.cuda.empty_cache()
print(a.scene,{k:v['aggregate'] for k,v in results.items()},flush=True)
a.output.write_text(json.dumps({'scene':a.scene,'source_only':True,'colmap_quality_filter':{'maximum_reprojection_error_pixels':2.,'minimum_track_views':3},'cold_geometry_load':True,'results':results},indent=2)+'\n')
