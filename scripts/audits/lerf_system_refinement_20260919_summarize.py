import json,hashlib
from pathlib import Path
import numpy as np
root=Path('output/lerf_system_refinement_20260919');rows={};bindings={};expected={'figurines':56,'ramen':71,'teatime':59,'waldo_kitchen':22};names=['final','baseline_calibrated','boundary_soft_calibrated','source_rerank32_calibrated']
for scene,count in expected.items():
 p=root/scene;d=json.loads((p/'report.json').read_text());s=json.loads((p/'selected/report.json').read_text());src=json.loads((p/'source/report.json').read_text());rel=json.loads((p/'relations.json').read_text());assert d['observation_count']==count and d['final_prediction_disagreement_pixels']==0
 rows[scene]=dict(metrics={},source=src['macro'],relations={k:v for k,v in rel.items() if k not in ['registered','views']})
 for name in names:
  assert s['results'][name]['count']==count and s['results'][name]['missing']==0
  rows[scene]['metrics'][name]=dict(iou=d['macro_membership_iou'][name],ap=d['macro_ap'][name],selected_iou=s['results'][name]['miou'])
 for filename in ['registration.json','preparation.json','report.json','scores.pt','graphs.pt','relations.json','source/report.json','selected/report.json','selected/selections.pt']:
  q=p/filename;bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
for q in [root/'source_asset_audit.json',root/'unary_absence_diagnostic.json',root/'unary_absence_native_trace.json']:
 bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
for q in Path('scripts/audits').glob('lerf_system_refinement_20260919*.py'):bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
macro={name:{metric:float(np.mean([rows[s]['metrics'][name][metric] for s in expected])) for metric in ['iou','ap','selected_iou']}for name in names};source_macro={name:{metric:float(np.mean([rows[s]['source'][name][metric] for s in expected])) for metric in ['probability_iou','selected_iou','selected_precision','selected_recall']}for name in ['final','baseline','boundary_soft','source_rerank32']}
report=dict(scenes=rows,scene_equal_macro=macro,source_scene_equal_macro=source_macro,observations=208,field_or_geometry_modified=False,source_only_graph=True,target_parameters_fit=False,full_final_readout_preserved_as_unary=True,independent_test=False,promoted=False,source_pseudo_masks_not_physical_truth=True,controls_not_global_object_graph=True,bindings=bindings)
q=Path('paper/artifacts/lerf_source_relation_control_20260919.json');assert not q.exists();q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(target=macro,source=source_macro),indent=2))
