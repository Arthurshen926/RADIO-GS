import json,hashlib
from pathlib import Path
import numpy as np
root=Path('output/graph_transfer_final_20260919');rows={};bindings={};expected={'figurines':56,'ramen':71,'teatime':59,'waldo_kitchen':22};names=['final','direct_dino_calibrated','distilled_structure32_calibrated']
for scene,count in expected.items():
 p=root/scene;d=json.loads((p/'report.json').read_text());s=json.loads((p/'selected/report.json').read_text());assert d['observation_count']==count and d['final_prediction_disagreement_pixels']==0
 rows[scene]={}
 for name in names:
  assert s['results'][name]['count']==count and s['results'][name]['missing']==0
  rows[scene][name]=dict(iou=d['macro_membership_iou'][name],ap=d['macro_ap'][name],selected_iou=s['results'][name]['miou'])
 for filename in ['registration.json','preparation.json','report.json','scores.pt','selected/report.json','selected/selections.pt']:
  q=p/filename;bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
for q in Path('scripts/audits').glob('graph_transfer_20260919*.py'):bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
macro={name:{metric:float(np.mean([rows[s][name][metric] for s in expected])) for metric in ['iou','ap','selected_iou']} for name in names}
report=dict(scenes=rows,scene_equal_macro=macro,observations=208,field_or_geometry_modified=False,source_only_graph=True,target_parameters_fit=False,parameters='16NN, lambda1, confidence0.5, source median distance normalization, no hard seed',full_final_readout_preserved_as_unary=True,independent_test=False,extension_after_two_scene_results=True,promoted=False,tests_passed=19,bindings=bindings)
q=Path('paper/artifacts/graph_transfer_final_four_scene_20260919.json');assert not q.exists();q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(macro,indent=2))
