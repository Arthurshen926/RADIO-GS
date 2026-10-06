import json,hashlib
from pathlib import Path
import numpy as np
root=Path('output/lerf_query_calibration_next19');scenes={};bindings={};expected={'figurines':56,'ramen':71,'teatime':59,'waldo_kitchen':22};arms=['final','baseline_calibrated','zmedian_calibrated','zridge_calibrated']
for scene,count in expected.items():
 p=root/scene;d=json.loads((p/'report.json').read_text());s=json.loads((p/'selected/report.json').read_text());src=json.loads((p/'source_report.json').read_text());matched=Path('output/lerf_query_calibration_next19_matched')/scene/'source_report.json';m=json.loads(matched.read_text());assert d['observation_count']==count and d['final_prediction_disagreement_pixels']==0
 for k in ['current_in_sample','absolute_loo']:
  for metric in ['probability_iou','selected_iou']:assert abs(src['macro'][k][metric]-m['macro'][k][metric])<1e-6
 scenes[scene]=dict(metrics={},source=src['macro'],matched_absolute=m['macro'],source_observations=src['observation_count'],source_query_groups=len(set(src['group_ids'])),changes=d['changes'])
 for arm in arms:
  assert s['results'][arm]['missing']==0 and s['results'][arm]['count']==count
  scenes[scene]['metrics'][arm]=dict(iou=d['macro_membership_iou'][arm],ap=d['macro_ap'][arm],selected_iou=s['results'][arm]['miou'])
 for name in ['registration.json','preparation.json','source_report.json','report.json','scores.pt','selected/report.json','selected/selections.pt']:
  q=p/name;bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
 bindings[str(matched)]=hashlib.sha256(matched.read_bytes()).hexdigest()
for q in [root/'native_target_compatibility.json',root/'protected_scope_replay.json',*Path('scripts/audits').glob('lerf_query_calibration_next19*.py')]:bindings[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
macro={arm:{k:float(np.mean([v['metrics'][arm][k]for v in scenes.values()]))for k in ['iou','ap','selected_iou']}for arm in arms};report=dict(scenes=scenes,scene_equal_macro=macro,observations=208,promoted=False,source_acceptance_not_jointly_passed=True,target_gt_fit=False,independent_test=False,field_and_geometry_unchanged=True,protected_scope_actual_selection_exact=True,all_source_calibration_targets_native_compatible=True,bindings=bindings)
p=Path('paper/artifacts/lerf_query_calibration_next19.json');assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(macro,indent=2))
