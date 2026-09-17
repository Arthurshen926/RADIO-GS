"""Append provable zero-support misses to prior frozen diagnostics; preserve originals."""
import json
from pathlib import Path
from radio_gs.training.scope_validation_cohort import complete_scope_cohort
from radio_gs.utils.immutable_artifacts import sha256_file
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene;field=root/'field';info=json.loads((field/'teachers.json').read_text());groupfile=field/'grounded_semantic_transfer96/report.json';groups=json.loads(groupfile.read_text())['groups'];group_frames=[[info['frames'][v] for v in g['views']] for g in groups]
 for name in ['partitioned_scope_validation_20260918','local_partitioned_scope_validation_20260918','bridged_scope_validation_20260918','local_scope_abstention_20260918']:
  source=root/name/'report.json';out=source.with_name('full_cohort_report.json');assert not out.exists();d=json.loads(source.read_text());arms=list(d['macro']);rows,added=complete_scope_cohort(d['rows'],info['train_frames'],group_frames,arms)
  for row in rows:row['phrase']=groups[row['group']]['phrase']
  macro={arm:{key:sum(r[arm][key] for r in rows)/len(rows) for key in ['iou','precision','recall','positive_support']} for arm in arms};folds={str(f):{arm:sum(r[arm]['iou'] for r in rows if r['fold']==f)/sum(r['fold']==f for r in rows) for arm in arms} for f in range(4)}
  report=dict(d,rows=rows,macro=macro,fold_metrics=folds,observations=len(rows),zero_support_observations_added=added,original_report_sha256=sha256_file(source),cohort_completion_implementation_sha256=sha256_file(__file__),cohort_completion_module_sha256=sha256_file('radio_gs/training/scope_validation_cohort.py'),cohort_rule='All heldout group/view detections; no-train-evidence groups are zero predictions by original initialized values/known arrays, not omitted; available-evidence omissions raise.')
  out.write_text(json.dumps(report,indent=2)+'\n');print(scene,name,len(rows),added,{k:v['iou'] for k,v in macro.items()})
