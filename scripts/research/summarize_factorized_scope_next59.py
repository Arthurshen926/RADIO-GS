import json
from pathlib import Path
from statistics import mean
controls={}
for mode in ['joint_view','per_instance','hierarchy']:
 rows={}
 for p in Path('output/factorized_scope_next59',mode).glob('*/anchored/pipeline_calibrated/evaluation.json'):
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];r={k:m[k] for k in ['iou','ap','localization']};r['selected']=m['selected']['miou'];rows[p.parts[3]]=r
 controls[mode]=dict(scenes=rows,scene_count=len(rows),macro={k:mean(v[k] for v in rows.values()) for k in ['iou','ap','localization','selected']})
Path('output/factorized_scope_next59/summary.json').write_text(json.dumps(dict(controls=controls,development_only=True,default_changed=False),indent=2)+'\n')
print(json.dumps({k:v['macro'] for k,v in controls.items()},indent=2))
