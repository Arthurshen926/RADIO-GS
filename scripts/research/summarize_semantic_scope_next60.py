import json
from pathlib import Path
from statistics import mean
controls={}
for family,modes in [('semantic_scope_next60',['semantic','joint']),('empty_bootstrap_next60',['bootstrap','response'])]:
 for mode in modes:
  rows={}
  for p in Path('output',family,mode).glob('*/anchored/pipeline_calibrated/evaluation.json'):
   m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];r={k:m[k] for k in ['iou','ap','localization']};r['selected']=m['selected']['miou'];rows[p.parts[3]]=r
  controls[family+'/'+mode]=dict(scenes=rows,scene_count=len(rows),macro={k:mean(v[k] for v in rows.values()) for k in ['iou','ap','localization','selected']})
Path('output/semantic_scope_next60/summary.json').write_text(json.dumps(dict(controls=controls,development_only=True,default_changed=False),indent=2)+'\n')
print({k:v['macro'] for k,v in controls.items()})
