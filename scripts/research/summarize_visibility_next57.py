import json
from pathlib import Path
from statistics import mean
controls={}
for family,modes in [('visibility_set_next57',['visible','deletion','relative_deletion']),('source_validation_next57',['decision','validated'])]:
 for mode in modes:
  rows={}
  for p in Path('output',family,mode).glob('*/anchored/pipeline_calibrated/evaluation.json'):
   m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];r={k:m[k] for k in ['iou','ap','localization']};r['selected']=m['selected']['miou'];rows[p.parts[3]]=r
  controls[family+'/'+mode]=dict(scenes=rows,scene_count=len(rows),macro={k:mean(v[k] for v in rows.values()) for k in ['iou','ap','localization','selected']})
Path('output/visibility_set_next57/summary.json').write_text(json.dumps(dict(controls=controls,retained='output/corroborated_set_next56/decision',development_only=True,default_changed=False),indent=2)+'\n')
print({k:v['macro']['selected'] for k,v in controls.items()})
