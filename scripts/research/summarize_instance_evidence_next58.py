import json
from pathlib import Path
from statistics import mean
controls={}
for family,modes in [('instance_evidence_next58',['local_identity','instance_balance','extent_unknown','clip_identity']),('clip_field_next58',['mixture','mixture_masscal','contrastive'])]:
 for mode in modes:
  rows={}
  for p in Path('output',family,mode).glob('*/anchored/pipeline_calibrated/evaluation.json'):
   m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];r={k:m[k] for k in ['iou','ap','localization']};r['selected']=m['selected']['miou'];rows[p.parts[3]]=r
  controls[family+'/'+mode]=dict(scenes=rows,scene_count=len(rows),macro={k:mean(v[k] for v in rows.values()) for k in ['iou','ap','localization','selected']})
Path('output/instance_evidence_next58/summary.json').write_text(json.dumps(dict(controls=controls,development_only=True,default_changed=False),indent=2)+'\n')
print(json.dumps({k:v['macro'] for k,v in controls.items()},indent=2))
