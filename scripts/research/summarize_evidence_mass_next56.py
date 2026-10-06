import json
from pathlib import Path
from statistics import mean
controls={}
for family,modes in [('evidence_mass_next56',['mass_scope','mass_masscal']),('corroborated_set_next56',['agreement','crossview','decision','anneal'])]:
 for mode in modes:
  rows={}
  for p in Path('output',family,mode).glob('*/anchored/pipeline_calibrated/evaluation.json'):
   m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];r={k:m[k] for k in ['iou','ap','localization']};r['selected']=m['selected']['miou'];rows[p.parts[3]]=r
  if rows:controls[family+'/'+mode]=dict(scenes=rows,scene_count=len(rows),macro={k:mean(v[k] for v in rows.values()) for k in ['iou','ap','localization','selected']})
Path('output/evidence_mass_next56/summary.json').write_text(json.dumps(dict(controls=controls,development_only=True,default_changed=False,selected_metric='2D projection of selected Gaussians, not physical 3D GT'),indent=2)+'\n')
print(json.dumps({k:dict(scene_count=v['scene_count'],macro=v['macro']) for k,v in controls.items()},indent=2))
