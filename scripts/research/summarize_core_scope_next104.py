"""Summarize matched positive-admission controls without selecting per-query winners."""
import json
from pathlib import Path
import numpy as np
root=Path('output/core_scope_next104');result={}
for arm in ['core','pending','core_update']:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene;path=root/arm/key/'anchored/pipeline_calibrated/evaluation.json'
  if not path.exists():continue
  def read(p):return json.loads(p.read_text())['metrics']['hierarchical_pipeline']['selected']
  b=read(Path('output/category_identity_scope_next98/high_footprint' if arm=='core_update' else 'output/reference_component_scope_next103/rebuilt')/key/'anchored/pipeline_calibrated/evaluation.json');c=read(path)
  queries=[]
  for q in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()):
   def v(m):return float(np.mean([r['iou'] for r in m['objects'] if r['query']==q['query']]))
   queries.append(dict(query=q['query'],baseline=v(b),candidate=v(c)))
  rows.append(dict(scene=scene,baseline=b['miou'],candidate=c['miou'],queries=queries))
 if rows:result[arm]=dict(complete_scenes=len(rows),macro=float(np.mean([r['candidate'] for r in rows])),scenes=rows)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
