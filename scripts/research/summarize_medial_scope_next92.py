"""Full-scene eroded versus medial-positive comparisons."""
import json
from pathlib import Path
import numpy as np
results={}
for arm,base,parent in [('primary','output/medial_scope_next92','output/grounded_scope_next88/source'),('description','output/medial_description_scope_next92','output/description_scope_next91')]:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene
  def metrics(b):return json.loads((Path(b)/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
  if not (Path(base)/key/'anchored/pipeline_calibrated/evaluation.json').exists():continue
  b=metrics(parent);c=metrics(base);queries=[]
  for q in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()):
   def val(m):return float(np.mean([r['iou'] for r in m['selected']['objects'] if r['query']==q['query']]))
   queries.append(dict(query=q['query'],baseline=val(b),candidate=val(c)))
  rows.append(dict(scene=scene,baseline=b['selected']['miou'],candidate=c['selected']['miou'],queries=queries))
 if rows:results[arm]=dict(complete_scenes=len(rows),baseline_macro=float(np.mean([r['baseline'] for r in rows])),candidate_macro=float(np.mean([r['candidate'] for r in rows])),scenes=rows)
out=Path('output/next92');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
