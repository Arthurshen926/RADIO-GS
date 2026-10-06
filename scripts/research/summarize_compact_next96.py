"""Matched exclusion policies with fixed source hypotheses."""
import json
from pathlib import Path
import numpy as np
results={}
for arm in ['low_count','low_footprint','high_footprint']:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene
  def m(base):return json.loads((Path(base)/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
  base=Path('output/compact_scope_next96')/arm
  if not (base/key/'anchored/pipeline_calibrated/evaluation.json').exists():continue
  b=m('output/boundary_scope_next95/'+arm.split('_')[0]);c=m(base);queries=[]
  for q in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()):
   def v(m):return float(np.mean([r['iou'] for r in m['selected']['objects'] if r['query']==q['query']]))
   queries.append(dict(query=q['query'],baseline=v(b),candidate=v(c)))
  rows.append(dict(scene=scene,baseline=b['selected']['miou'],candidate=c['selected']['miou'],queries=queries))
 if rows:results[arm]=dict(complete_scenes=len(rows),baseline_macro=float(np.mean([r['baseline'] for r in rows])),candidate_macro=float(np.mean([r['candidate'] for r in rows])),scenes=rows)
Path('output/next96/summary.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
