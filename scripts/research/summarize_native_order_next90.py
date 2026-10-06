"""Matched native ranking and admission controls; no query-wise branch choice."""
import json
from pathlib import Path
import numpy as np
results={}
for arm in ['native','qualified_native']:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene
  def metric(base):return json.loads((Path(base)/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
  destination=Path('output/native_order_scope_next90')/arm/'source'/key/'anchored/pipeline_calibrated/evaluation.json'
  if not destination.exists():continue
  b=metric('output/grounded_scope_next88/source');c=metric(str(destination.parents[3]));queries=[]
  for row in json.loads((Path('output/native_order_seeds_next90')/arm/key/'proposals.json').read_text()):
   def value(m):return float(np.mean([r['iou'] for r in m['selected']['objects'] if r['query']==row['query']]))
   queries.append(dict(query=row['query'],baseline=value(b),candidate=value(c)))
  rows.append(dict(scene=scene,baseline=b['selected']['miou'],candidate=c['selected']['miou'],queries=queries))
 if rows:results[arm]=dict(complete_scenes=len(rows),baseline_macro=float(np.mean([r['baseline'] for r in rows])),candidate_macro=float(np.mean([r['candidate'] for r in rows])),scenes=rows)
Path('output/next90/summary.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
