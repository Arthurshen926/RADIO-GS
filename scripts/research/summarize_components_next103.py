import json
from pathlib import Path
import numpy as np
results={}
for arm,root in [('query','output/component_scope_next103/rebuilt'),('reference','output/reference_component_scope_next103/rebuilt')]:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene;p=Path(root)/key/'anchored/pipeline_calibrated/evaluation.json'
  if not p.exists():continue
  def read(path):return json.loads(path.read_text())['metrics']['hierarchical_pipeline']['selected']
  b=read(Path('output/coverage_scope_next97/high_footprint')/key/'anchored/pipeline_calibrated/evaluation.json');c=read(p);queries=[]
  for q in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()):
   def value(d):return float(np.mean([r['iou'] for r in d['objects'] if r['query']==q['query']]))
   queries.append(dict(query=q['query'],baseline=value(b),candidate=value(c)))
  rows.append(dict(scene=scene,baseline=b['miou'],candidate=c['miou'],queries=queries))
 if rows:results[arm]=dict(complete_scenes=len(rows),macro=float(np.mean([r['candidate'] for r in rows])),scenes=rows)
out=Path('output/next103');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print([(a,r['complete_scenes'],r['macro']) for a,r in results.items()])
