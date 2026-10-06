"""Report every arm under the same selected-projection evaluator."""
import json
from pathlib import Path
import numpy as np
arms={}
for arm in ['raw','contrast']:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene
  def metrics(base):return json.loads((Path(base)/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
  retained=metrics('output/region_seed_positive_next80');native=metrics('output/source_scope_next81');tight=metrics('output/content_scope_next83/source');m=metrics('output/counterfactual_scope_next84/'+arm);queries=[]
  for r in json.loads((Path('output/counterfactual_seeds_next84')/arm/key/'proposals.json').read_text()):
   def value(m):return float(np.mean([v['iou'] for v in m['selected']['objects'] if v['query']==r['query']]))
   queries.append(dict(query=r['query'],native_fallback=r['native_fallback'],retained=value(retained),native_source_init=value(native),tight_likelihood=value(tight),candidate=value(m)))
  rows.append(dict(scene=scene,retained=retained['selected']['miou'],native_source_init=native['selected']['miou'],tight_likelihood=tight['selected']['miou'],candidate=m['selected']['miou'],queries=queries))
 arms[arm]=dict(scenes=rows,macro={k:float(np.mean([r[k] for r in rows])) for k in ['retained','native_source_init','tight_likelihood','candidate']})
out=Path('output/next84');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(arms,indent=2)+'\n');print(json.dumps(arms,indent=2))
