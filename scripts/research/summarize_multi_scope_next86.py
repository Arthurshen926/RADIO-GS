"""Report complete scene means and per-query changes without winner stitching."""
import json,argparse
p=argparse.ArgumentParser();p.add_argument("--iteration",default="next86");p.add_argument("--baseline-base",default="output/grounded_scope_next85/source");a=p.parse_args()
from pathlib import Path
import numpy as np
out=Path('output')/a.iteration;results={}
for policy in ['union','unknown','factorized']:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene
  def metric(base):return json.loads((Path(base)/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
  path=Path('output/multi_scope_'+a.iteration)/policy/key/'anchored/pipeline_calibrated/evaluation.json'
  if not path.exists():continue
  baseline=metric(a.baseline_base);candidate=metric(str(path.parents[3]));changed=json.loads((Path('output/multi_seeds_'+a.iteration)/key/'proposals.json').read_text());queries=[]
  for q in changed:
   def value(m):return float(np.mean([r['iou'] for r in m['selected']['objects'] if r['query']==q['query']]))
   queries.append(dict(query=q['query'],secondary=q['secondary_rank'],baseline=value(baseline),candidate=value(candidate)))
  rows.append(dict(scene=scene,baseline=baseline['selected']['miou'],candidate=candidate['selected']['miou'],queries=queries))
 if rows:results[policy]=dict(complete_scenes=len(rows),scenes=rows,baseline_macro=float(np.mean([r['baseline'] for r in rows])),candidate_macro=float(np.mean([r['candidate'] for r in rows])))
(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
