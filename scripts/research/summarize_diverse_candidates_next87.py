"""Matched four-scene candidate diversity result."""
import json,argparse
p=argparse.ArgumentParser();p.add_argument("--iteration",default="next87");a=p.parse_args()
from pathlib import Path
import numpy as np
rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene
 def metrics(base):return json.loads((Path(base)/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
 b=metrics('output/grounded_scope_next85/source');c=metrics('output/grounded_scope_'+a.iteration+'/source');queries=[]
 for q in json.loads((Path('output/grounded_seeds_'+a.iteration)/key/'proposals.json').read_text()):
  def value(m):return float(np.mean([r['iou'] for r in m['selected']['objects'] if r['query']==q['query']]))
  queries.append(dict(query=q['query'],baseline=value(b),candidate=value(c),native_fallback=q['native_fallback']))
 rows.append(dict(scene=scene,baseline=b['selected']['miou'],candidate=c['selected']['miou'],semantic_iou=c['iou'],queries=queries))
s=dict(scenes=rows,baseline_macro=float(np.mean([r['baseline'] for r in rows])),candidate_macro=float(np.mean([r['candidate'] for r in rows])));(Path('output')/a.iteration/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
