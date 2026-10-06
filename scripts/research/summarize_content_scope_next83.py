"""Matched content-reranking comparison, plus retained-next80 reference; no winner mixing."""
import json
from pathlib import Path
import numpy as np
arms={}
for initialization in ['semantic','source']:
 rows=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene
  def metrics(base):return json.loads((Path(base)/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
  retained=metrics('output/region_seed_positive_next80');parent=metrics('output/region_seed_positive_next80' if initialization=='semantic' else 'output/source_scope_next81');candidate=metrics(f'output/content_scope_next83/{initialization}');proposals=json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text());queries=[]
  for p in proposals:
   def value(m):return float(np.mean([r['iou'] for r in m['selected']['objects'] if r['query']==p['query']]))
   queries.append(dict(query=p['query'],retained=value(retained),matched_parent=value(parent),candidate=value(candidate)))
  rows.append(dict(scene=scene,retained=retained['selected']['miou'],matched_parent=parent['selected']['miou'],candidate=candidate['selected']['miou'],semantic_iou=candidate['iou'],semantic_ap=candidate['ap'],queries=queries))
 arms[initialization]=dict(scenes=rows,retained_macro=float(np.mean([r['retained'] for r in rows])),matched_parent_macro=float(np.mean([r['matched_parent'] for r in rows])),candidate_macro=float(np.mean([r['candidate'] for r in rows])))
out=Path('output/next83');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(arms,indent=2)+'\n');print(json.dumps(arms,indent=2))
