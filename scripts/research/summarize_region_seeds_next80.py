"""Fixed-evaluator scene and query outcomes for the missing-seed intervention."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=['figurines','ramen','teatime','waldo_kitchen']);p.add_argument('--positive',action='store_true');a=p.parse_args();rows=[]
for scene in a.scenes:
 key='lerf_'+scene;b=Path('output/full_text_decode_next79/complete/all')/key/'anchored/pipeline_calibrated/evaluation.json';c=Path('output/region_seed_positive_next80' if a.positive else 'output/region_seed_decode_next80')/key/'anchored/pipeline_calibrated/evaluation.json'
 old=json.loads(b.read_text())['metrics']['hierarchical_pipeline'];new=json.loads(c.read_text())['metrics']['hierarchical_pipeline'];proposals=json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text());fit=json.loads((c.parents[2]/'fit.json').read_text());queries=[]
 for r in proposals:
  def value(m):return float(np.mean([v['iou'] for v in m['selected']['objects'] if v['query']==r['query']]))
  history=next((h for h in fit['history'] if h['query']==r['query']),{})
  queries.append(dict(query=r['query'],baseline=value(old),candidate=value(new),active_views=history.get('views',0)))
 rows.append(dict(scene=scene,baseline_selected=old['selected']['miou'],candidate_selected=new['selected']['miou'],baseline_iou=old['iou'],candidate_iou=new['iou'],baseline_ap=old['ap'],candidate_ap=new['ap'],queries=queries))
out=Path('output/next80');out.mkdir(exist_ok=True);report=dict(scenes=rows,baseline_macro_selected=float(np.mean([r['baseline_selected'] for r in rows])),candidate_macro_selected=float(np.mean([r['candidate_selected'] for r in rows])))
(out/('positive_summary.json' if a.positive else 'summary.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
