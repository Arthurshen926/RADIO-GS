"""Matched source-initialization outcomes with all changed-query counts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--all-queries',action='store_true');a=p.parse_args();rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;oldpath=Path('output/region_seed_positive_next80')/key;newpath=Path('output/all_source_scope_next81' if a.all_queries else 'output/source_scope_next81')/key
 old=json.loads((oldpath/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline'];new=json.loads((newpath/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline'];fit=json.loads((newpath/'fit.json').read_text());queries=[]
 for h in fit['history']:
  def value(m):return float(np.mean([v['iou'] for v in m['selected']['objects'] if v['query']==h['query']]))
  queries.append(dict(**h,baseline_iou=value(old),candidate_iou=value(new)))
 load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
 b=load(oldpath/'typed_readout.pt');c=load(newpath/'typed_readout.pt');changed={q['query'] for q in queries};other=[i for i,t in enumerate(b['texts']) if t not in changed]
 rows.append(dict(scene=scene,baseline_selected=old['selected']['miou'],candidate_selected=new['selected']['miou'],semantic_scores_equal=torch.equal(b['scores'],c['scores']),other_selections_equal=torch.equal(b['selection_scores'][:,other],c['selection_scores'][:,other]),queries=queries))
report=dict(scenes=rows,baseline_macro_selected=float(np.mean([r['baseline_selected'] for r in rows])),candidate_macro_selected=float(np.mean([r['candidate_selected'] for r in rows])))
out=Path('output/next81');out.mkdir(exist_ok=True);(out/('all_summary.json' if a.all_queries else 'summary.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
