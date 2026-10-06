import json,torch
from pathlib import Path
rows=[]
for arm in ['all','category','geometry']:
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene;p=Path('output/fixed_measure_next100')/arm/key/'typed_readout.pt'
  if not p.exists():continue
  load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
  x=load(p);b=load(Path('output/category_identity_scope_next98/high_footprint')/key/'typed_readout.pt');ids=[q['query_index'] for q in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];other=[i for i in range(len(x['texts'])) if i not in ids]
  r=dict(arm=arm,scene=scene,semantic_exact=torch.equal(x['scores'],b['scores']),other_columns_exact=torch.equal(x['selection_scores'][:,other],b['selection_scores'][:,other]))
  if arm=='all':
   previous=load(Path('output/uncertain_scope_next99/all')/key/'typed_readout.pt')['selection_scores']
   r['replay_hard_selection_exact']=torch.equal(x['selection_scores']>.5,previous>.5)
   r['replay_max_soft_difference']=float((x['selection_scores']-previous).abs().max())
  rows.append(r)
Path('output/fixed_measure_next100/retained_outputs.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
