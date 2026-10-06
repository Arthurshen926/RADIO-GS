"""Compare selected sets to their matched raster control, using no target labels."""
import json
from pathlib import Path
import torch

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);rows=[]
for arm in ['all','geometry']:
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene;p=Path('output/uncertain_scope_next99')/arm/key
  if not (p/'typed_readout.pt').exists():continue
  a=load(Path('output/category_identity_scope_next98/high_footprint')/key/'typed_readout.pt');b=load(p/'typed_readout.pt')
  ids=[r['query_index'] for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];other=[i for i in range(len(a['texts'])) if i not in ids];queries=[]
  for qi in ids:
   old=(a['selection_scores'][:,qi]>.5)&a['selection_observed'];new=(b['selection_scores'][:,qi]>.5)&b['selection_observed'];queries.append(dict(query=a['texts'][qi],old_selected=int(old.sum()),new_selected=int(new.sum()),removed=int((old&~new).sum()),added=int((new&~old).sum())))
  rows.append(dict(arm=arm,scene=scene,semantic_exact=torch.equal(a['scores'],b['scores']),other_selection_exact=torch.equal(a['selection_scores'][:,other],b['selection_scores'][:,other]),queries=queries))
Path('output/next99/selected_sets.json').write_text(json.dumps(rows,indent=2)+'\n');print([(r['arm'],r['scene'],sum(q['old_selected'] for q in r['queries']),sum(q['new_selected'] for q in r['queries'])) for r in rows])
