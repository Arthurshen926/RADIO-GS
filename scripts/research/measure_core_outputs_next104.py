import json,torch
from pathlib import Path
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
rows=[]
for arm in ['core','pending','core_update']:
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene;p=Path('output/core_scope_next104')/arm/key/'typed_readout.pt'
  if not p.exists():continue
  x=load(p);b=load(Path('output/category_identity_scope_next98/high_footprint')/key/'typed_readout.pt');ids=[r['query_index'] for r in json.loads((Path('output/core_seeds_next104')/key/'report.json').read_text()) if arm!='core_update' or r['changed']];other=[i for i in range(len(x['texts'])) if i not in ids]
  rows.append(dict(arm=arm,scene=scene,updated_query_indices=ids,semantic_exact=torch.equal(x['scores'],b['scores']),retained_selection_exact=torch.equal(x['selection_scores'][:,other],b['selection_scores'][:,other])))
Path('output/core_scope_next104/retained_outputs.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
