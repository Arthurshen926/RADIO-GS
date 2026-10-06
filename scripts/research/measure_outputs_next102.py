import json,torch
from pathlib import Path
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
rows=[]
for arm,root in [('agreement','output/partner_exclusion_next102/agreement'),('veto','output/partner_exclusion_next102/veto'),('clean','output/clean_scope_next102/clean'),('rebuilt','output/clean_scope_next102/rebuilt')]:
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene;p=Path(root)/key/'typed_readout.pt'
  if not p.exists():continue
  x=load(p);b=load(Path('output/category_identity_scope_next98/high_footprint')/key/'typed_readout.pt');ids=[q['query_index'] for q in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];other=[i for i in range(len(x['texts'])) if i not in ids]
  rows.append(dict(arm=arm,scene=scene,semantic_exact=torch.equal(x['scores'],b['scores']),other_columns_exact=torch.equal(x['selection_scores'][:,other],b['selection_scores'][:,other])))
Path('output/partner_exclusion_next102/retained_outputs.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
