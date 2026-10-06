import json,torch
from pathlib import Path
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
report=[];counts={}
for arm,seedbase,root in [('query','output/component_seeds_next103','output/component_scope_next103/rebuilt'),('reference','output/reference_component_seeds_next103','output/reference_component_scope_next103/rebuilt')]:
 totals=dict(queries=0,changed=0,judgments=0,A=0,B=0,C=0,all_rejected_fallback=0)
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  key='lerf_'+scene
  for r in json.loads((Path(seedbase)/key/'report.json').read_text()):
   totals['queries']+=1;totals['changed']+=r['changed'];totals['all_rejected_fallback']+=r['all_rejected_fallback']
   for j in r['judgments']:totals['judgments']+=1;totals[j['answer']]+=1
  p=Path(root)/key/'typed_readout.pt'
  if not p.exists():continue
  x=load(p);b=load(Path('output/category_identity_scope_next98/high_footprint')/key/'typed_readout.pt');ids=[q['query_index'] for q in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];other=[i for i in range(len(x['texts'])) if i not in ids]
  report.append(dict(arm=arm,scene=scene,semantic_exact=torch.equal(x['scores'],b['scores']),other_columns_exact=torch.equal(x['selection_scores'][:,other],b['selection_scores'][:,other])))
 counts[arm]=totals
out=Path('output/next103');(out/'source_counts.json').write_text(json.dumps(counts,indent=2)+'\n');(out/'retained_outputs.json').write_text(json.dumps(report,indent=2)+'\n');print(counts);print(report)
