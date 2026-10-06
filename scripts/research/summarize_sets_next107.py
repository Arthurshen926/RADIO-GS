"""All four scenes and all changed-query results; no GT-selected method stitching."""
import json
from pathlib import Path
import numpy as np,torch
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
scenes=['figurines','ramen','teatime','waldo_kitchen'];arms={'core':'output/core_scope_next104/core','retained':'output/core_scope_next104/core_update','text':'output/set_scope_next107/text','box':'output/set_scope_next107/box','components':'output/set_scope_next107/components','source_fusion':'output/source_fusion_next107/box','self_supported':'output/self_supported_scope_next107/self_supported','unknown':'output/self_supported_scope_next107/self_supported_unknown'};report={'arms':{},'complete':True}
for arm,base in arms.items():
 rows={}
 for scene in scenes:
  key='lerf_'+scene;root=Path(base)/key;p=root/'anchored/pipeline_calibrated/evaluation.json'
  if not p.exists():report['complete']=False;continue
  e=json.loads(p.read_text())['metrics']['hierarchical_pipeline']['selected'];groups={}
  for o in e['objects']:groups.setdefault(o['query'],[]).append(o['iou'])
  r=dict(miou=e['miou'],queries={q:float(np.mean(v)) for q,v in groups.items()})
  if arm not in ['core','retained']:
   pred=load(root/'typed_readout.pt');parent=load(Path(arms['core'])/key/'typed_readout.pt');ids=[x['query_index'] for x in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];other=[i for i in range(parent['selection_scores'].shape[1]) if i not in ids];r.update(semantic_exact=torch.equal(parent['scores'],pred['scores']),other_selection_columns_exact=torch.equal(parent['selection_scores'][:,other],pred['selection_scores'][:,other]))
  rows[scene]=r
 report['arms'][arm]=dict(scenes=rows,macro_miou=float(np.mean([r['miou'] for r in rows.values()])) if len(rows)==4 else None)
report['source']={}
for arm in ['text','box','components']:
 records=[]
 for scene in scenes:
  key='lerf_'+scene
  for p in sorted((Path('output/set_discovery_next107')/arm/key).glob('*.json')):
   if p.name=='contract.json':continue
   d=json.loads(p.read_text());rows=d['rows'];records.append(dict(scene=scene,query=d['query'],query_index=d['query_index'],views=len(rows),examples=sum(x['examples'] for x in rows),instances=sum(x['instances'] for x in rows),expanded_views=sum(x['added_pixels']>0 for x in rows),core_pixels=sum(x['core_pixels'] for x in rows),union_pixels=sum(x['union_pixels'] for x in rows)))
 report['source'][arm]=records
out=Path('output/next107');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('complete',report['complete']);print({a:dict(macro=r['macro_miou'],scenes={s:x['miou'] for s,x in r['scenes'].items()}) for a,r in report['arms'].items()})
