"""Report matched source discovery controls after frozen prediction generation."""
import json
from pathlib import Path
import torch,numpy as np
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=Path('output/next106/summary.json');r=json.loads(p.read_text());parent=load('output/core_scope_next104/core_update/lerf_ramen/typed_readout.pt');r['set_discovery']={}
for arm in ['text','exemplar']:
 root=Path(f'output/{arm}_set_scope_next106/lerf_ramen');e=json.loads((root/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']['selected'];pred=load(root/'typed_readout.pt');objects=[o for o in e['objects'] if o['query']=='onion segments'];rows=[]
 for o in objects:
  gt=load_binary_mask(Path(o['gt_path']),threshold=10);m=load_binary_mask(Path(o['pred_path']),threshold=10,grayscale=True);inter=int((gt&m).sum());rows.append(dict(frame=o['frame'],iou=o['iou'],precision=inter/max(1,int(m.sum())),recall=inter/max(1,int(gt.sum())),gt_pixels=int(gt.sum()),pred_pixels=int(m.sum())))
 cols=[i for i in range(parent['selection_scores'].shape[1]) if i!=9];source=json.loads((Path(f'output/{arm}_set_tracks_next106/lerf_ramen/9.json')).read_text())['rows'];r['set_discovery'][arm]=dict(scene_miou=e['miou'],onion_miou=float(np.mean([o['iou'] for o in rows])),mean_precision=float(np.mean([o['precision'] for o in rows])),mean_recall=float(np.mean([o['recall'] for o in rows])),views=rows,source_views=len(source),expanded_source_views=sum(x['added_pixels']>0 for x in source),source_core_pixels=sum(x['core_pixels'] for x in source),source_union_pixels=sum(x['union_pixels'] for x in source),semantic_exact=torch.equal(parent['scores'],pred['scores']),other_selection_columns_exact=torch.equal(parent['selection_scores'][:,cols],pred['selection_scores'][:,cols]))
p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['set_discovery'],indent=2))
