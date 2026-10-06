"""Full LERF cohort, joint IoU/AP/selection reporting without per-query choices."""
import json,torch
from pathlib import Path
scenes=['figurines','ramen','teatime','waldo_kitchen'];root=Path('output/consolidated_response_control');rows=[]
for s in scenes:
    base=Path('output/conditioned_identity/full_source')/s;parent=torch.load(base/'sets'/('lerf_'+s)/'typed_readout.pt',map_location='cpu',weights_only=False);arms={}
    for a in ['baseline','native_protected','source_complement','fusion','native_floor']:
        armdir=root/s/'fusion/native_protected' if a=='fusion' else root/s/a
        p=base/'evaluation/evaluation.json' if a=='baseline' else armdir/'evaluation/evaluation.json';d=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];arms[a]={k:d[k] for k in ['iou','ap','localization','observations']};arms[a]['selected_iou']=d['selected']['miou']
        if a!='baseline':
            new=torch.load(armdir/'typed_readout.pt',map_location='cpu',weights_only=False);arms[a]['selection_tensor_exact']=torch.equal(parent['selection_scores'],new['selection_scores']);arms[a]['semantic_observed_exact']=torch.equal(parent['semantic_observed'],new['semantic_observed'])
    rows.append(dict(scene=s,arms=arms))
result=dict(scenes=rows,macro={a:{m:sum(r['arms'][a][m] for r in rows)/4 for m in ['iou','ap','localization','selected_iou']} for a in ['baseline','native_protected','source_complement','fusion','native_floor']},default_changed=True,default_profile='identity_response',default_solver='backprojection',development_only=True,target_rgb_used=False,new_vfm_calls=0,geometry_changed=False,selection_changed=False)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result['macro'])
for r in rows:print(r['scene'],r['arms'])
