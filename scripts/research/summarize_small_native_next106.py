import json
from pathlib import Path
import torch
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
base=Path('output/next106');report={'experiment':'next106','promoted':False,'scope':'two-query source diagnosis; ramen onion native-top1 old/expanded control; development only'}
report['source_diagnosis']={}
for key,qi in [('lerf_ramen',9),('lerf_waldo_kitchen',7)]:
 root=base/key;r=json.loads((root/f'{qi}_unpaired.json').read_text());p=json.loads((root/f'{qi}_pair_funnel.json').read_text());c=json.loads((root/'small_regions.json').read_text());report['source_diagnosis'][key]=dict(query=r['query'],counts=c['counts'],source_frames=c['source_frames'],new_small_in_top64=r['new_small_in_top64'],single_view_checks=r['verified'],positive_margins=sum(x['margin']>0 for x in r['rows']),new_small_positive_margins=sum(x['margin']>0 and x['small'] for x in r['rows']),pair_funnel_exact=p['reproduces_old'],pair_outcomes=p['outcomes'])
key='lerf_ramen';parentroot=Path('output/core_scope_next104/core_update');parent=load(parentroot/key/'typed_readout.pt');report['controls']={}
for arm in ['old','expanded']:
 root=Path('output/small_native_scope_next106')/arm/key;r=load(root/'typed_readout.pt');e=json.loads((root/'anchored/pipeline_calibrated/evaluation.json').read_text());metric=e['metrics']['hierarchical_pipeline']['selected'];rows=[o for o in metric['objects'] if o['query']=='onion segments'];columns=[i for i in range(parent['selection_scores'].shape[1]) if i!=9];entry=dict(scene_miou=metric['miou'],onion_objects=rows,semantic_exact=torch.equal(parent['scores'],r['scores']),other_selection_columns_exact=torch.equal(parent['selection_scores'][:,columns],r['selection_scores'][:,columns]),seed=json.loads((Path('output/small_native_seeds_next106')/arm/key/'9.json').read_text()));entry['onion_miou']=sum(o['iou'] for o in rows)/len(rows);report['controls'][arm]=entry
report['retained_parent']=dict(selected_projection_macro_miou=.6259859003434166,semantic_miou=.669300891,semantic_ap=.841459268,changed=False)
(base/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['controls'],indent=2))
