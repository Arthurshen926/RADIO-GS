"""Keep fixed-recipe method results separate from geometry proxy measurements."""
import json
from pathlib import Path
from statistics import mean
roots={
 'next48':'output/decision_sets_next49/baseline',
 'rgb_next51':'output/refined_selection_next51/fit',
 'initial_depth':'output/depth_selection_next52/fit',
 'recovered_depth':'output/depth_selection_next52_recovered/fit',
 'ray_distribution':'output/ray_selection_next52/fit'}
rows={}
for name,root in roots.items():
    scenes={}
    for p in sorted(Path(root).glob('*/anchored/pipeline_calibrated/evaluation.json')):
        m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];scenes[p.parts[-4]]=dict(iou=m['iou'],ap=m['ap'],selected=m['selected']['miou'])
    if name=='next48':scenes={k:dict(selected=v['selected']) for k,v in scenes.items()}
    if scenes:rows[name]=dict(scenes=scenes,macro_selected=mean(m['selected'] for m in scenes.values()))
geometry={}
for name,root in [('initial_depth','output/carrier_depth_next52'),('recovered_depth','output/carrier_depth_next52_recovered'),('ray_distribution','output/carrier_ray_next52')]:
    geometry[name]={}
    for p in Path(root).glob('*/fit.json'):
        m=json.loads(p.read_text());geometry[name][m['scene']]=dict(before=m['before'],after=m['after'],depth_train_frames=len(m['depth_training_frames']),rgb_train_frames=len(m['source_frames']))
result=dict(development_only=True,default_changed=False,method_results=rows,geometry_proxy=geometry,full_semantic_rebuild_complete=False)
p=Path('output/carrier_depth_next52/summary.json');p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in rows.items()},indent=2))
