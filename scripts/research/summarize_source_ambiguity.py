"""Post-prediction diagnosis of category-ambiguity experiments."""
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
BASE=Path('output/system_v5/scannet')
ARMS={k:Path('output/source_category_ambiguity')/k for k in ['entropy','margin']}
rows={}; totals={}; classes={}; spoiled_transitions={}
for name,path in ARMS.items():
 rows[name]={};totals[name]={'corrected_points':0,'spoiled_points':0,'changed_valid_points':0,'valid_points':0};classes[name]={}
 spoiled_transitions[name]={}
 for folder in sorted(path.glob('scene*')):
  scene=folder.name
  before=json.loads((BASE/scene/'evaluation/report.json').read_text())['point_metrics']['source']
  after=json.loads((folder/'evaluation/report.json').read_text())['point_metrics']['source']
  spec=json.loads(Path(f'configs/consolidated/scannet_native/{scene}.evaluation.json').read_text())
  truth=np.asarray(PlyData.read(spec['ground_truth'])['vertex'].data['label'])
  b=np.load(BASE/scene/'evaluation/source_point_predictions.npz')['19']
  a=np.load(folder/'evaluation/source_point_predictions.npz')['19']
  valid=np.isin(truth,spec['splits']['19'])
  pairs={'corrected_points':int((valid&(b!=truth)&(a==truth)).sum()),'spoiled_points':int((valid&(b==truth)&(a!=truth)).sum()),'changed_valid_points':int((valid&(b!=a)).sum()),'valid_points':int(valid.sum())}
  for k,v in pairs.items():totals[name][k]+=v
  selected=valid&(b==truth)&(a!=truth)
  transitions,counts=np.unique(np.stack([truth[selected],a[selected]],1),axis=0,return_counts=True)
  for (source,target),count in zip(transitions,counts):
   key=f'{source}->{target}'
   spoiled_transitions[name][key]=spoiled_transitions[name].get(key,0)+int(count)
  rows[name][scene]={'final_delta_pp':{s:100*(after[s]['miou']-before[s]['miou']) for s in ['19','15','10']},'paired_points':pairs}
  for c,r in after['19']['per_class'].items():
   if r['gt_count']:
    entry=classes[name].setdefault(c,{'name':r['name'],'scenes':[],'deltas_pp':[]})
    entry['scenes'].append(scene);entry['deltas_pp'].append(100*(r['iou']-before['19']['per_class'][c]['iou']))
 for entry in classes[name].values():entry['mean_delta_pp']=float(np.mean(entry['deltas_pp']))
summary={'per_scene':rows,'point_totals_not_scene_macro':totals,'spoiled_transitions':spoiled_transitions,'per_class_valid_scene_means_not_headline_metric':classes,'labels_read_only_after_predictions':True,'independent_test':False}
Path('output/source_category_ambiguity/comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(totals,indent=2))
