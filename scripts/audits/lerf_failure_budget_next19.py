"""Read-only development diagnosis: calibration versus ranking, no parameters fitted."""
import json
from pathlib import Path
import numpy as np
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/lerf_failure_budget_next19')
OUT.mkdir(exist_ok=False)
SCENES=['figurines','ramen','teatime','waldo_kitchen']
registration={'diagnostic_only':True,'target_metrics_used_for_diagnosis':True,'parameter_fitting':False,'bins':{'poor_ranking_ap_below':.25,'good_ranking_ap_at_least':.75,'low_iou_below':.1},'implementation_sha256':sha256_file(__file__)}
(OUT/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
rows=[];scenes={};bindings={}
for scene in SCENES:
 root=Path('output/graph_transfer_final_20260919')/scene
 path=root/'report.json';r=json.loads(path.read_text());bindings[str(path)]=sha256_file(path)
 path=root/'scores.pt';bindings[str(path)]=sha256_file(path);data=torch.load(path,map_location='cpu',weights_only=False)
 key='distilled_structure32_calibrated';p=data['scores'][key];obs=data['observed'];counts=(p[obs]>.5).sum(0);maxima=p[obs].amax(0)
 sr=[]
 for old in r['observations']:
  qi=data['texts'].index(old['query']);ap=old['ap'][key];iou=old['membership_iou'][key]
  row={'scene':scene,'frame':old['frame'],'query':old['query'],'ap':ap,'iou':iou,'selected_elements':int(counts[qi]),'maximum_membership':float(maxima[qi]),'positive_support_recall':old['positive_support_recall'],'poor_ranking':ap<.25,'good_ranking_low_iou':ap>=.75 and iou<.1,'zero_selection':int(counts[qi])==0}
  rows.append(row);sr.append(row)
 scenes[scene]={'observations':len(sr),'mean_ap':float(np.mean([x['ap'] for x in sr])),'mean_iou':float(np.mean([x['iou'] for x in sr])),'poor_ranking_count':sum(x['poor_ranking'] for x in sr),'good_ranking_low_iou_count':sum(x['good_ranking_low_iou'] for x in sr),'zero_selection_observations':sum(x['zero_selection'] for x in sr),'mean_positive_support':float(np.mean([x['positive_support_recall'] for x in sr]))}
print(json.dumps(scenes,indent=2))
result={'scenes':scenes,'observations':rows,'count':len(rows),'source_bindings':bindings,'registration_sha256':sha256_file(OUT/'registration.json'),'interpretation':'Bins describe development outcomes, not a query-time classifier. High AP and low IoU suggests calibration/readout failure; low AP requires ranking/localization repair. Positive support is coverage, not accurate geometry. Never use target labels to choose thresholds.'}
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
