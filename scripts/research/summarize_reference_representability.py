"""Macro-average within each scene before averaging across the eight queries."""
import json
from pathlib import Path
root=Path('output/reference_representability');scenes=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex'];rows=[json.loads((root/s/'diagnosis.json').read_text()) for s in scenes]
summary={domain:{arm:{metric:sum(r['summary'][domain][arm][metric] for r in rows)/len(rows) for metric in ['iou','precision','recall','mae','huber']} for arm in ['joint120','joint480','single120','single480']} for domain in ['known','full']}
result=dict(scenes=rows,macro=summary,total_source_views=sum(r['source_views'] for r in rows),target_labels_used=False,target_rgb_used=False,diagnostic_only=True,default_changed=False,global_optimum_proven=False)
target_rows=[]
for scene in scenes:
    row=dict(scene=scene)
    for name,path in [('baseline120',Path(f'output/reference_integration/fresh/{scene}/evaluation/evaluation.json')),('baseline480',Path(f'output/reference_resolution/long_solver/{scene}/evaluation/evaluation.json')),('full480',root/scene/'full_domain_evaluation/evaluation.json')]:
        row[name]=json.loads(path.read_text())['metrics']
    target_rows.append(row)
result['target_evaluation']=dict(rows=target_rows,macro={name:{metric:sum(r[name][metric] for r in target_rows)/8 for metric in ['iou','average_precision']} for name in ['baseline120','baseline480','full480']})
result['target_labels_used']=True
result['target_labels_used_for_training']=False
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(summary)
for r in rows:print(r['scene'],r['source_views'],{d:{a:round(v['iou']*100,3) for a,v in r['summary'][d].items()} for d in summary})

print(result['target_evaluation']['macro'])
