"""Summarize fixed native-region factorization controls without result selection."""
import json
from pathlib import Path
scenes=['figurines','ramen','teatime','waldo_kitchen']
roots={'baseline':'native_write_next75/all/joint_extent','leaf':'region_slots_next76/leaf/all/semantic_only','hierarchy':'region_slots_next76/hierarchy/all/semantic_only','leaf_joint':'region_slots_next76/leaf/all/joint_extent','hierarchy_joint':'region_slots_next76/hierarchy/all/joint_extent'}
rows={}
for variant,root in roots.items():
 rows[variant]={}
 for scene in scenes:
  p=Path('output')/root/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
  if not p.exists():continue
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
  rows[variant][scene]={k:m[k] for k in ['iou','ap','localization']}
  rows[variant][scene].update(selected_projection_iou=m['selected']['miou'],evaluation=str(p))
means={v:{k:sum(r[k] for r in rr.values())/4 for k in ['iou','ap','selected_projection_iou']} for v,rr in rows.items() if len(rr)==4}
p=Path('output/next76');p.mkdir(exist_ok=True);result=dict(per_scene=rows,complete_four_scene_macro=means,development_only=True)
(p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for v,rr in rows.items():print(v,{s:{k:round(r[k]*100,3) for k in ['iou','ap','selected_projection_iou']} for s,r in rr.items()})
query_deltas={}
for variant in ['leaf','hierarchy']:
 query_deltas[variant]={}
 for scene in ['figurines','ramen']:
  def path(v):return Path('output')/roots[v]/('lerf_'+scene)/'anchored/pipeline_calibrated/query_results.json'
  if not path(variant).exists():continue
  def mean_queries(p):
   grouped={}
   for r in json.loads(p.read_text())['hierarchical_pipeline']:grouped.setdefault(r['query'],[]).append(r)
   return {q:{k:sum(r[k] for r in rr)/len(rr) for k in ['iou','ap']} for q,rr in grouped.items()}
  before,after=mean_queries(path('baseline')),mean_queries(path(variant))
  query_deltas[variant][scene]=[dict(query=q,**r,delta_iou=r['iou']-before[q]['iou'],delta_ap=r['ap']-before[q]['ap']) for q,r in after.items()]
(p/'query_deltas.json').write_text(json.dumps(query_deltas,indent=2)+'\n')
