"""Completed frozen-protocol results for source-only region correspondence."""
import json
from pathlib import Path
scenes=['figurines','ramen','teatime','waldo_kitchen'];roots={'baseline':'native_write_next75/all/joint_extent','correspondence':'correspondence_write_next77/all/semantic_only','joint':'correspondence_write_next77/all/joint_extent'};rows={}
for variant,root in roots.items():
 rows[variant]={}
 for scene in scenes:
  p=Path('output')/root/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
  if not p.exists():continue
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];rows[variant][scene]={k:m[k] for k in ['iou','ap','localization']};rows[variant][scene].update(selected_projection_iou=m['selected']['miou'],evaluation=str(p))
means={v:{k:sum(r[k] for r in rr.values())/4 for k in ['iou','ap','selected_projection_iou']} for v,rr in rows.items() if len(rr)==4}
out=Path('output/next77');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(dict(per_scene=rows,complete_four_scene_macro=means,development_only=True),indent=2)+'\n')
for v,rr in rows.items():print(v,{s:{k:round(r[k]*100,3) for k in ['iou','ap','selected_projection_iou']} for s,r in rr.items()})
query_rows={}
for scene in scenes:
 def getpath(variant):return Path('output')/roots[variant]/('lerf_'+scene)/'anchored/pipeline_calibrated/query_results.json'
 if not getpath('correspondence').exists():continue
 def mean_queries(path):
  grouped={}
  for r in json.loads(path.read_text())['hierarchical_pipeline']:grouped.setdefault(r['query'],[]).append(r)
  return {q:{k:sum(r[k] for r in rr)/len(rr) for k in ['iou','ap']} for q,rr in grouped.items()}
 before,after=mean_queries(getpath('baseline')),mean_queries(getpath('correspondence'))
 query_rows[scene]=[dict(query=q,**r,delta_iou=r['iou']-before[q]['iou'],delta_ap=r['ap']-before[q]['ap']) for q,r in after.items()]
(out/'query_deltas.json').write_text(json.dumps(query_rows,indent=2)+'\n')
