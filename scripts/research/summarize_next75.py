"""Read completed frozen-protocol results; no evaluation or query selection."""
import json
from pathlib import Path
scenes=['figurines','ramen','teatime','waldo_kitchen']
variants={'baseline':'expanded_semantic_next72/all/old_extent','fine':'native_write_next75/all/semantic_only','fine_joint':'native_write_next75/all/joint_extent','inverse':'native_inverse_next75/all/semantic_only'}
rows={}
for variant,base in variants.items():
 rows[variant]={}
 for scene in scenes:
  p=Path('output')/base/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
  if not p.exists():continue
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
  rows[variant][scene]={'iou':m['iou'],'ap':m['ap'],'selected_projection_iou':m['selected']['miou'],'localization':m['localization'],'evaluation':str(p)}
means={v:{k:sum(r[k] for r in scene_rows.values())/4 for k in ['iou','ap','selected_projection_iou','localization']} for v,scene_rows in rows.items() if len(scene_rows)==4}
result={'per_scene':rows,'complete_four_scene_macro':means,'all_results_development':True,'independent_3d_ground_truth':False}
out=Path('output/next75');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
# Describe failures and gains after the fixed recipe; never select a readout per query.
query_rows={}
for scene in scenes:
 old=Path('output')/variants['baseline']/('lerf_'+scene)/'anchored/pipeline_calibrated/query_results.json'
 new=Path('output')/variants['fine']/('lerf_'+scene)/'anchored/pipeline_calibrated/query_results.json'
 if not new.exists():continue
 def by_query(path):
  grouped={}
  for r in json.loads(path.read_text())['hierarchical_pipeline']:grouped.setdefault(r['query'],[]).append(r)
  return {q:{k:sum(r[k] for r in rr)/len(rr) for k in ['iou','ap']} for q,rr in grouped.items()}
 before,after=by_query(old),by_query(new)
 query_rows[scene]=sorted([dict(query=q,**v,delta_iou=v['iou']-before[q]['iou'],delta_ap=v['ap']-before[q]['ap']) for q,v in after.items()],key=lambda r:r['iou'])
(out/'query_deltas.json').write_text(json.dumps(query_rows,indent=2)+'\n')
