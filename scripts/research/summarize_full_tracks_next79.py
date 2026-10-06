"""Full-query temporal-source controls; pilots excluded from scene comparison."""
import json
from pathlib import Path
roots={'baseline':'output/native_write_next75/all/joint_extent','full_sequence_old_fit':'output/full_text_decode_next79/complete/old','full_sequence_all_fit':'output/full_text_decode_next79/complete/all'};rows={};queries={}
for variant,root in roots.items():
 rows[variant]={};queries[variant]={}
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  p=Path(root)/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
  if not p.exists():continue
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];rows[variant][scene]=dict(iou=m['iou'],ap=m['ap'],selected_projection_iou=m['selected']['miou'],evaluation=str(p));q={}
  for r in m['selected']['objects']:q.setdefault(r['query'],[]).append(r['iou'])
  queries[variant][scene]={k:sum(v)/len(v) for k,v in q.items()}
means={v:{k:sum(r[k] for r in rr.values())/4 for k in ['iou','ap','selected_projection_iou']} for v,rr in rows.items() if len(rr)==4}
out=Path('output/next79');out.mkdir(exist_ok=True);(out/'full_tracks_summary.json').write_text(json.dumps(dict(per_scene=rows,complete_four_scene_macro=means,query_selected_iou=queries,development_only=True,pilots_included=False),indent=2)+'\n')
for v,rr in rows.items():print(v,{s:round(r['selected_projection_iou']*100,4) for s,r in rr.items()})

p=Path('output/full_text_decode_next79/matched293/old/lerf_figurines/anchored/pipeline_calibrated/evaluation.json')
if p.exists():
 summary=json.loads((out/'full_tracks_summary.json').read_text());m=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
 summary['matched_step_control']=dict(scene='figurines',source_cohort='old',tracking='full source',steps=293,selected_projection_iou=m['selected']['miou'],evaluation=str(p));(out/'full_tracks_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
