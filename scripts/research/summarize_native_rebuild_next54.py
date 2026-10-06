"""Keep native-field construction, calibration and scope controls separate."""
import json
from pathlib import Path
from statistics import mean
root=Path('output/native_rebuild_next54');summary={}
for mode in ['mixture','finest','mixture_balanced','finest_balanced','mixture_scope','mixture_querycal','mixture_masscal','mixture_ranking','finest_ranking']:
 rows={}
 for p in (root/mode).glob('*/anchored/pipeline_calibrated/evaluation.json'):
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];rows[p.parts[3]]={k:m[k] for k in ['iou','ap','localization']};rows[p.parts[3]]['selected']=m['selected']['miou']
 if mode.endswith('_ranking'):rows={k:dict(ap=v['ap']) for k,v in rows.items()}
 if rows:summary[mode]=dict(scenes=rows,scene_count=len(rows),macro={k:mean(v[k] for v in rows.values()) for k in (['ap'] if mode.endswith('_ranking') else ['iou','ap','localization','selected'])},ranking_only=mode.endswith('_ranking'))
selected_rows={}
for p in Path('output/native_selection_next54/fit').glob('*/anchored/pipeline_calibrated/evaluation.json'):
 m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];selected_rows[p.parts[3]]=m['selected']['miou']
if selected_rows:summary['selection_refit']=dict(scenes=selected_rows,scene_count=len(selected_rows),macro=dict(selected=mean(selected_rows.values())),ranking_only=False)
(root/'summary.json').write_text(json.dumps(dict(controls=summary,development_only=True,default_changed=False,four_teacher_retraining=False,selection_unchanged_except_rejected_refit=True),indent=2)+'\n')
print(json.dumps({m:dict(scene_count=v['scene_count'],macro=v['macro']) for m,v in summary.items()},indent=2))
