import json
from pathlib import Path
from statistics import mean
import torch
root=Path('output/text_video_next63');controls={}
for arm,base in [('decision_parent',Path('output/corroborated_set_next56/decision')),('hierarchy_parent',Path('output/factorized_scope_next59/hierarchy')),('decision',root/'decision'),('response',root/'response')]:
 scenes={}
 for p in base.glob('*/anchored/pipeline_calibrated/evaluation.json'):
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];scenes[p.parts[-4]]={**{k:m[k] for k in ['iou','ap','localization']},'selected':m['selected']['miou']}
 controls[arm]=dict(scenes=scenes,scene_count=len(scenes),macro={k:mean(v[k] for v in scenes.values()) for k in ['iou','ap','localization','selected']} if len(scenes)==4 else None)
tracks={}
for d in (root/'tracks').iterdir():
 rows=[]
 for p in d.glob('*.pt'):
  t=torch.load(p,map_location='cpu',weights_only=False);rows.append(dict(query=t['query'],seed_frame=t['seed_frame'],frames=len(t['frames']),present=sum(r['present'] for r in t['frames'].values())))
 tracks[d.name]=rows
(root/'summary.json').write_text(json.dumps(dict(controls=controls,tracks=tracks,development_only=True,default_changed=False),indent=2)+'\n')
print(json.dumps({a:v['macro'] for a,v in controls.items()},indent=2))
