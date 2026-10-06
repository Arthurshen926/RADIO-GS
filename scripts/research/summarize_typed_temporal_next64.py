import json
from pathlib import Path
from statistics import mean
root=Path('output/typed_temporal_next64');controls={}
for arm,base in [('retained',Path('output/text_video_next63/decision')),('union',Path('output/text_video_next63/response')),('abstain',root/'abstain'),('positive',root/'positive')]:
 scenes={}
 for p in base.glob('*/anchored/pipeline_calibrated/evaluation.json'):
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];scenes[p.parts[-4]]={**{k:m[k] for k in ['iou','ap','localization']},'selected':m['selected']['miou']}
 controls[arm]=dict(scenes=scenes,scene_count=len(scenes),macro={k:mean(v[k] for v in scenes.values()) for k in ['iou','ap','localization','selected']} if len(scenes)==4 else None)
(root/'summary.json').write_text(json.dumps(dict(controls=controls,development_only=True,default_changed=False),indent=2)+'\n')
print(json.dumps({a:v['macro'] for a,v in controls.items()},indent=2))
