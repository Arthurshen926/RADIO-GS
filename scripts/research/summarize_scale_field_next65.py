import json
from pathlib import Path
from statistics import mean
root=Path('output/scale_field_next65');controls={}
for arm,base in [('mixture_native',Path('output/native_rebuild_next54/mixture')),('retained',Path('output/text_video_next63/decision')),('native',root/'native'),('response',root/'response'),('coherent',root/'coherent')]:
 scenes={}
 for p in base.glob('*/anchored/pipeline_calibrated/evaluation.json'):
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];scenes[p.parts[-4]]={**{k:m[k] for k in ['iou','ap','localization']},'selected':m['selected']['miou']}
 controls[arm]=dict(scenes=scenes,scene_count=len(scenes),macro={k:mean(v[k] for v in scenes.values()) for k in ['iou','ap','localization','selected']} if len(scenes)==4 else None)
size={d.name:sum(p.stat().st_size for p in d.glob('*_field.pt') if p.name!='native_field.pt') for d in (root/'native').iterdir()}
(root/'summary.json').write_text(json.dumps(dict(controls=controls,slot_artifact_bytes=size,development_only=True,default_changed=False),indent=2)+'\n')
print(json.dumps({a:v['macro'] for a,v in controls.items()},indent=2))
