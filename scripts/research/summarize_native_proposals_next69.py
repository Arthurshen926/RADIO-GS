"""Full-scene recipe comparisons for subset consistency experiments."""
import json
from pathlib import Path
from statistics import mean
root=Path('output/native_proposals_next69')
bases={'retained_next63':Path('output/text_video_next63/decision'),'next67_exposed':Path('output/subset_consistency_next67/exposed')}
bases['next68_seeded']=Path('output/dual_observation_next68/seeded')
bases.update({p.name:p for p in root.iterdir() if p.is_dir() and p.name not in ('evidence','scope')})
controls={}
for arm,base in bases.items():
 scenes={}
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  p=base/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
  if not p.exists():continue
  m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];scenes[scene]={**{k:m[k] for k in ['iou','ap','localization']},'selected':m['selected']['miou']}
 controls[arm]=dict(scenes=scenes,scene_count=len(scenes),macro={k:mean(v[k] for v in scenes.values()) for k in ['iou','ap','localization','selected']} if len(scenes)==4 else None)
(root/'summary.json').write_text(json.dumps(dict(controls=controls,development_only=True,default_changed=False,selected_is_2d_projection=True),indent=2)+'\n')
print(json.dumps({a:v['macro'] for a,v in controls.items()},indent=2))
