"""Report complete scene cohorts without selecting per-query winners."""
import json
from pathlib import Path
from statistics import mean
root=Path('output/instance_scope_next62')
controls={}
for arm,base in [('parent',Path('output/factorized_scope_next59/hierarchy')),('graph',root/'graph'),('identity',root/'identity')]:
    scenes={}
    for p in base.glob('*/anchored/pipeline_calibrated/evaluation.json'):
        m=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
        scenes[p.parts[-4]]={**{k:m[k] for k in ['iou','ap','localization']},'selected':m['selected']['miou']}
    controls[arm]=dict(scenes=scenes,scene_count=len(scenes),macro={k:mean(v[k] for v in scenes.values()) for k in ['iou','ap','localization','selected']} if len(scenes)==4 else None)
(root/'summary.json').write_text(json.dumps(dict(controls=controls,development_only=True,default_changed=False),indent=2)+'\n')
print(json.dumps(controls,indent=2))
