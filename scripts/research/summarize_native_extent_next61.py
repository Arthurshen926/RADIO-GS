"""Aggregate matched development evaluations and source extent decisions."""
import json
from pathlib import Path
from statistics import mean
import torch
root=Path('output/native_extent_next61')
controls={}
for arm,base in [('parent',Path('output/factorized_scope_next59/hierarchy')),('native',root/'native'),('source',root/'source')]:
    scenes={}
    for p in base.glob('*/anchored/pipeline_calibrated/evaluation.json'):
        m=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
        scenes[p.parts[-4]]={**{k:m[k] for k in ['iou','ap','localization']},'selected':m['selected']['miou']}
    controls[arm]=dict(scenes=scenes,macro={k:mean(v[k] for v in scenes.values()) for k in ['iou','ap','localization','selected']} if scenes else {})
evidence={}
for scene in (root/'source_evidence').iterdir():
    queries={}
    for p in scene.glob('*.pt'):
        d=torch.load(p,map_location='cpu',weights_only=False)
        for text,choices,margins in zip(d['texts'],d['choices'],d['margins']):
            r=queries.setdefault(text,dict(instances=0,enclosure=0,accepted=0,alternative_instances=0))
            for choice,margin in zip(choices,margins):
                r['instances']+=1;r['enclosure']+=int(choice>0);r['accepted']+=int(margin[choice]>0);r['alternative_instances']+=int(len(margin)>1)
    evidence[scene.name]=queries
result=dict(controls=controls,source_evidence=evidence,development_only=True,default_changed=False)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({a:d['macro'] for a,d in controls.items()},indent=2))
