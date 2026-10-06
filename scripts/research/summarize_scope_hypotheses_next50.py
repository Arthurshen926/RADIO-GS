"""Summarize fixed-recipe scope experiments without selecting query winners."""
import json
from pathlib import Path
from statistics import mean
root=Path('output/scope_hypotheses_next50')
rows={}
for arm in ['latent','hard','crossview','corroborated']:
    scenes={}; changes=[]
    for scene in ['figurines','ramen','teatime','waldo_kitchen']:
        key='lerf_'+scene
        p=root/arm/key/'anchored/pipeline_calibrated/evaluation.json'
        if not p.exists():continue
        m=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
        baseline=json.loads((Path('output/decision_sets_next49/baseline')/key/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
        old={(r['frame'],r['query']):r['iou'] for r in baseline['selected']['objects']}
        by_query={}
        for r in m['selected']['objects']:
            by_query.setdefault(r['query'],[]).append(r['iou'])
            changes.append(dict(scene=scene,frame=r['frame'],query=r['query'],before=old[r['frame'],r['query']],after=r['iou'],delta=r['iou']-old[r['frame'],r['query']]))
        scenes[scene]=dict(iou=m['iou'],ap=m['ap'],selected=m['selected']['miou'],queries={q:mean(v) for q,v in by_query.items()})
    if scenes:rows[arm]=dict(scenes=scenes,macro={k:mean(v[k] for v in scenes.values()) for k in ['iou','ap','selected']},changes=changes)
summary=dict(development_only=True,default_changed=False,arms=rows)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({a:r['macro'] for a,r in rows.items()},indent=2))
