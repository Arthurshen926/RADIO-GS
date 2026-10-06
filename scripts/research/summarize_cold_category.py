import json
from pathlib import Path
root=Path('output/cold_category');names=[r['scene'] for r in json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes']]
base={n:json.loads(Path(f'output/native_category_transfer/{n}/public/category/evaluation/report.json').read_text()) for n in names}
result=dict(scenes=names,arms={},scope='same8 development scenes, geometry, native fields and176 source teacher views',independent_test=False,public_sota_equivalence_verified=False)
for arm in ['categorical','source_exclusive','exclusive_scope','public']:
 reports={n:json.loads((root/n/arm/'evaluation/report.json').read_text()) for n in names}
 result['arms'][arm]=dict(macro={k:{s:sum(r['point_metrics'][k][s]['miou'] for r in reports.values())/len(names) for s in ['19','15','10']} for k in ['map','source']},delta_vs_retained={n:{s:reports[n]['point_metrics']['source'][s]['miou']-base[n]['point_metrics']['source'][s]['miou'] for s in ['19','15','10']} for n in names})
result['retained']={s:sum(r['point_metrics']['source'][s]['miou'] for r in base.values())/len(names) for s in ['19','15','10']}
result['public_replay_max_metric_difference']=max(abs(result['arms']['public']['macro'][k][s]-result['arms']['categorical']['macro'][k][s]) for k in ['map','source'] for s in ['19','15','10'])
result['decision']='cold categorical alternative deployable without historical scene parameters; retained system_v1 remains default because10class regresses; two source-learning/evidence variants not promoted'
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({a:r['macro'] for a,r in result['arms'].items()},indent=2))
