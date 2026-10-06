"""Aggregate all complete matched scene runs, never select per-scene winners."""
import json
from pathlib import Path
root=Path('output/native_category_transfer')
rows=json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes']
names=[r['scene'] for r in rows]
old=json.loads(Path('output/scannet_runtime_integration/summary.json').read_text())['scenes']
raw=json.loads(Path('output/scannet_native_completion/summary.json').read_text())
result=dict(scenes=names,arms={},reference=raw['legacy_eight_scene_macro'],raw_native=raw['matched_eight_scene_macro'],independent_test=False,official_protocol_equivalence_verified=False)
for arm in ['transferred','source_refit','scope_transfer','joint_category','public/category']:
 reports={n:json.loads((root/n/arm/'evaluation/report.json').read_text()) for n in names}
 macro={k:{s:sum(r['point_metrics'][k][s]['miou'] for r in reports.values())/len(names) for s in ['19','15','10']} for k in ['map','source']}
 delta={n:reports[n]['point_metrics']['source']['19']['miou']-old[n]['point_metrics']['source']['19']['miou'] for n in names}
 result['arms'][arm]=dict(macro=macro,source_19_delta_vs_legacy=delta,improved_scenes=sum(x>0 for x in delta.values()),regressed_scenes=sum(x<0 for x in delta.values()))
result['public_metric_max_difference']=max(abs(result['arms']['transferred']['macro'][k][s]-result['arms']['public/category']['macro'][k][s]) for k in ['map','source'] for s in ['19','15','10'])
result['decision']='retain native map + transferred source-trained calibration + existing source solver as eight-scene development candidate; do not promote refit, scope, or joint categorical alternatives'
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for arm,r in result['arms'].items():print(arm,r['improved_scenes'],{k:{s:100*v for s,v in d.items()} for k,d in r['macro'].items()})
for n in names:print(n,round(100*old[n]['point_metrics']['source']['19']['miou'],4),round(100*(old[n]['point_metrics']['source']['19']['miou']+result['arms']['transferred']['source_19_delta_vs_legacy'][n]),4))
