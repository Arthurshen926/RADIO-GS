"""Complete semantic/selected metrics, all-query deltas and verifier costs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',default='output/source_referent_verification')
p.add_argument('--policies',nargs='+',choices=['veto','affirmative','contradiction'],default=['veto','affirmative']);a=p.parse_args();root=Path(a.root)


def read(folder):
    m=json.loads((folder/'evaluation.json').read_text())['metrics']['hierarchical_pipeline']
    metrics={k:m[k] for k in ['iou','ap','localization','observations']};metrics['selected_iou']=m['selected']['miou']
    per=defaultdict(lambda:defaultdict(list))
    for r in json.loads((folder/'query_results.json').read_text())['hierarchical_pipeline']:
        for k in ['iou','ap']:per[r['query']][k].append(r[k])
    for r in m['selected']['objects']:per[r['query']]['selected_iou'].append(r['iou'])
    return metrics,{q:{k:sum(v)/len(v) for k,v in d.items()} for q,d in per.items()}


rows,queries=[],[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
    arms,query_arms={},{}
    for cohort in ['original','tiled']:
        path=(Path('output/consolidated_response_control')/scene/'fusion/native_protected/evaluation' if cohort=='original'
              else Path('output/tiled_source_evidence')/scene/'global/evaluation')
        arms[cohort],query_arms[cohort]=read(path)
        for arm in a.policies:
            name=f'{cohort}_{arm}';arms[name],query_arms[name]=read(root/arm/cohort/scene/'evaluation')
    rows.append(dict(scene=scene,arms=arms))
    for q in query_arms['original']:
        values={k:v[q] for k,v in query_arms.items()}
        queries.append(dict(scene=scene,query=q,arms=values,delta={f'{cohort}_{arm}':
            {k:values[f'{cohort}_{arm}'][k]-values[cohort][k] for k in values[cohort]}
            for cohort in ['original','tiled'] for arm in a.policies}))
result=dict(scenes=rows,queries=queries,macro={arm:{k:sum(r['arms'][arm][k] for r in rows)/len(rows)
    for k in ['iou','ap','localization','selected_iou']} for arm in rows[0]['arms']},
    verification=json.loads((root/'verification.json').read_text()),same_geometry=True,
    same_solver_and_thresholds=True,extra_semantic_teacher=True,no_generated_query_aliases=True,
    independent_test=False,selected_metric_is_independent_3d_iou=False)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['macro'],indent=2))
for q in queries:
    if (q['scene'],q['query']) in [('figurines','waldo'),('teatime','paper napkin')]:print(json.dumps(q,indent=2))
