"""Compare native proxy, actual-mask and context identity on complete cohorts."""
import json
from collections import defaultdict
from pathlib import Path

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
ARMS = {'baseline': 'output/consolidated_response_control',
        'direct': 'output/direct_source_identity', 'context': 'output/direct_source_context'}
rows, queries = [], []
for scene in SCENES:
    arms, query_arms, costs = {}, {}, {}
    for arm, path in ARMS.items():
        root = Path(path) / scene
        folder = root / ('fusion/native_protected/evaluation' if arm == 'baseline' else 'evaluation')
        data = json.loads((folder / 'evaluation.json').read_text())['metrics']['hierarchical_pipeline']
        arms[arm] = {k: data[k] for k in ['iou', 'ap', 'localization', 'observations']}
        arms[arm]['selected_iou'] = data['selected']['miou']
        grouped = defaultdict(lambda: defaultdict(list))
        for r in json.loads((folder / 'query_results.json').read_text())['hierarchical_pipeline']:
            for m in ['iou', 'ap']:
                grouped[r['query']][m].append(r[m])
        for r in data['selected']['objects']:
            grouped[r['query']]['selected_iou'].append(r['iou'])
        query_arms[arm] = {q: {m: sum(v) / len(v) for m, v in fields.items()} for q, fields in grouped.items()}
        if arm != 'baseline':
            record = json.loads((root / 'native/identity.json').read_text())
            costs[arm] = {k: record[k] for k in ['instances', 'encoded_instances', 'empty_masks_unknown', 'seconds', 'crop']}
    rows.append(dict(scene=scene, arms=arms, cost=costs))
    for q in query_arms['baseline']:
        values = {arm: query_arms[arm][q] for arm in ARMS}
        queries.append(dict(scene=scene, query=q, arms=values, delta={
            a: {m: values[a][m] - values['baseline'][m] for m in values[a]} for a in ['direct', 'context']}))
result = dict(scenes=rows, queries=queries, macro={a: {m: sum(r['arms'][a][m] for r in rows) / 4
    for m in ['iou', 'ap', 'localization', 'selected_iou']} for a in ARMS},
    same_detections=True, same_native_encoder=True, same_identity_density_rule=True,
    direct_context_encoder_budget_matched=True, baseline_extra_encoding_budget_matched=False,
    foreground_scope_never_expanded=True, unseen_scene_validation=False, default_changed=False)
Path('output/direct_source_identity/summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['macro'], indent=2))
for arm in ['direct', 'context']:
    print(arm, 'new encoded crops', sum(r['cost'][arm]['encoded_instances'] for r in rows),
          'encoder worker seconds excluding model load', sum(r['cost'][arm]['seconds'] for r in rows))
