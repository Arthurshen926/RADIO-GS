"""Report both frozen LERF outputs for every query, without choosing winners."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('output/containment_scope_control')
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
ARMS = ['baseline', 'structural', 'semantic', 'assignment']


def read(path):
    return json.loads(path.read_text())


rows, queries = [], []
for scene in SCENES:
    arms, query_arms = {}, {}
    for arm in ARMS:
        folder = (Path('output/consolidated_response_control') / scene /
                  'fusion/native_protected/evaluation' if arm == 'baseline'
                  else Path('output/containment_scope_assignment') / scene / arm / 'evaluation' if arm == 'assignment'
                  else ROOT / scene / arm / 'evaluation')
        data = read(folder / 'evaluation.json')['metrics']['hierarchical_pipeline']
        arms[arm] = {k: data[k] for k in ['iou', 'ap', 'localization', 'observations']}
        arms[arm]['selected_iou'] = data['selected']['miou']
        grouped = defaultdict(lambda: defaultdict(list))
        for r in read(folder / 'query_results.json')['hierarchical_pipeline']:
            for metric in ['iou', 'ap']:
                grouped[r['query']][metric].append(r[metric])
        for r in data['selected']['objects']:
            grouped[r['query']]['selected_iou'].append(r['iou'])
        query_arms[arm] = {q: {m: sum(v) / len(v) for m, v in metrics.items()}
                           for q, metrics in grouped.items()}
    evidence = read(ROOT / scene / 'evidence/containment.json')
    assignment = read(Path('output/containment_scope_assignment') / scene / 'evidence/containment.json')
    evidence['counts'].update(assignment['counts'])
    rows.append(dict(scene=scene, arms=arms,
                     modified_query_views={a: sum(v) for a, v in evidence['counts'].items()}))
    for query, base in query_arms['baseline'].items():
        index = evidence['queries'].index(query)
        queries.append(dict(scene=scene, query=query,
                            modified_views={a: v[index] for a, v in evidence['counts'].items()},
                            arms={a: query_arms[a][query] for a in ARMS},
                            delta={a: {m: query_arms[a][query][m] - base[m] for m in base}
                                   for a in ARMS[1:]}))
summary = dict(scenes=rows, queries=queries,
               macro={a: {m: sum(r['arms'][a][m] for r in rows) / len(rows)
                          for m in ['iou', 'ap', 'localization', 'selected_iou']} for a in ARMS},
               development_only=True, positive_evidence_unchanged_in=['structural', 'semantic'],
               positive_evidence_augmented_in=['assignment'],
               geometry_unchanged=True, new_vfm_calls=0, target_rgb_used=False,
               same_map_four_query_validated=False, unseen_scene_validation=False,
               default_changed=False)
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['macro'], indent=2))
for r in rows:
    print(r['scene'], r['arms'])
for arm in ARMS[1:]:
    print(arm, 'largest semantic regressions')
    for q in sorted(queries, key=lambda q: q['delta'][arm]['iou'])[:5]:
        print(q['scene'], q['query'], q['delta'][arm])
