"""Complete-cohort evidence, cost and dual-output comparison; no winner mixing."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('output/tiled_source_evidence')
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
ARMS = ['baseline', 'global', 'bounded']
rows, queries = [], []
for scene in SCENES:
    metrics, query_arms = {}, {}
    for arm in ARMS:
        folder = (Path('output/consolidated_response_control') / scene / 'fusion/native_protected/evaluation'
                  if arm == 'baseline' else ROOT / scene / arm / 'evaluation')
        data = json.loads((folder / 'evaluation.json').read_text())['metrics']['hierarchical_pipeline']
        metrics[arm] = {m: data[m] for m in ['iou', 'ap', 'localization', 'observations']}
        metrics[arm]['selected_iou'] = data['selected']['miou']
        grouped = defaultdict(lambda: defaultdict(list))
        for r in json.loads((folder / 'query_results.json').read_text())['hierarchical_pipeline']:
            for m in ['iou', 'ap']:
                grouped[r['query']][m].append(r[m])
        for r in data['selected']['objects']:
            grouped[r['query']]['selected_iou'].append(r['iou'])
        query_arms[arm] = {q: {m: sum(v) / len(v) for m, v in values.items()} for q, values in grouped.items()}
    coverage = json.loads((ROOT / scene / 'combined/coverage.json').read_text())
    bounded = json.loads((ROOT / scene / 'bounded_evidence/bounds.json').read_text())
    teachers = [json.loads(p.read_text()) for p in sorted((ROOT / scene / 'tiles').glob('provider_*.json'))]
    budget = {k: sum(p[k] for p in teachers) for k in ['seconds', 'image_encoder_calls', 'text_grounding_calls']}
    budget['source_frames'] = sum(len(p['source_frames']) for p in teachers)
    rows.append(dict(scene=scene, arms=metrics, additional_teacher_cost=budget,
                     bounded_query_views=sum(bounded['bounded_views'])))
    for qi, q in enumerate(coverage['queries']):
        if q['query'] not in query_arms['baseline']:
            continue
        result = {a: query_arms[a][q['query']] for a in ARMS}
        queries.append(dict(scene=scene, **q, bounded_views=bounded['bounded_views'][qi], arms=result,
            delta={a: {m: result[a][m] - result['baseline'][m] for m in result[a]} for a in ARMS[1:]}))
summary = dict(scenes=rows, queries=queries, macro={a: {m: sum(r['arms'][a][m] for r in rows) / len(rows)
    for m in ['iou', 'ap', 'localization', 'selected_iou']} for a in ARMS},
    baseline_teacher_budget_matched=False, global_vs_bounded_teacher_budget_matched=True,
    geometry_changed=False, native_feature_field_changed=False, unseen_scene_validation=False,
    target_rgb_used=False, public_sota=False, default_changed=False)
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['macro'], indent=2))
for r in rows:
    print(r['scene'], r['arms'], r['additional_teacher_cost'])
for arm in ARMS[1:]:
    print(arm, 'largest semantic gains / regressions')
    ordered = sorted(queries, key=lambda q: q['delta'][arm]['iou'])
    for q in ordered[:5] + ordered[-5:]:
        print(q['scene'], q['query'], q['delta'][arm])
