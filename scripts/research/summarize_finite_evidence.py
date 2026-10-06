"""Read fixed full-cohort metrics, including every regression and sparse case."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
p = argparse.ArgumentParser()
p.add_argument('--output', default='output/finite_source_evidence')
ROOT = Path(p.parse_args().output)


def read(folder):
    data = json.loads((folder / 'evaluation.json').read_text())['metrics']['hierarchical_pipeline']
    metrics = {m: data[m] for m in ['iou', 'ap', 'localization', 'observations']}
    metrics['selected_iou'] = data['selected']['miou']
    values = defaultdict(lambda: defaultdict(list))
    for row in json.loads((folder / 'query_results.json').read_text())['hierarchical_pipeline']:
        for m in ['iou', 'ap']:
            values[row['query']][m].append(row[m])
    for row in data['selected']['objects']:
        values[row['query']]['selected_iou'].append(row['iou'])
    return metrics, {q: {m: sum(x)/len(x) for m, x in fields.items()} for q, fields in values.items()}


rows, queries = [], []
for scene in SCENES:
    folders = {
        'original': Path('output/consolidated_response_control') / scene / 'fusion/native_protected/evaluation',
        'original_finite': ROOT / 'original' / scene / 'evaluation',
        'tiled': Path('output/tiled_source_evidence') / scene / 'global/evaluation',
        'tiled_finite': ROOT / 'tiled' / scene / 'evaluation',
    }
    for cohort in ['original', 'tiled']:
        control = ROOT / cohort / scene / 'readout_only/evaluation'
        if (control / 'evaluation.json').exists():
            folders[cohort + '_readout_only'] = control
    arms, per_query = {}, {}
    for name, folder in folders.items():
        arms[name], per_query[name] = read(folder)
    rows.append(dict(scene=scene, arms=arms))
    for query in per_query['original']:
        values = {name: per_query[name][query] for name in folders}
        queries.append(dict(scene=scene, query=query, arms=values, delta={
            cohort: {m: values[cohort + '_finite'][m] - values[cohort][m]
                     for m in values[cohort]} for cohort in ['original', 'tiled']}))
result = dict(scenes=rows, queries=queries, macro={a: {m: sum(row['arms'][a][m] for row in rows)/len(rows)
    for m in ['iou', 'ap', 'localization', 'selected_iou']} for a in folders},
    fixed_evidence_per_pair=True, fixed_geometry=True, new_teacher_calls=0,
    counts_are_not_independent_view_count=True, calibrated_identity_posterior=False,
    native_reference_strength=1, untouched_scene_validation=False, default_changed=False)
(ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['macro'], indent=2))
for q in queries:
    if q['scene'] == 'figurines' and q['query'] == 'waldo':
        print(json.dumps(q, indent=2))
for cohort in ['original', 'tiled']:
    for metric in ['iou', 'selected_iou']:
        ranked = sorted(queries, key=lambda q: q['delta'][cohort][metric])
        print(cohort, metric, 'worst', [(q['scene'], q['query'], q['delta'][cohort][metric]) for q in ranked[:5]])
        print(cohort, metric, 'best', [(q['scene'], q['query'], q['delta'][cohort][metric]) for q in ranked[-5:]])
