"""Close the two bounded mechanisms with actual selection and identity errors."""
import json
from pathlib import Path

import numpy as np

from scripts.research.prepare_native_phrase_binding import ROOT, write


def error_types(fit):
    rows = []
    for scene in fit['validation']['scenes']:
        name = scene['scene'];targets = np.load(ROOT / 'entities' / (name + '.npz'))
        annotation = json.loads((Path('output/referent_grounding_20261005/raw') / name /
            (name + '.aggregation.json')).read_text())['segGroups']
        classes = {int(g['objectId']) + 1: g['label'] for g in annotation}
        queries = json.loads((ROOT / 'language' / (name + '.json')).read_text())
        instances = targets['instances'];purity = targets['purity']
        selected = {}
        for key, values in scene['conditions'].items():
            counts = dict(correct_instance=0, wrong_instance_same_class=0, wrong_class=0, mixed_region=0)
            for q, seed in zip(queries, values['seeds']):
                column = int(purity[seed].argmax());identity = int(instances[column]);target = q['raw_object_id'] + 1
                if purity[seed, column] <= .5:
                    kind = 'mixed_region'
                elif identity == target:
                    kind = 'correct_instance'
                elif classes.get(identity) == classes[target]:
                    kind = 'wrong_instance_same_class'
                else:
                    kind = 'wrong_class'
                counts[kind] += 1
            selected[key] = counts
        rows.append(dict(scene=name, queries=len(queries), conditions=selected))
    totals = {condition: {kind: sum(r['conditions'][condition][kind] for r in rows)
        for kind in rows[0]['conditions'][condition]} for condition in rows[0]['conditions']}
    return dict(scenes=rows, totals=totals, note='Seed-region majority identity; this is not full selected-set purity.')


def main():
    names = ['native', 'uniform', 'native_instance', 'uniform_instance']
    fits = {name: json.loads((ROOT / name / 'fit.json').read_text()) for name in names}
    candidate = fits['native_instance']['validation']['conditions']
    controls = [fits[name]['validation']['conditions']['original']['iou'] for name in ['native', 'uniform_instance']]
    gate = dict(candidate=candidate['original']['iou'], native_round1=controls[0], matched_uniform=controls[1],
        uniform_intervention=candidate['uniform']['iou'], query_mean=candidate['query_mean']['iou'],
        reversed_witness=candidate['reversed']['iou'],
        passed=(candidate['original']['iou'] > max(controls) and
            candidate['original']['iou'] >= candidate['uniform']['iou'] + .005 and
            candidate['original']['iou'] > candidate['query_mean']['iou']),
        benchmark_metrics_used=False, tuning_rounds=2, additional_search=False)
    previous = json.loads(Path('output/real_referent_grounding_20261005/local_scope/fit.json').read_text())
    old = {row['scene']: row['iou'] for row in previous['validation']['scenes']}
    deltas = {}
    for name in names:
        values = np.array([r['conditions']['original']['iou'] - old[r['scene']] for r in fits[name]['validation']['scenes']])
        rng = np.random.default_rng(20261006)
        interval = np.quantile(values[rng.integers(0, len(values), (20000, len(values)))].mean(1), [.025, .975])
        deltas[name] = dict(mean=float(values.mean()), scene_deltas=values.tolist(), improved=int((values > 0).sum()),
            bootstrap95=interval.tolist(), rooms=len(values), uncertainty='only four repeatedly opened development rooms')
    report = dict(status='complete', default='system_v9', fits=names, benchmark_jobs=0,
        round1_gate=json.loads((ROOT / 'development_gate.json').read_text()), round2_gate=gate,
        metrics={name: {**{k: fits[name][k] for k in ['condition', 'selected_epoch', 'parameter_count', 'seconds',
            'validation', 'calibration']}, 'instance_supervised_training_queries':
            fits[name]['instance_supervised_training_queries'] if name.endswith('_instance') else 0} for name in names},
        errors={name: error_types(fits[name]) for name in names}, improvement_over_previous_local=deltas,
        role_extraction=json.loads((ROOT / 'roles/fit.json').read_text()),
        phrase_recognition=json.loads((ROOT / 'phrase_diagnosis.json').read_text()),
        privileged_interventions=json.loads((ROOT / 'privileged_interventions.json').read_text()),
        native_map_changed=False, deployed_query_permissions_changed=False, official_sota_claim=False)
    write(ROOT / 'final_comparison.json', report)
    print(json.dumps(dict(gate=gate, metrics={name: fits[name]['validation']['conditions'] for name in names},
        errors={name: report['errors'][name]['totals']['original'] for name in names}), indent=2), flush=True)


if __name__ == '__main__':
    main()
