"""Attribute retained-query information separately from corrected supervision."""
import json
from pathlib import Path

import numpy as np

from scripts.research.fit_query_conditioned_membership import ROOT, write
from scripts.research.summarize_typed_binding import BASE, compact, effects, compare_arrays


def main():
    fit = json.loads((ROOT / 'fit.json').read_text())
    anchor_free = json.loads((ROOT / 'anchor_free_fit.json').read_text())
    exact = json.loads((ROOT / 'exact_development.json').read_text())
    development = fit['development'];groups = {}
    for correct in [False, True]:
        rooms = []
        for scene in development:
            rows = [r for r in scene['rows'] if r['anchor_correct'] == correct]
            if not rows:continue
            rooms.append(dict(scene=scene['scene'], queries=len(rows),
                metrics={name: np.mean([r['conditions'][name][0] for r in rows], axis=0).tolist()
                         for name in ['anchor_only', 'query_evidence', 'query_mean']},
                old=np.mean([r['old_anchor'] for r in rows], axis=0).tolist()))
        groups[str(correct)] = dict(rooms=rooms, queries=sum(r['queries'] for r in rooms),
            scene_macro={name: np.mean([r['metrics'][name] for r in rooms], axis=0).tolist()
                         for name in ['anchor_only', 'query_evidence', 'query_mean']},
            old=np.mean([r['old'] for r in rooms], axis=0).tolist())
    deltas = np.array([r['conditions']['query_evidence'][0][0] - r['conditions']['anchor_only'][0][0] for r in development])
    rng = np.random.default_rng(20261006)
    interval = np.quantile(deltas[rng.integers(0, len(deltas), (20000, len(deltas)))].mean(1), [.025, .975])
    report = dict(status='complete', default_changed=False, development_gate=fit['development_gate'],
        training_queries=fit['training_queries'], training_samples=fit['training_samples'],
        thresholds=fit['thresholds'], development_macro=fit['scene_macro'], old_anchor=fit['old_anchor_macro'],
        query_information_effect=dict(mean=float(deltas.mean()), per_scene=deltas.tolist(), bootstrap95=interval.tolist()),
        estimated_anchor_groups=groups, benchmark_jobs=0, independent_final_test=False)
    report['anchor_free'] = {key: anchor_free[key] for key in ['threshold', 'scene_macro', 'delta_to_old_bridge',
        'per_scene_delta', 'development_gate', 'no_estimated_identity_feature', 'training_samples']}
    report['fits'] = 3
    report['exact_calibration'] = exact['calibration']
    report['exact_development'] = exact['scene_macro']
    report['final_development_gates'] = exact['development_gates']
    final_groups = {}
    for correct in [False, True]:
        subsets = [[r for r in exact['rows'] if r['scene'] == scene['scene'] and r['anchor_correct'] == correct]
                   for scene in development]
        subsets = [rows for rows in subsets if rows]
        final_groups[str(correct)] = dict(queries=sum(map(len, subsets)), scene_macro={name:
            np.mean([np.mean([r['metrics'][name] for r in rows], axis=0) for rows in subsets], axis=0).tolist()
            for name in ['anchor_only', 'query_evidence', 'query_only']})
    report['exact_estimated_anchor_groups'] = final_groups
    report['anchor_label_diagnosis'] = json.loads((ROOT / 'anchor_label_diagnosis.json').read_text())
    if any(exact['development_gates'].values()):
        primary = 'query_evidence' if exact['development_gates']['query_evidence'] else 'query_only'
        names = [primary, 'anchor_only']
        systems = {name: json.loads((ROOT / name / 'system/summary.json').read_text()) for name in names}
        evaluations = {name: json.loads((ROOT / name / 'system/uqis/identity_scope/result.json').read_text()) for name in names}
        evaluations['v9'] = json.loads((BASE / 'uqis/identity_scope/result.json').read_text())
        evaluations['previous_bridge'] = json.loads(Path('output/typed_referent_binding_20261006/native_bridge/system/uqis/identity_scope/result.json').read_text())
        exact = {name: compare_arrays(ROOT / name / 'system', evaluations[name]) for name in names}
        change = effects(evaluations[primary], evaluations['v9']);metrics = {name: compact(d) for name, d in evaluations.items()}
        promoted = all(change[m]['mean'] >= .02 and change[m]['bootstrap95'][0] > 0 for m in ['text', 'image'])
        promoted &= metrics[primary]['all_four_success_count'] >= metrics['v9']['all_four_success_count']
        promoted &= all(exact[primary][key] for key in ['native_exact', 'point_rank_exact', 'point_selection_exact'])
        report.update(benchmark_jobs=sum(d['completed'] for d in systems.values()), benchmark_metrics=metrics,
            effects=change, information_control=effects(evaluations[primary], evaluations['anchor_only']),
            previous_bridge_effects=effects(evaluations[primary], evaluations['previous_bridge']),
            unchanged_outputs=exact, primary_public_metrics={b: systems[primary]['macro'][b] for b in ['lerf', 'nvos', 'scannet']},
            uqis_promotion_gate=bool(promoted))
    write(ROOT / 'comparison.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['estimated_anchor_groups']}, indent=2))


if __name__ == '__main__':main()
