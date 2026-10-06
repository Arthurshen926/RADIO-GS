"""Validate a complete component recipe, with no per-query outcome selection."""
import json
from pathlib import Path

import numpy as np

from scripts.research.fit_query_conditioned_membership import ROOT, write
from scripts.research.summarize_typed_binding import BASE, compare_arrays, compact, effects


def main():
    root = ROOT / 'image_convergence';system = root / 'system'
    current = json.loads((system / 'uqis/identity_scope/result.json').read_text())
    previous = json.loads((BASE / 'uqis/identity_scope/result.json').read_text())
    summary = json.loads((system / 'summary.json').read_text())
    exact = compare_arrays(system, current)
    exact.update(text_ranks_exact=True, text_sets_exact=True, image_matches_previous_component=True)
    bridge = Path('output/typed_referent_binding_20261006/native_bridge/system')
    for row in current['explicit_selection']['per_query']:
        if row['modality'] not in ['text', 'image']:continue
        scene, qid = row['scene_id'], row['query_id'];parent = BASE if row['modality'] == 'text' else bridge
        for directory in ['predictions', 'selections']:
            path = Path('uqis') / scene / 'query/registered/identity_scope' / directory / (qid + '.npy')
            same = np.array_equal(np.load(system / path), np.load(parent / path))
            key = ('text_ranks_exact' if directory == 'predictions' else 'text_sets_exact') if row['modality'] == 'text' else 'image_matches_previous_component'
            exact[key] &= same
    delta = effects(current, previous);metrics = dict(v9=compact(previous), candidate=compact(current))
    old_public = json.loads(Path('output/real_direct_membership_20261006/final_validation.json').read_text())['macro']
    public = {name: summary['macro'][name] for name in ['lerf', 'nvos', 'scannet']}
    public_difference = {name: {key: value - old_public[name][key] for key, value in row.items()} for name, row in public.items()}
    base_recipe = json.loads(Path('configs/pipeline/system_v9.json').read_text())
    new_recipe = json.loads((root / 'recipe.json').read_text())
    public_jobs_same = ([j for j in base_recipe['jobs'] if j['benchmark'] != 'uqis'] ==
                        [j for j in new_recipe['jobs'] if j['benchmark'] != 'uqis'])
    success = lambda d: {(r['scene_id'], r['instance_id']) for r in d['explicit_selection']['paired_query_robustness']['per_target']
        if r['all_four_iou_at_least_050']}
    old, new = success(previous), success(current)
    gate = (delta['image']['mean'] >= .02 and delta['image']['bootstrap95'][0] > 0 and len(new) >= len(old) and
        all(exact[key] for key in ['native_exact', 'point_rank_exact', 'point_selection_exact', 'text_ranks_exact', 'text_sets_exact', 'image_matches_previous_component']))
    report = dict(status='complete', completed_jobs=summary['completed'], metrics=metrics, effects=delta, unchanged_outputs=exact,
        all_four_transitions=dict(old=len(old), new=len(new), gained=sorted(new - old), lost=sorted(old - new)),
        public_metrics=public, public_difference=public_difference, public_jobs_identical=public_jobs_same,
        image_component_gate=bool(gate), all_predictions_before_scoring=summary['all_predictions_before_scoring'],
        historical_query_predictions_read=summary['historical_query_predictions_read'],
        selection_basis='component chosen from prior opened development results; fresh complete replay, not an independent new gain',
        independent_final_test=False, new_query_supervised_models_included=False)
    write(root / 'comparison.json', report);print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
