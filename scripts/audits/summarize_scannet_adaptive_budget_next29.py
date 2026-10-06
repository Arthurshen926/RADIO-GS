"""Summarize the predeclared adaptive source budget with all eight scenes."""
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    root = Path('output/scannet_expanded_budget_next29')
    plan_path = root/'adaptive_plan.json'
    plan = json.loads(plan_path.read_text())
    registration = json.loads((root/'evaluation_registration.json').read_text())
    assert registration['plan_sha256'] == sha256_file(plan_path)
    assert registration['runtime_sha256'] == sha256_file('radio_gs/querying/source_accepted_category.py')
    assert registration['evaluator_sha256'] == sha256_file('scripts/audits/evaluate_scannet_adaptive_budget_next29.py')
    rows = {}
    for scene in plan['scenes']:
        previous_path = Path('output/source_accepted_runtime_next29')/scene/'report.json'
        previous = json.loads(previous_path.read_text())
        assert previous['complete'] and not any(previous['prediction_mismatches'].values())
        if scene in plan['rebuild_scenes']:
            path = root/scene/'evaluation/report.json'
            candidate = json.loads(path.read_text())
            assert candidate['complete_field_rebuilt'] and candidate['source_view_budget'] == 32
            assert candidate['gt_opened_after_prediction']
            assert candidate['adaptive_plan_sha256'] == sha256_file(plan_path)
            assert candidate['implementation_sha256'] == registration['evaluator_sha256']
            assert candidate['metric_implementation_sha256'] == registration['metric_sha256']
            assert candidate['geometry_sha256'] == previous['geometry_sha256']
            frozen_root = (Path('output/scannet_coverage_native_probe_20260919')
                           if scene in ('scene0000_00', 'scene0062_00')
                           else Path('output/scannet_coverage_native_next28'))/scene
            old_eval = json.loads((frozen_root/'evaluation_complete_next28/report.json').read_text())
            assert candidate['gt_sha256'] == old_eval['gt_sha256']
            metrics = candidate['metrics']['source_accepted']
            decision = candidate['deployment_decision']
        else:
            path = previous_path
            metrics, decision = previous['metrics'], previous['decision']
        rows[scene] = dict(source_views=32 if scene in plan['rebuild_scenes'] else 16,
            decision=decision, baseline={s: previous['metrics'][s]['miou'] for s in ('19', '15', '10')},
            candidate={s: metrics[s]['miou'] for s in ('19', '15', '10')},
            baseline_sha256=sha256_file(previous_path), candidate_sha256=sha256_file(path))
    assert len(rows) == 8
    macro = {s: {arm: sum(r[arm][s] for r in rows.values())/8
                 for arm in ('baseline', 'candidate')} for s in ('19', '15', '10')}
    for s, values in macro.items():
        values['delta_percentage_points'] = 100*(values['candidate']-values['baseline'])
    result = dict(complete=True, independent_test=False, source_only_budget_selection=True,
                  selection_rule_fixed_before_reconstruction=True, all_eight_scenes_in_denominator=True,
                  larger_training_and_input_budget=True, per_scene_target_winner_selection=False,
                  macro=macro, rows=rows, plan_sha256=sha256_file(plan_path),
                  evaluation_registration_sha256=sha256_file(root/'evaluation_registration.json'),
                  implementation_sha256=sha256_file(__file__))
    (root/'summary.json').open('x').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
