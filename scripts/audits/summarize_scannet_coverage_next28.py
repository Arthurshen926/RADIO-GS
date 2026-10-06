"""Eight-scene fixed native-path coverage control, with no scene selection."""
import json
from pathlib import Path

import numpy as np

from radio_gs.utils.immutable_artifacts import sha256_file


SCENES = ('scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00',
          'scene0140_00', 'scene0347_00', 'scene0400_00', 'scene0590_00')
OLD_ROOT = Path('output/scannet_coverage_native_probe_20260919')
NEW_ROOT = Path('output/scannet_coverage_native_next28')
VIS_ROOT = Path('output/scannet_visibility_audit_next28')


def main():
    rows = {}
    for scene in SCENES:
        root = (OLD_ROOT if scene in SCENES[:2] else NEW_ROOT) / scene
        execution = json.loads((root / 'execution.json').read_text())
        assert execution['complete'], scene
        evaluation_path = root / 'evaluation.json'
        evaluation = json.loads(evaluation_path.read_text())
        visibility_path = VIS_ROOT / scene / 'report.json'
        visibility = json.loads(visibility_path.read_text())
        plan_path = root / 'base/source_plan.json'
        plan = json.loads(plan_path.read_text())
        assert len(plan['train_frames']) == len(visibility['training_views']) == 16
        assert set(plan['train_frames']) == set(visibility['equal_budget_geometry_selected_frames'])
        assert set(plan['validation_frames']) == set(visibility['validation_views'])
        assert evaluation['geometry_unchanged'] and not evaluation['source_frame_selection_used_semantic_gt']
        assert evaluation['geometry_sha256'] == visibility['geometry_sha256']
        equivalence_path = Path('output/native_primary_readout_equivalence_20260919') / scene / 'report.json'
        equivalence = json.loads(equivalence_path.read_text())
        assert not any(equivalence['reference_prediction_mismatches'].values())
        baseline = {split: evaluation['baseline_metrics'][split]['miou'] for split in ('19', '15', '10')}
        control = {split: evaluation['metrics'][split]['miou'] for split in ('19', '15', '10')}
        rows[scene] = dict(baseline=baseline, control=control,
                           delta_percentage_points={split: 100*(control[split]-baseline[split]) for split in baseline},
                           current_visibility=visibility['current_train']['volume_fraction'],
                           selected_visibility=visibility['equal_budget_geometry_selection']['volume_fraction'],
                           artifacts={'evaluation': sha256_file(evaluation_path),
                                      'plan': sha256_file(plan_path),
                                      'visibility': sha256_file(visibility_path),
                                      'baseline_equivalence': sha256_file(equivalence_path)})
    macro = {split: dict(baseline=float(np.mean([r['baseline'][split] for r in rows.values()])),
                         control=float(np.mean([r['control'][split] for r in rows.values()])))
             for split in ('19', '15', '10')}
    for item in macro.values():
        item['delta_percentage_points'] = 100*(item['control']-item['baseline'])
    result = dict(schema='scannet_equal_budget_coverage_native_next28.v1', complete=True,
                  scenes=list(SCENES), geometry_fixed=True, source_view_budget=16,
                  source_selection_used_semantic_gt=False, independent_test=False,
                  native_classification_only=True, complete_four_query_method=False,
                  all_scenes_in_denominator=True, macro=macro, rows=rows,
                  implementation_sha256=sha256_file(__file__))
    output = Path('output/scannet_coverage_native_next28/summary.json')
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
