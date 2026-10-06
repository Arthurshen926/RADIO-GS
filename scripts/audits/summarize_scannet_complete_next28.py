"""All-scene ScanNet comparison after rebuilding the complete semantic field."""
import json
from pathlib import Path

import numpy as np

from radio_gs.utils.immutable_artifacts import sha256_file


SCENES = ('scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00',
          'scene0140_00', 'scene0347_00', 'scene0400_00', 'scene0590_00')
SPLITS = ('19', '15', '10')
OLD_ROOT = Path('output/scannet_coverage_native_probe_20260919')
NEW_ROOT = Path('output/scannet_coverage_native_next28')
VIS_ROOT = Path('output/scannet_visibility_audit_next28')
FROZEN_ROOT = Path('output/scannet_frozen_field_20260918')


def main():
    rows = {}
    for scene in SCENES:
        root = (OLD_ROOT if scene in SCENES[:2] else NEW_ROOT) / scene
        full_base_path = root / 'base_full_next28/execution.json'
        attempt = 'complete_next28_retry2' if scene in SCENES[:2] else 'complete_next28'
        full_exec_path = root / (attempt + '_execution.json')
        report_path = root / 'evaluation_complete_next28/report.json'
        base = json.loads(full_base_path.read_text())
        execution = json.loads(full_exec_path.read_text())
        report = json.loads(report_path.read_text())
        assert base['base_finished'] and execution['complete']
        assert report['scene'] == scene and report['complete_field_rebuilt']
        assert report['gt_opened_after_prediction'] and not report['independent_test']
        assert report['source_view_budget'] == 16
        visibility_path = VIS_ROOT / scene / 'report.json'
        visibility = json.loads(visibility_path.read_text())
        plan_path = root / 'base_full_next28/source_plan.json'
        plan = json.loads(plan_path.read_text())
        assert set(plan['train_frames']) == set(visibility['equal_budget_geometry_selected_frames'])
        assert set(plan['validation_frames']) == set(visibility['validation_views'])
        frozen = json.loads((FROZEN_ROOT / scene / 'official_protocol_evaluation/report.json').read_text())
        assert report['geometry_sha256'] == frozen['geometry_sha256']
        assert report['gt_sha256'] == frozen['gt_sha256']
        assert report['baseline_metrics'] == frozen['metrics']
        native_control_path = root / 'evaluation.json'
        native_control = json.loads(native_control_path.read_text())
        assert native_control['geometry_sha256'] == report['geometry_sha256']
        ladder_path = root / 'calibration_ladder_next28/report.json'
        ladder = json.loads(ladder_path.read_text())
        assert ladder['post_hoc_diagnostic'] and not ladder['benchmark_arm']
        assert ladder['original_evaluation_sha256'] == sha256_file(report_path)
        assert ladder['field_sha256'] == report['field_sha256']
        for readout in ('native', 'bounded_scope'):
            for split in SPLITS:
                assert ladder['metrics'][readout][split]['miou'] == report['metrics'][readout][split]['miou']
        source_acceptance = report['source_acceptance']
        rows[scene] = {
            'baseline': {s: frozen['metrics'][s]['miou'] for s in SPLITS},
            'native_only_control': {s: native_control['metrics'][s]['miou'] for s in SPLITS},
            'native': {s: report['metrics']['native'][s]['miou'] for s in SPLITS},
            'calibrated_native': {s: ladder['metrics']['calibrated_native'][s]['miou']
                                  for s in SPLITS},
            'bounded_scope_category_control': {
                s: report['metrics']['bounded_scope'][s]['miou'] for s in SPLITS},
            'source_acceptance': source_acceptance,
            'artifacts': {name: sha256_file(path) for name, path in {
                'complete_base': full_base_path,
                'complete_execution': full_exec_path,
                'evaluation': report_path,
                'native_only_control': native_control_path,
                'calibration_ladder': ladder_path,
                'source_plan': plan_path,
                'visibility': visibility_path,
            }.items()},
        }
    macro = {}
    for readout in ('baseline', 'native_only_control', 'native', 'calibrated_native',
                    'bounded_scope_category_control'):
        macro[readout] = {s: float(np.mean([row[readout][s] for row in rows.values()]))
                          for s in SPLITS}
    deltas = {
        readout: {s: 100 * (macro[readout][s] - macro['baseline'][s]) for s in SPLITS}
        for readout in ('native_only_control', 'native', 'calibrated_native',
                        'bounded_scope_category_control')
    }
    result = {
        'schema': 'scannet_equal_budget_complete_field_next28.v1',
        'complete': True,
        'scenes': list(SCENES),
        'all_scenes_in_denominator': True,
        'input_track': 'provided_unlabelled_mesh',
        'source_selection_used_semantic_gt': False,
        'target_gt_opened_after_predictions': True,
        'independent_test': False,
        'same_geometry_and_metric_as_frozen_baseline': True,
        'source_view_budget': 16,
        'complete_field_rebuilt': True,
        'scan_category_protocol_primary_readout': 'native',
        'bounded_scope_is_category_control': True,
        'calibrated_native_is_post_hoc_decomposition': True,
        'four_query_benchmark_result': False,
        'macro': macro,
        'delta_percentage_points': deltas,
        'complete_native_minus_native_only_percentage_points': {
            s: 100 * (macro['native'][s] - macro['native_only_control'][s])
            for s in SPLITS},
        'rows': rows,
        'implementation_sha256': sha256_file(__file__),
    }
    out = NEW_ROOT / 'complete_summary.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'macro': macro, 'delta_percentage_points': deltas}, indent=2))


if __name__ == '__main__':
    main()
