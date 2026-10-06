"""Summarize a fixed four-scene selected-output control without winner picking."""
import json
from pathlib import Path

import numpy as np

from radio_gs.utils.immutable_artifacts import sha256_file


SCENES = {'figurines': 56, 'ramen': 71, 'teatime': 59, 'waldo_kitchen': 22}
ROOT = Path('output/full_field_rebuild_next19')


def main():
    rows = {}
    for scene, count in SCENES.items():
        base = ROOT / scene
        parent_path = base / 'evaluation/report.json'
        candidate_path = base / 'evaluation_selected_next28/report.json'
        training_path = base / 'reconstruction/field/selected_scope_final_next28/report.json'
        parent = json.loads(parent_path.read_text())
        candidate = json.loads(candidate_path.read_text())
        training = json.loads(training_path.read_text())
        assert parent['complete'] and candidate['complete']
        assert parent['observations'] == candidate['observations'] == count
        assert parent['geometry_sha256'] == candidate['geometry_sha256']
        assert parent['encoder_sha256'] == candidate['encoder_sha256']
        assert candidate['field_sha256'] == training['field_sha256']
        assert candidate['cold_functional_replay_exact']
        assert training['source_only'] and not training['benchmark_data_used']
        assert training['parent_name'] == 'bounded_fallback_admission'
        for metric in ('miou', 'macro_ap'):
            assert parent['lerf2d']['native_compact_calibrated'][metric] == candidate['lerf2d']['native_compact_calibrated'][metric]
        assert parent['selected']['native_compact_calibrated']['miou'] == candidate['selected']['native_compact_calibrated']['miou']
        for modality in ('image', 'prompt'):
            assert parent['reference_development']['macro'][modality] == candidate['reference_development']['macro'][modality]
        before = [parent['lerf2d']['final']['miou'], parent['lerf2d']['final']['macro_ap'], parent['selected']['final']['miou']]
        after = [candidate['lerf2d']['final']['miou'], candidate['lerf2d']['final']['macro_ap'], candidate['selected']['final']['miou']]
        rows[scene] = dict(observations=count, before=before, after=after,
                           delta_percentage_points=[100 * (b - a) for a, b in zip(before, after)],
                           reference_text_before=parent['reference_development']['macro']['text']['iou'],
                           reference_text_after=candidate['reference_development']['macro']['text']['iou'],
                           source_validation=training['history'][-1]['validation'],
                           selected_training_step=training['selected_step'],
                           bindings={'parent_report': sha256_file(parent_path),
                                     'candidate_report': sha256_file(candidate_path),
                                     'training_report': sha256_file(training_path)})
    before = np.mean([row['before'] for row in rows.values()], axis=0).tolist()
    after = np.mean([row['after'] for row in rows.values()], axis=0).tolist()
    result = dict(schema='selected_scope_final_next28_summary.v1', complete=True,
                  scenes=list(SCENES), observations=sum(SCENES.values()),
                  method='Fixed 100-step source-only selected-output head control on the same expanded field',
                  independent_test=False, official_four_query=False, benchmark_gt_used_for_training=False,
                  same_geometry_encoder_native_field=True, candidate_default_promoted=False,
                  parent_source_acceptance_is_not_candidate_acceptance=True,
                  first_two_candidate_reports_inherited_parent_source_acceptance=True,
                  macro_before=before, macro_after=after,
                  macro_delta_percentage_points=[100 * (b - a) for a, b in zip(before, after)],
                  metrics=['native_grid_2d_iou', 'native_grid_2d_ap', 'selected_element_render_iou'],
                  rows=rows, implementation_sha256=sha256_file(__file__))
    output = Path('output/selected_scope_final_next28_summary.json')
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('macro_before', 'macro_after', 'macro_delta_percentage_points')}, indent=2))


if __name__ == '__main__':
    main()
