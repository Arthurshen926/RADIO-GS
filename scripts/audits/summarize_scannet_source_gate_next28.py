"""Conservative source-acceptance gate for the ScanNet category control."""
import json
from pathlib import Path

import numpy as np

from radio_gs.utils.immutable_artifacts import sha256_file


SCENES = ('scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00',
          'scene0140_00', 'scene0347_00', 'scene0400_00', 'scene0590_00')
SPLITS = ('19', '15', '10')


def main():
    complete_path = Path('output/scannet_coverage_native_next28/complete_summary.json')
    complete = json.loads(complete_path.read_text())
    assert complete['complete'] and tuple(complete['scenes']) == SCENES
    rows = {}
    for scene in SCENES:
        root = (Path('output/scannet_coverage_native_probe_20260919')
                if scene in SCENES[:2] else Path('output/scannet_coverage_native_next28')) / scene
        report = json.loads((root / 'evaluation_complete_next28/report.json').read_text())
        ladder = json.loads((root / 'calibration_ladder_next28/report.json').read_text())
        accepted = report['source_acceptance']
        assert isinstance(accepted, bool)
        readout = 'bounded_scope' if accepted else 'calibrated_native'
        prediction = (root / 'evaluation_complete_next28' / (readout + '.npz')
                      if accepted else root / 'calibration_ladder_next28/calibrated_native.npz')
        assert sha256_file(prediction) == (
            report['prediction_sha256'][readout] if accepted
            else ladder['prediction_sha256'][readout])
        metrics = report['metrics'][readout] if accepted else ladder['metrics'][readout]
        rows[scene] = dict(source_acceptance=accepted, selected_readout=readout,
                           miou={s: metrics[s]['miou'] for s in SPLITS},
                           prediction_sha256=sha256_file(prediction),
                           evaluation_sha256=sha256_file(root / 'evaluation_complete_next28/report.json'),
                           ladder_sha256=sha256_file(root / 'calibration_ladder_next28/report.json'))
    macro = {s: float(np.mean([r['miou'][s] for r in rows.values()])) for s in SPLITS}
    result = dict(schema='scannet_source_gated_category_diagnostic_next28.v1',
                  complete=True, scenes=list(SCENES), all_scenes_in_denominator=True,
                  source_gate_rule='Accept bounded scope only when its preexisting source acceptance is true; otherwise use calibrated native.',
                  source_gate_uses_target_labels=False, candidate_policy_assembled_after_target_evaluations=True,
                  post_hoc_development_diagnostic=True, independent_test=False,
                  promotion_eligible=False, raw_bounded_scope_source_acceptance_all=False,
                  macro=macro,
                  delta_over_frozen_percentage_points={s: 100*(macro[s]-complete['macro']['baseline'][s])
                                                       for s in SPLITS},
                  rows=rows, complete_summary_sha256=sha256_file(complete_path),
                  implementation_sha256=sha256_file(__file__))
    out = Path('output/scannet_coverage_native_next28/source_gated_summary.json')
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'macro': macro,
                      'delta_over_frozen_percentage_points': result['delta_over_frozen_percentage_points'],
                      'source_failures': [s for s, row in rows.items() if not row['source_acceptance']]}, indent=2))


if __name__ == '__main__':
    main()
