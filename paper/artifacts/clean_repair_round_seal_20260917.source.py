"""Seal completed scheduled controls without claiming all research gates passed."""
import argparse
import json
import shutil
from pathlib import Path

from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, default=Path('paper/artifacts'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ['ropefixed_rebuilds_complete_20260917.json',
             'clean_nvos_eight_task_summary_20260917.json',
             'pose_covered_lerf_comparison_20260917.json']
    reports = {name: json.loads((args.artifacts / name).read_text()) for name in names}
    summary = reports[names[0]]
    gates = ['all_scheduled_rebuilds_finished', 'all_precision_validations_finished',
             'all_clean_joint_controls_finished', 'all_pose_diagnostics_finished',
             'all_uniform_structural_controls_finished', 'all_clean_bounded_controls_finished']
    if not all(summary.get(key) is True for key in gates):
        raise RuntimeError('Scheduled execution remains incomplete')
    if len(reports[names[1]]['tasks']) != 8:
        raise RuntimeError('Incomplete NVOS task cohort')
    unresolved = [
        'Independent dense geometry, boundaries and thin-structure validation',
        'Verified physical instance identity and whole/part supervision',
        'Shared open-vocabulary extent inference that improves text and image',
        'All signed reference constraints satisfied',
        'Same complete method on ScanNet OVS and independent final test',
    ]
    result = dict(
        schema='clean_repair_round_seal.v1', scheduled_execution_complete=True,
        full_research_checklist_passed=False, independent_final_test=False,
        unresolved_acceptance_gates=unresolved,
        reports={str(args.artifacts / name): sha256_file(args.artifacts / name)
                 for name in names},
        implementation_sha256=sha256_file(__file__),
        decision='Retain scoped development gains and rejected controls; no universal promotion.',
    )
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, args.output.with_suffix('.source.py'))


if __name__ == '__main__':
    main()
