"""Final CPU-only identity check for the completed next31 method controls.

This does not rerun tests/metrics or certify SOTA. Heavy checkpoint and prediction
identities remain covered by each experiment's existing completion receipts.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checked, input_receipts = {}, {}

    def read(name):
        path = ROOT / name
        input_receipts[name] = sha(path)
        return json.loads(path.read_text())

    def check(name, expected):
        path = ROOT / name
        key = str(path.relative_to(ROOT))
        previous = checked.get(key)
        if previous is not None and previous['expected'] != expected:
            raise RuntimeError(f'Conflicting bound identity: {key}')
        checked[key] = dict(expected=expected, actual=sha(path))

    authority = read('paper/artifacts/frozen_cross_benchmark_method_20260918.json')
    check(authority['source_manifest'], authority['source_manifest_sha256'])
    manifest = read(authority['source_manifest'])
    frozen_errors = []
    for name, expected in manifest.items():
        actual = sha(ROOT / name)
        if actual != expected:
            frozen_errors.append(dict(path=name, expected=expected, actual=actual))
    for name, expected in authority['evaluation_protocols'].items():
        check(name, expected)

    suites, test_files = [], []
    for name in ('output/new_method_tests_next31/report.json',
                 'output/new_method_tests_next31/proposal_addendum/report.json'):
        suite = read(name)
        assert suite['complete'] and suite['exit_code'] == 0
        assert not suite['inputs_changed_during_test']
        for path, expected in suite.get('input_hashes', suite.get('inputs')).items():
            check(path, expected)
        cases = int(re.fullmatch(r'(\d+) passed in [\d.]+s', suite['result_summary'])[1])
        files = [x for x in suite['command'] if x.startswith('tests/')]
        test_files.extend(files)
        suites.append(dict(receipt=name, count=cases, files=files,
                           result=suite['result_summary']))
    assert len(test_files) == len(set(test_files)), 'Duplicate test files across suites'

    lerf_dir = 'output/selected_visibility_next31_v2/'
    lerf = read(lerf_dir + 'registration.json')
    check('scripts/audits/train_visibility_selected_scope_next31.py', lerf['implementation_sha256'])
    check('radio_gs/training/selected_output_supervision.py', lerf['visibility_implementation_sha256'])
    lerf_eval = read(lerf_dir + 'evaluation_registration.json')
    check(lerf_dir + 'registration.json', lerf_eval['training_registration_sha256'])
    check('scripts/audits/evaluate_full_field_rebuild_next19.py', lerf_eval['evaluator_sha256'])
    check('scripts/audits/evaluate_selected_visibility_next31.py', lerf_eval['runner_sha256'])
    lerf_summary = read(lerf_dir + 'summary.json')
    assert lerf_summary['complete'] and lerf_summary['observations'] == 208
    assert not lerf_summary['mainline_promoted'] and not lerf_summary['independent_test']

    scan_dir = 'output/scannet_selected_visibility_next31/'
    scan = read(scan_dir + 'registration.json')
    check('scripts/audits/train_registered_selected_scope_next31.py', scan['implementation_sha256'])
    check('radio_gs/training/selected_output_supervision.py', scan['visibility_implementation_sha256'])
    check('scripts/audits/run_scannet_selected_visibility_next31.py', scan['driver_sha256'])
    scan_eval = read(scan_dir + 'evaluation_registration.json')
    check(scan_dir + 'registration.json', scan_eval['source_registration_sha256'])
    for path, expected in scan_eval['implementations'].items():
        check(path, expected)
    scan_summary = read(scan_dir + 'summary.json')
    assert scan_summary['complete'] and scan_summary['scenes'] == 8
    assert scan_summary['paired_source_refits'] == 16
    assert scan_summary['cold_native_and_retained_prediction_replay_exact']
    assert scan_summary['inherited_rejected_scenes'] == ['scene0590_00']
    assert not scan_summary['independent_test']
    check(scan_dir + 'evaluation_registration.json', scan_summary['evaluation_registration_sha256'])
    check(scan_dir + 'registration.json', scan_summary['registration_sha256'])
    check('scripts/audits/summarize_scannet_selected_visibility_next31.py', scan_summary['implementation_sha256'])

    mixture = read('output/native_mixture_next31/registration.json')
    for path, expected in mixture['implementations'].items():
        check(path, expected)
    seal_names = [
        'paper/artifacts/matched_source_fusion_next31.json',
        'paper/artifacts/nvos_component_reference_next31.json',
        'paper/artifacts/scannet_selected_visibility_next31.json',
        'paper/artifacts/native_mixture_source_pilots_next31.json',
    ]
    skipped_heavy = []
    for name in seal_names:
        seal = read(name)
        for section in ('implementations', 'artifacts', 'artifact_hashes'):
            for path, expected in seal.get(section, {}).items():
                if (ROOT / path).suffix in ('.pt', '.npz', '.npy', '.png'):
                    skipped_heavy.append(path)
                    continue
                check(path, expected)
    mixture_seal = read(seal_names[-1])
    assert mixture_seal['candidate_rejected'] and not mixture_seal['benchmark_evaluation_launched']
    assert not mixture_seal['default_promoted']
    nvos = read(seal_names[1])
    assert nvos['candidate_rejected'] and nvos['all_eight_baseline_masks_replay_exact']
    errors = [dict(path=p, **v) for p, v in checked.items() if v['actual'] != v['expected']]
    passed = not errors and not frozen_errors
    report = dict(
        schema='new_method_independent_final_review.next31', passed=passed,
        frozen_source_count=len(manifest), frozen_source_errors=frozen_errors,
        frozen_and_new_binding_count=len(checked), binding_errors=errors,
        binding_checks=checked, input_receipts=input_receipts,
        unique_tests_passed=sum(s['count'] for s in suites), test_suites=suites,
        tests_rerun=False, benchmark_metrics_rerun=False, gpu_used=False,
        heavy_assets_not_rehashed=sorted(set(skipped_heavy)),
        scope='Frozen source/contract identity, completed experiment code/report bindings, disjoint test receipts',
        independent_test_certified=False, unified_method_certified=False, sota_certified=False,
        remaining_submission_blockers=[
            'One deployable four-query method and input contract must be frozen jointly.',
            'Same-permission, same-geometry/budget historical and SOTA controls remain incomplete.',
            'Repeatedly inspected development cohorts cannot replace independent frozen validation.',
            'LERF selected rendering is not physical 3D truth; UQIS remains paused.',
            'Granularity-preserving fusion and query-competitive scope validation remain unresolved.',
        ],
        implementation_sha256=sha(Path(__file__)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in ('passed', 'frozen_source_count',
                     'frozen_and_new_binding_count', 'unique_tests_passed',
                     'frozen_source_errors', 'binding_errors')}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
