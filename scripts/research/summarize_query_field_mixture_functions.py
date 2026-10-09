"""Complete unchanged-tensor support-function study, including actual inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.research import run_query_field_relations as common
from scripts.research import summarize_query_field_protected as protected
from scripts.research.run_query_field_mixture_functions import ROOT, PREVIOUS, ARMS, frozen_design, read, write, receipt, check
from scripts.research.summarize_query_field_relations import collect, paired

RESULT = Path('docs/experiments/2026-10-09-mixture-function-results.json')


def runtime_audit(design, arm):
    """Record an executed failed gate without turning tolerance into a waiver."""
    cases = {'text', 'image', 'point_2d', 'point_3d', 'category'}
    if not design.get('stopped_source_counterexample'):
        path = ROOT / 'validation/runtime/report.json'; runtime = read(path)
        if len(runtime['records']) != 1 or any(set(r['exact_replay']) != cases
                or not all(r['exact_replay'].values()) or r['forbidden_open_attempts']
                for r in runtime['records']):
            raise ValueError('standalone replay failed')
        return runtime, None, True, True, [path]

    # This study finished its benchmarks but failed the original strict replay.
    # Keep that gate failed; a complete report must include negative results.
    strict_path = ROOT / 'validation/runtime' / (arm + '.comparison.json')
    strict = read(strict_path)
    if set(strict['exact_replay']) != cases or strict['copied']['forbidden_open_attempts']:
        raise ValueError('strict replay coverage or source denial failed')
    strict_passed = all(strict['exact_replay'].values())
    runtime = dict(strict_cold_replay_passed=strict_passed, original_gate_unchanged=True,
        report_status='passed' if strict_passed else 'failed', strict_trace=strict)
    details = {}
    paths = [strict_path, ROOT / 'validation/runtime.failed0/failure.json',
        ROOT / 'execution/cold_retry.json',
        Path('scripts/research/validate_query_field_mixture_replay.py'),
        Path('scripts/research/validate_query_field_legacy_replay.py')]
    for directory in ['runtime_numeric', 'runtime_legacy_numeric']:
        path = ROOT / 'validation' / directory / 'report.json'; detail = read(path)
        comparisons = [detail['original_repeat']] + [r['comparison'] for r in detail['records']]
        if len(detail['records']) != 2 or any(set(c) != cases for c in comparisons):
            raise ValueError('supplemental cold replay coverage incomplete')
        if detail['tolerance'] != 1e-6 or not detail['numerically_equivalent'] or any(
                not q['exact_selections'] or not 0 <= q['maximum_absolute_error'] <= 1e-6
                for c in comparisons for q in c.values()):
            raise ValueError('supplemental numerical replay failed')
        if any(r['forbidden_open_attempts'] for r in detail['records']):
            raise ValueError('supplemental copied replay accessed sources')
        details[directory] = detail; paths.append(path)
    return runtime, details, strict_passed, True, paths


def diagnose():
    from scripts.research import diagnose_query_field_mixture_activation as points
    from scripts.research import diagnose_query_field_mixture_signed as signed
    if not (ROOT / 'diagnostics/mixture_activation/report.json').exists(): points.main()
    if not (ROOT / 'diagnostics/signed_activation/report.json').exists(): signed.main()


def main():
    design = frozen_design(); read(ROOT / 'scoring_complete.json')
    common.ROOT = ROOT; common.ARMS = ARMS; common.verify_predictions(design)
    previous = read(design['prior_summary']['path']); arm = ARMS[0]
    data = dict(previous['per_scene']); cohort = read(design['arms'][arm]['path'])
    data[arm], paths = collect(cohort)
    for job in cohort['scannet']:
        metrics = read(Path(cohort['root']) / 'predictions/scannet' / job['name'] / 'evaluation.json')['metrics']
        for split in ['19', '15', '10']:
            data[arm].setdefault('ScanNet typed mIoU' + split, {})[job['name']] = metrics['typed'][split]['miou']
    macro = {a: {k: float(np.mean(list(v.values()))) for k, v in row.items()} for a, row in data.items()}
    differences = {a + '__to__' + arm: {k: paired(data[a][k], data[arm][k], k) for k in data[arm]}
                   for a in data if a != arm}
    construction = []
    evidence = paths + [ROOT / p for p in ['design.json', 'construction_complete.json', 'predictions_complete.json', 'scoring_complete.json']]
    for job in design['fields']:
        directory = ROOT / 'fits' / arm / job['key']; report = read(directory / 'construction.json')
        parent = read(job['parent_manifest']['path']); check(job['parent_manifest'])
        export = read(directory / 'field/manifest.json')
        spec = dict(export['spec']); spec['evidence_kind'] = parent['spec']['evidence_kind']
        if spec != parent['spec'] or export['files'] != parent['files'] or export['tensor_bytes'] != parent['tensor_bytes']:
            raise ValueError('function-only intervention changed tensors or another readout')
        if not report['exact_tensor_payload'] or report['additional_tensor_bytes'] or report['training_steps']:
            raise ValueError('undeclared tensor/supervision change')
        if design.get('stopped_source_counterexample') and not report['regional_confidence_retained']:
            raise ValueError('source counterexample repair absent from an export')
        source = read(Path(job['observations']) / 'observations.json')
        if set(report['source_frames']) - set(source['splits']['train_frames']):
            raise ValueError('nontraining source in replay')
        if len(report['replay']) != 2 * len(report['source_frames']): raise ValueError('incomplete source replay')
        for case in report['replay']:
            if max(case[k] for k in ['positive_error', 'negative_error', 'coefficient_error']) > 1e-5:
                raise ValueError('functional replay failed')
        if not any(q['known_elements'] for q in report['replay']): raise ValueError('only empty replay')
        construction.append(dict(key=job['key'], family=job['family'], **report))
        evidence.extend([directory / 'construction.json', directory / 'field/manifest.json'])
    if len(construction) != 30: raise ValueError('incomplete field cohort')
    runtime, runtime_detail, strict_passed, numeric_passed, runtime_paths = runtime_audit(design, arm)
    evidence.extend(runtime_paths)
    protected.ARMS = ARMS; native = protected.native_invariance({arm: cohort})
    if native[arm]['exact_native_scores'] != 268: raise ValueError('native output changed')
    point_path = ROOT / 'diagnostics/mixture_activation/report.json'; point = read(point_path)
    signed_path = ROOT / 'diagnostics/signed_activation/report.json'; signed = read(signed_path)
    budget_path = ROOT / 'diagnostics/reference_budgets.json'; budgets = read(budget_path)
    if point['queries'] != 67 or point['known_correct'] != 59 or len(signed['records']) != 8:
        raise ValueError('actual input audit incomplete')
    evidence.extend([point_path, signed_path, budget_path, Path(design['prior_summary']['path']),
        Path(__file__), Path(protected.__file__), Path('scripts/research/diagnose_query_field_mixture_activation.py'),
        Path('scripts/research/diagnose_query_field_mixture_signed.py'),
        Path('scripts/research/diagnose_query_field_support_composition.py')])
    stopped = None
    carrier = None
    if design.get('stopped_source_counterexample'):
        stopped = read(design['stopped_source_counterexample']['path'])
        if stopped['benchmark_scoring_started']: raise ValueError('source correction followed scoring')
        evidence.extend([Path(design['stopped_source_counterexample']['path']),
                         Path('scripts/research/summarize_query_field_confidence_mixture.py')])
        carrier_path = ROOT / 'diagnostics/uqis_carrier_correspondence.json'
        carrier = read(carrier_path)
        if len(carrier['records']) != 9 or carrier['exact_mapped_scenes'] != 9:
            raise ValueError('incomplete source-budget carrier feasibility audit')
        evidence.extend([carrier_path, Path('scripts/research/audit_query_field_carrier_correspondence.py')])
    result = dict(complete=True, primary_arm=arm, fields=30, macro=macro, per_scene=data, differences=differences,
        completion_scope='All benchmark predictions, scoring and listed diagnostic audits executed; not model acceptance',
        strict_cold_replay_passed=strict_passed, numerical_cold_replay_passed=numeric_passed,
        all_validation_gates_passed=strict_passed and numeric_passed,
        construction=construction, functional_replay=dict(cases=sum(len(r['replay']) for r in construction),
            max_absolute_error=max(q[k] for r in construction for q in r['replay']
                for k in ['positive_error', 'negative_error', 'coefficient_error']),
            nonempty_cases=sum(q['known_elements'] > 0 for r in construction for q in r['replay']),
            direct_source_raster_vs_compiled_gaussian=True, target_rows='fixed sampled carrier rows plus every replay support row',
            benchmark_queries_used=False),
        field_tensor_bytes=sum(r['field_tensor_bytes'] for r in construction), additional_tensor_bytes=0,
        exact_parent_tensor_payload=True, native_invariance=native, runtime=runtime,
        runtime_raw_replay=runtime_detail,
        point_activation=point, signed_activation=signed, v10_targets=previous['v10_targets'],
        reference_budgets=budgets,
        stopped_earlier_trial=stopped,
        uqis_carrier_correspondence=carrier,
        source_only=True, training_steps=0, source_access_at_query=False, model_promotion=False,
        independent_final_test=False, per_task_model_selection=False, uncompressed_functional_reference=True,
        full_v10_function_preservation=False, implementation_commit=design['implementation_commit'],
        evidence=[receipt(p) for p in sorted(set(evidence), key=str)])
    write(ROOT / 'summary.json', result); write(RESULT, result)
    for metric in result['v10_targets']:
        print(metric, {a: round(100*macro[a][metric], 4) for a in ['protected_training', 'soft_paired_function', arm]}, flush=True)
    print('functional replay', result['functional_replay'], flush=True)
    print('known support', point['summary'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--diagnose', action='store_true'); args = p.parse_args()
    torch.set_num_threads(4)
    diagnose() if args.diagnose else main()
