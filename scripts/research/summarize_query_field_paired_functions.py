"""Complete functional replay, selected-set evidence and full-cohort results."""
from pathlib import Path

import numpy as np
import torch

from scripts.research import run_query_field_relations as common
from scripts.research import summarize_query_field_protected as protected
from scripts.research import diagnose_query_field_observation_mass as mass
from scripts.research.run_query_field_paired_functions import ROOT, PRIOR, ARMS, frozen_design, read, write, receipt, check
from scripts.research.summarize_query_field_relations import collect, paired

RESULT_PATH = Path('docs/experiments/2026-10-09-paired-function-results.json')


def diagnose():
    extra = dict(
        protected_training=PRIOR / 'protected_training/predictions/uqis',
        joint_conditional_relation=Path('output/query_field_20261008/joint_element_relations_v1/joint_conditional_relation/predictions/uqis'))
    design = read(ROOT / 'design.json')
    if design.get('hard_reference_root'):
        extra['paired_function'] = Path(design['hard_reference_root']) / 'paired_function/predictions/uqis'
    mass.main(ROOT, ARMS[0], extra)


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
    evidence = paths + [ROOT / p for p in ['design.json', 'construction_complete.json', 'predictions_complete.json', 'scoring_complete.json']]
    execution = [read(p) for p in sorted((ROOT / 'execution').glob('*.json'))]
    evidence.extend(sorted((ROOT / 'execution').glob('*.json')))
    construction = []
    for job in design['fields']:
        path = ROOT / 'fits' / arm / job['key']; report = read(path / 'construction.json')
        check(job['parent_manifest']); parent = read(job['parent_manifest']['path']); export = read(path / 'field/manifest.json')
        for key in ['elements', 'observed', 'eligible_anchors', 'semantic_native_dim', 'local_native_dim',
                    'semantic_code_dim', 'local_code_dim', 'group_dim', 'external_scene_dependencies']:
            if export[key] != parent[key]: raise ValueError('field carrier/native contract differs')
        spec = dict(export['spec']); spec.pop('evidence_kind')
        if spec != parent['spec'] or report['training_steps'] or not report['protected_state_unchanged']:
            raise ValueError('undeclared parent/readout changes')
        if report['field_tensor_bytes'] != export['tensor_bytes'] or report['evidence_bytes'] != export['tensor_bytes'] - parent['tensor_bytes']:
            raise ValueError('export cost accounting differs')
        source = read(Path(job['observations']) / 'observations.json')
        if set(report['source_frames']) - set(source['splits']['train_frames']):
            raise ValueError('nontraining source accessed')
        if len(report['replay']) != 2 * len(report['source_frames']): raise ValueError('incomplete functional replay')
        if max(max(r['positive_error'], r['negative_error']) for r in report['replay']) > 1e-5:
            raise ValueError('functional replay failed')
        if not any(r['known_elements'] > 0 for r in report['replay']):
            raise ValueError('functional replay only exercised empty evidence')
        if design.get('hard_reference_root'):
            hard_path = Path(design['hard_reference_root']) / 'fits/paired_function' / job['key'] / 'construction.json'
            hard = read(hard_path); check(report['hard_reference'])
            if not report['exact_hard_association_coordinates'] or report['source_frames'] != hard['source_frames'] or report['source_regions'] != hard['source_regions']:
                raise ValueError('soft target extension changed fixed source/association conditions')
            for soft_view, hard_view in zip(report['source_records'], hard['source_records']):
                if any(soft_view[k] != hard_view[k] for k in ['frame', 'regions']):
                    raise ValueError('soft target extension changed source region trace')
            for soft_case, hard_case in zip(report['replay'], hard['replay']):
                if any(soft_case[k] != hard_case[k] for k in ['source_frame', 'support', 'weights', 'scale']):
                    raise ValueError('soft target extension changed replay supports')
            evidence.append(hard_path)
        construction.append(dict(key=job['key'], family=job['family'], **report))
        evidence.extend([path / 'construction.json', path / 'field/manifest.json', Path(job['observations']) / 'observations.json'])
    if len(construction) != 30: raise ValueError('incomplete field cohort')
    runtime_path = ROOT / 'validation/runtime/report.json'; runtime = read(runtime_path)
    if len(runtime['records']) != 1 or any(set(r['exact_replay']) != {'text', 'image', 'point_2d', 'point_3d', 'category'}
            or not all(r['exact_replay'].values()) or r['forbidden_open_attempts'] for r in runtime['records']):
        raise ValueError('standalone replay failed')
    protected.ARMS = ARMS; native = protected.native_invariance({arm: cohort})
    if native[arm]['exact_native_scores'] != 268: raise ValueError('protected native output changed')
    point_path = ROOT / 'diagnostics/point_observation_mass/report.json'
    if not point_path.exists(): diagnose()
    point = read(point_path); known = {}
    for a in point['summary']:
        cases = [q['arms'][a] for scene in point['scenes'] for q in scene['queries'] if q['arms'][a]['correct_registered_support']]
        if len(cases) != 59: raise ValueError('known-support denominator differs')
        known[a] = dict(queries=len(cases), **{k: float(np.mean([q[k] for q in cases])) for k in
            ['observed_target_recall', 'all_target_recall', 'actual_iou', 'actual_precision']})
    uqis = read(Path(cohort['root']) / 'predictions/uqis/evaluation.json')['explicit_selection']
    evidence.extend([Path(__file__), Path(protected.__file__), Path(mass.__file__),
        Path('scripts/research/summarize_query_field_relations.py'), point_path, runtime_path, Path(design['prior_summary']['path'])])
    result = dict(complete=True, primary_arm=arm, fields=30, macro=macro, per_scene=data, differences=differences,
        construction=construction, functional_replay=dict(cases=sum(len(r['replay']) for r in construction),
            max_absolute_error=max(max(q['positive_error'], q['negative_error']) for r in construction for q in r['replay']),
            nonempty_cases=sum(q['known_elements'] > 0 for r in construction for q in r['replay']),
            direct_source_raster_vs_compiled_gaussian=True, benchmark_queries_used=False),
        field_tensor_bytes=sum(r['field_tensor_bytes'] for r in construction),
        added_evidence_bytes=sum(r['evidence_bytes'] for r in construction),
        native_invariance=native, runtime=runtime, known_support=known, uqis_diagnostics=uqis['identity_diagnosis'],
        v10_targets=previous['v10_targets'], source_only=True, training_steps=0,
        source_access_at_query=False, model_promotion=False, independent_final_test=False,
        per_task_model_selection=False, uncompressed_functional_reference=True,
        full_v10_function_preservation=False, implementation_commit=design.get('implementation_commit', '2b67bed'),
        execution_adjustments=execution)
    if design.get('hard_reference_root'):
        activation_path = ROOT / 'diagnostics/paired_activation/report.json'
        signed_path = ROOT / 'diagnostics/signed_activation/report.json'
        activation, signed = read(activation_path), read(signed_path)
        if not activation['exact_query_associations'] or activation['queries'] != 67 or len(signed['records']) != 8:
            raise ValueError('incomplete conditional activation diagnosis')
        result['soft_target_extension'] = dict(exact_source_and_association=True,
            relation_availability={k: sum(r['relation_availability'][k] for r in construction)
                for k in ['observed_rows', 'available_rows', 'hard_available_rows']},
            point_activation=activation, signed_activation=signed)
        evidence.extend([activation_path, signed_path, Path('scripts/research/summarize_query_field_soft_functions.py')])
    result['evidence'] = [receipt(p) for p in sorted(set(evidence), key=str)]
    write(ROOT / 'summary.json', result); write(RESULT_PATH, result)
    for metric in result['v10_targets']:
        print(metric, {a: round(100*macro[a][metric], 4) for a in ['protected_training', 'joint_conditional_relation', arm]}, flush=True)
    print('functional replay', result['functional_replay'], 'bytes', result['added_evidence_bytes'], flush=True)
    print('known support', known, flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--diagnose', action='store_true'); args = p.parse_args()
    torch.set_num_threads(4)
    diagnose() if args.diagnose else main()
