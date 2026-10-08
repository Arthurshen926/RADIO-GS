"""Complete matched joint element/rendered conditional-member results."""
from pathlib import Path

import numpy as np
import torch

from scripts.research import run_query_field_relations as common
from scripts.research import summarize_query_field_protected as protected
from scripts.research.run_query_field_joint_elements import ROOT, CONTROL, ARMS, frozen_design, read, write, receipt, check
from scripts.research.summarize_query_field_relations import collect, paired


def main():
    design = frozen_design(); read(ROOT / 'scoring_complete.json')
    common.ROOT = ROOT; common.ARMS = ARMS; common.verify_predictions(design)
    previous = read(CONTROL / 'summary.json'); arm = ARMS[0]
    data = dict(previous['per_scene']); cohort = read(design['arms'][arm]['path'])
    data[arm], paths = collect(cohort)
    evidence = [ROOT / p for p in ['design.json', 'construction_complete.json', 'predictions_complete.json', 'scoring_complete.json']]
    evidence.extend(paths)
    for job in cohort['scannet']:
        metrics = read(Path(cohort['root']) / 'predictions/scannet' / job['name'] / 'evaluation.json')['metrics']
        for split in ['19', '15', '10']:
            data[arm].setdefault('ScanNet typed mIoU' + split, {})[job['name']] = metrics['typed'][split]['miou']
    macro = {a: {k: float(np.mean(list(v.values()))) for k, v in row.items()} for a, row in data.items()}
    differences = {a + '__to__' + arm: {k: paired(data[a][k], data[arm][k], k) for k in data[arm]}
                   for a in data if a != arm}
    training = []; total_bytes = 0
    for job in design['fields']:
        check(job['matched_element_training']); control = read(job['matched_element_training']['path'])
        out = ROOT / 'fits' / arm / job['key']; report = read(out / 'training.json'); manifest = read(out / 'field/manifest.json')
        parent = read(job['parent_manifest']['path'])
        reference = read(CONTROL / 'fits/conditional_relation' / job['key'] / 'field/manifest.json')
        for key in ['trace', 'sample_trace_sha256', 'parent_sha256', 'source_frames', 'training_regions', 'steps',
                    'epochs', 'seed', 'learning_rate', 'per_class', 'row_exposure']:
            if report[key] != control[key]:
                raise ValueError('joint fit changed matched construction: ' + key)
        if not report['protected_state_unchanged'] or (report['element_weight'], report['rendered_weight']) != (.5, .5):
            raise ValueError('joint protected state/objective differs')
        for key in ['elements', 'observed', 'eligible_anchors', 'semantic_native_dim', 'local_native_dim',
                    'semantic_code_dim', 'local_code_dim', 'group_dim', 'external_scene_dependencies', 'tensor_bytes']:
            if manifest[key] != reference[key]:
                raise ValueError('joint export changed capacity: ' + key)
        if manifest['tensor_bytes'] - parent['tensor_bytes'] != 4 * (manifest['elements'] + 1441):
            raise ValueError('unexpected scene payload beyond the declared residual/confidence')
        if report['rendered_logit_gradient']['absolute_gradient_total'] <= 0:
            raise ValueError('rendered branch did not supervise members')
        if not all(np.isfinite(v) for v in report['source_development']['metrics'].values()):
            raise ValueError('nonfinite source-development result')
        gradient_coverage = report['rendered_logit_gradient']['nonzero_observed_fraction']
        if not 0 < gradient_coverage <= 1:
            raise ValueError('invalid actual rendered-logit gradient coverage')
        total_bytes += manifest['tensor_bytes']
        training.append(dict(key=job['key'], family=job['family'], **{k: v for k, v in report.items() if k not in {'history', 'trace'}}))
        evidence.extend([out / 'training.json', out / 'field/manifest.json'])
    if len(training) != 30 or sum(r['steps'] for r in training) != 50902:
        raise ValueError('incomplete joint construction cohort')
    source = {k: float(np.mean([r['source_development']['metrics'][k] for r in training]))
              for k in training[0]['source_development']['metrics']}
    source.update(element_improved_scenes=sum(r['source_development']['metrics']['refined_bce'] <
        r['source_development']['metrics']['parent_bce'] for r in training),
        rendered_improved_scenes=sum(r['source_development']['metrics']['refined_rendered_bce'] <
        r['source_development']['metrics']['parent_rendered_bce'] for r in training),
        training_skipped_steps=sum(r['skipped_steps'] for r in training),
        development_regions=sum(r['source_development']['regions'] for r in training),
        development_skipped=sum(r['source_development']['skipped'] for r in training),
        sampled_observed_fraction=float(np.mean([r['row_exposure']['sampled_rows']/r['row_exposure']['observed_rows'] for r in training])),
        rendered_nonzero_logit_gradient_observed_fraction=float(np.mean([r['rendered_logit_gradient']['nonzero_observed_fraction'] for r in training])))
    runtime_path = ROOT / 'validation/runtime/report.json'; runtime = read(runtime_path)
    cases = {'text', 'image', 'point_2d', 'point_3d', 'category'}
    if len(runtime['records']) != 1 or any(set(r['exact_replay']) != cases or
            not all(r['exact_replay'].values()) or r['forbidden_open_attempts'] for r in runtime['records']):
        raise ValueError('joint standalone cold replay failed')
    protected.ARMS = ARMS; native = protected.native_invariance({arm: cohort})
    point_path = ROOT / 'diagnostics/point_observation_mass/within_query_comparison.json'; point = read(point_path)
    if point['known_support'][arm]['queries'] != 59:
        raise ValueError('known-correct point denominator changed')
    explicit = read(Path(cohort['root']) / 'predictions/uqis/evaluation.json')['explicit_selection']
    diagnostics = dict(modalities={m: dict(scene_macro=v['scene_macro'], empty_selections=v['empty_selections'])
        for m, v in explicit['modalities'].items()}, dominant_selected_identity=explicit['identity_diagnosis'])
    evidence.extend([CONTROL / 'summary.json', runtime_path, point_path,
        ROOT / 'diagnostics/point_observation_mass/report.json', Path(__file__), Path(protected.__file__),
        Path('scripts/research/summarize_query_field_relations.py')])
    execution = []
    for path in sorted((ROOT / 'execution').glob('*.json')):
        record = read(path)
        if 'training' in record: check(record['training'])
        if 'code' in record: check(record['code'])
        execution.append(record); evidence.append(path)
    result = dict(complete=True, fields=30, exports=30, primary_arm=arm, implementation_commit='7d9fcc8',
        macro=macro, per_scene=data, differences=differences, training=training, source_development=source,
        export_tensor_bytes=total_bytes, exact_conditional_control_source_traces=True, exact_element_samples=True,
        protected_native_invariance=native, runtime=runtime, known_support_diagnostics=point,
        uqis_diagnostics={arm: diagnostics}, v10_targets=previous['v10_targets'],
        execution_adjustments=execution,
        source_only=True, source_access_at_query=False, model_promotion=False, independent_final_test=False,
        per_task_model_selection=False, source_feature_binding_interventions_run_for_this_arm=False,
        evidence=[receipt(p) for p in sorted(set(evidence), key=str)])
    write(ROOT / 'summary.json', result); write('docs/experiments/2026-10-08-joint-relation-results.json', result)
    for metric in result['v10_targets']:
        print(metric, {a: round(100*macro[a][metric], 4) for a in ['protected_training', 'conditional_relation', arm]}, flush=True)
    print('source development and gradient coverage', source, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
