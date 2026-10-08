"""Complete element-supervision and shared conditional-member attribution."""
from pathlib import Path

import numpy as np
import torch

from scripts.research import run_query_field_relations as common
from scripts.research import summarize_query_field_protected as protected
from scripts.research.run_query_field_elements import ROOT, CONTROL, ARMS, frozen_design, read, write, receipt
from scripts.research.summarize_query_field_relations import collect, paired


def main():
    design = frozen_design(); read(ROOT / 'scoring_complete.json')
    common.ROOT = ROOT; common.ARMS = ARMS
    common.verify_predictions(design)
    previous = read(CONTROL / 'summary.json')
    data = {a: previous['per_scene'][a] for a in ['parent', 'previous_common', 'protected_training']}
    cohorts = {a: read(design['arms'][a]['path']) for a in ARMS}
    evidence = [ROOT / name for name in ['design.json', 'construction_complete.json', 'predictions_complete.json', 'scoring_complete.json']]
    diagnostics = {}
    for arm, cohort in cohorts.items():
        data[arm], paths = collect(cohort); evidence.extend(paths)
        for job in cohort['scannet']:
            metrics = read(Path(cohort['root']) / 'predictions/scannet' / job['name'] / 'evaluation.json')['metrics']
            for split in ['19', '15', '10']:
                data[arm].setdefault('ScanNet typed mIoU' + split, {})[job['name']] = metrics['typed'][split]['miou']
        explicit = read(Path(cohort['root']) / 'predictions/uqis/evaluation.json')['explicit_selection']
        diagnostics[arm] = dict(modalities={m: dict(scene_macro=r['scene_macro'], empty_selections=r['empty_selections'])
            for m, r in explicit['modalities'].items()}, dominant_selected_identity=explicit['identity_diagnosis'])
    macro = {a: {k: float(np.mean(list(v.values()))) for k, v in row.items()} for a, row in data.items()}
    comparisons = [(a, b) for a in ['parent', 'previous_common', 'protected_training'] for b in ARMS] + [ARMS]
    differences = {a + '__to__' + b: {k: paired(data[a][k], data[b][k], k) for k in data[b]} for a, b in comparisons}
    training, total_bytes, extra_bytes = [], {a: 0 for a in ARMS}, {a: 0 for a in ARMS}
    for job in design['fields']:
        parent_manifest = read(job['parent_manifest']['path'])
        control = read(job['matched_control_training']['path'])
        samples = []
        for arm in ARMS:
            out = ROOT / 'fits' / arm / job['key']; report = read(out / 'training.json'); manifest = read(out / 'field/manifest.json')
            if not report['protected_state_unchanged'] or report['trace'] != control['trace']:
                raise ValueError('source pair trace/protection mismatch')
            for key in ['elements', 'observed', 'eligible_anchors', 'semantic_native_dim', 'local_native_dim',
                        'semantic_code_dim', 'local_code_dim', 'group_dim', 'external_scene_dependencies']:
                if manifest[key] != parent_manifest[key]:
                    raise ValueError('parent carrier/native contract changed: ' + key)
            expected_extra = 0 if arm == 'element_relation' else 4 * (manifest['elements'] + 1441)
            if manifest['tensor_bytes'] - parent_manifest['tensor_bytes'] != expected_extra:
                raise ValueError('unexpected scene payload beyond decoder and scalar confidence')
            samples.append(report['sample_trace_sha256'])
            evidence.extend([out / 'training.json', out / 'field/manifest.json'])
            training.append(dict(arm=arm, key=job['key'], family=job['family'],
                **{k: v for k, v in report.items() if k not in {'arm', 'history', 'trace'}}))
            total_bytes[arm] += manifest['tensor_bytes']
            extra_bytes[arm] += manifest['tensor_bytes'] - parent_manifest['tensor_bytes']
        if samples[0] != samples[1]:
            raise ValueError('element samples/targets differ between arms')
    source = {}
    for arm in ARMS:
        rows = [r for r in training if r['arm'] == arm]
        if len(rows) != 30 or sum(r['steps'] for r in rows) != 50902:
            raise ValueError('field/source region coverage mismatch')
        source[arm] = dict(parent_bce=float(np.mean([r['source_development']['metrics']['parent_bce'] for r in rows])),
            refined_bce=float(np.mean([r['source_development']['metrics']['refined_bce'] for r in rows])),
            improved_scenes=sum(r['source_development']['metrics']['refined_bce'] < r['source_development']['metrics']['parent_bce'] for r in rows),
            training_skipped_steps=sum(r['skipped_steps'] for r in rows),
            development_regions=sum(r['source_development']['regions'] for r in rows),
            development_skipped=sum(r['source_development']['skipped'] for r in rows),
            sampled_observed_fraction=float(np.mean([r['row_exposure']['sampled_rows'] / r['row_exposure']['observed_rows'] for r in rows])),
            ambiguous_sample_fraction=float(np.mean([r['ambiguous_sample_fraction'] for r in rows])),
            mean_known_confidence=float(np.mean([r['mean_sample_known_confidence'] for r in rows])))
    runtime_path = ROOT / 'validation/runtime/report.json'; runtime = read(runtime_path)
    if any(not all(r['exact_replay'].values()) or r['forbidden_open_attempts'] for r in runtime['records']):
        raise ValueError('standalone copied-export replay failed')
    protected.ARMS = ARMS
    native = protected.native_invariance(cohorts)
    conditioning_path = ROOT / 'diagnostics/conditioning/report.json'
    point_path = ROOT / 'diagnostics/point_observation_mass/within_query_comparison.json'
    conditioning = read(conditioning_path); point = read(point_path)
    if conditioning['fields'] != 30 or len(conditioning['scenes']) != 30:
        raise ValueError('conditioning diagnostic cohort incomplete')
    fitted = {r['key']: r for r in training if r['arm'] == 'conditional_relation'}
    for scene in conditioning['scenes']:
        if not np.isclose(scene['metrics']['original'],
                          fitted[scene['key']]['source_development']['metrics']['refined_bce'], atol=1e-5, rtol=1e-5):
            raise ValueError('copied source audit does not replay the fitted source result')
    for arm in ARMS:
        if point['known_support'][arm]['queries'] != 59:
            raise ValueError('known-correct point diagnostic denominator changed')
    evidence.extend([runtime_path, Path(__file__), Path(protected.__file__), CONTROL / 'summary.json',
                     Path('scripts/research/summarize_query_field_relations.py'), conditioning_path, point_path,
                     ROOT / 'diagnostics/point_observation_mass/report.json'])
    result = dict(complete=True, fields=30, exports=60, primary_arm='conditional_relation',
        implementation_commit='4bdf77a', macro=macro, per_scene=data, differences=differences,
        training=training, source_development=source, export_tensor_bytes=total_bytes, added_tensor_bytes=extra_bytes,
        exact_control_source_traces=True, exact_element_samples=True, protected_native_invariance=native,
        runtime=runtime, uqis_diagnostics=diagnostics, v10_targets=previous['v10_targets'],
        source_conditioning_diagnostics=conditioning, known_support_diagnostics=point,
        source_only=True, source_access_at_query=False, model_promotion=False, independent_final_test=False,
        per_task_model_selection=False, evidence=[receipt(p) for p in sorted(set(evidence), key=str)])
    write(ROOT / 'summary.json', result)
    write('docs/experiments/2026-10-08-element-relation-results.json', result)
    for metric in result['v10_targets']:
        print(metric, {a: round(100 * macro[a][metric], 4) for a in macro}, flush=True)
    print('source development', source, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
