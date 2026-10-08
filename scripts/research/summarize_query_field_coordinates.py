"""Complete capacity-control results, including unchanged registered queries."""
from pathlib import Path

import numpy as np
import torch

from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_coordinates import ROOT, PARENT_ROOT, ARMS, frozen_design, read, write, receipt
from scripts.research.summarize_query_field_relations import collect, paired
from scripts.research.summarize_query_field_protected import point_invariance


def main():
    design = frozen_design(); read(ROOT / 'scoring_complete.json')
    common.ROOT = ROOT; common.ARMS = ARMS
    common.verify_predictions(design)
    prior = read(PARENT_ROOT / 'summary.json')
    cohort = read(design['arms'][ARMS[0]]['path'])
    row, files = collect(cohort)
    for job in cohort['scannet']:
        result = read(Path(cohort['root']) / 'predictions/scannet' / job['name'] / 'evaluation.json')['metrics']
        for split in ['19', '15', '10']:
            row.setdefault('ScanNet typed mIoU' + split, {})[job['name']] = result['typed'][split]['miou']
    data = dict(prior['per_scene'], native_coordinates=row)
    macro = dict(prior['macro'], native_coordinates={k: float(np.mean(list(v.values()))) for k, v in row.items()})
    differences = {a: {k: paired(data[a][k], row[k], k) for k in row}
                   for a in ['parent', 'previous_common', 'protected_training', 'functional_native']}
    original = read(PARENT_ROOT / 'protected_training/cohort.json')
    invariance = point_invariance(dict(protected_training=original, functional_native=cohort))
    training, total_bytes, matched_sources = [], 0, []
    for job in design['fields']:
        out = ROOT / 'fits' / ARMS[0] / job['key']
        report = read(out / 'training.json'); manifest = read(out / 'field/manifest.json')
        shared_path = PARENT_ROOT / 'fits/functional_native' / job['key'] / 'training.json'
        shared = read(shared_path); parent_manifest = read(Path(job['input']) / 'manifest.json')
        files.extend([out / 'training.json', out / 'field/manifest.json', shared_path])
        if not report['protected_state_unchanged']:
            raise ValueError('coordinate state protection failed')
        for key in ['parent_sha256', 'observations_sha256', 'epochs', 'seed', 'pairs',
                    'temperature', 'trust_weight', 'steps', 'training_regions',
                    'each_region_visits', 'source_frames']:
            if report[key] != shared[key]:
                raise ValueError('coordinate/shared source recipe differs: ' + key)
        for key in ['regions', 'frames', 'seed', 'used_for_optimization', 'used_for_selection']:
            if report['source_development'][key] != shared['source_development'][key]:
                raise ValueError('coordinate/shared development cohort differs: ' + key)
        for channel in ['semantic', 'local']:
            if not np.isclose(report['source_development']['metrics'][channel]['parent_kl'],
                              shared['source_development']['metrics'][channel]['parent_kl'],
                              rtol=0, atol=1e-7):
                raise ValueError('coordinate/shared development parent function differs')
        for key in ['spec', 'elements', 'observed', 'eligible_anchors', 'semantic_native_dim',
                    'local_native_dim', 'semantic_code_dim', 'local_code_dim', 'group_dim',
                    'tensor_bytes', 'external_scene_dependencies']:
            if manifest[key] != parent_manifest[key]:
                raise ValueError('coordinate export capacity/runtime contract differs: ' + key)
        if report['benchmark_queries_used'] or report['source_access_at_query']:
            raise ValueError('coordinate source-only contract differs')
        matched_sources.append(job['key'])
        training.append(dict(key=job['key'], family=job['family'], **{k: v for k, v in report.items() if k != 'history'}))
        total_bytes += manifest['tensor_bytes']
    if len(training) != 30 or sum(r['steps'] for r in training) != 50902:
        raise ValueError('coordinate cohort or source region coverage differs')
    source = {c: dict(parent_kl=float(np.mean([r['source_development']['metrics'][c]['parent_kl'] for r in training])),
        refined_kl=float(np.mean([r['source_development']['metrics'][c]['refined_kl'] for r in training])),
        improved_scenes=sum(r['source_development']['metrics'][c]['refined_kl'] < r['source_development']['metrics'][c]['parent_kl'] for r in training),
        mean_direction_drift=float(np.mean([r['mean_direction_drift'][c] for r in training])),
        updated_observed_fraction=float(np.mean([r['row_exposure'][c]['rows_updated'] / r['row_exposure'][c]['observed_rows'] for r in training])))
        for c in ['semantic', 'local']}
    runtime = read(ROOT / 'validation/runtime/report.json')
    if any(not all(r['exact_replay'].values()) or r['forbidden_open_attempts'] for r in runtime['records']):
        raise ValueError('coordinate runtime check failed')
    files.extend([PARENT_ROOT / 'summary.json', Path(__file__),
        *[ROOT / name for name in ['design.json', 'construction_complete.json', 'predictions_complete.json',
                                  'scoring_complete.json', 'validation/runtime/report.json']]])
    result = dict(complete=True, fields=30, exports=30, arm=ARMS[0], macro=macro, per_scene=data,
        differences=differences, training=training, source_development=source, export_tensor_bytes=total_bytes,
        registered_query_invariance=invariance, runtime=runtime, v10_targets=prior['v10_targets'],
        matched_source_recipes=dict(fields=matched_sources, count=len(matched_sources),
            steps=sum(r['steps'] for r in training), development_regions=sum(r['source_development']['regions'] for r in training),
            export_capacity_unchanged=True, optimizer_parameterization_matched=False),
        source_only=True, source_access_at_query=False, model_promotion=False, per_task_model_selection=False,
        independent_final_test=False, cpu_trial_used_for_benchmarks=False,
        evidence=[receipt(p) for p in sorted(set(files), key=str)])
    write(ROOT / 'summary.json', result)
    write('docs/experiments/2026-10-08-native-coordinate-results.json', result)
    for metric in prior['v10_targets']:
        print(metric, {a: round(100 * macro[a][metric], 4) for a in macro}, flush=True)
    print('source development', source)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
