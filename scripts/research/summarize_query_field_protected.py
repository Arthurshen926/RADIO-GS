"""Full five-family results and causal contrasts for protected-native fits."""
from pathlib import Path

import numpy as np
import torch

from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_protected import ROOT, PREVIOUS, ARMS, frozen_design, read, write, receipt, check
from scripts.research.summarize_query_field_relations import collect, paired


def point_invariance(cohorts):
    """Native-only construction must preserve all registered query outputs."""
    p, f = [Path(cohorts[a]['root']) / 'predictions' for a in ARMS[1:]]
    result = dict(uqis_queries=0, nvos_queries=0, unsupported_uqis=0, exact=True)
    for job in cohorts[ARMS[1]]['uqis']:
        reports = [read(root / 'uqis/receipts' / (job['name'] + '.json')) for root in [p, f]]
        other = {q['query_id']: q for q in reports[1]['queries']}
        for query in reports[0]['queries']:
            if query['modality'] not in {'point_2d', 'point_3d'}:
                continue
            qid = query['query_id']; result['uqis_queries'] += 1
            result['unsupported_uqis'] += query['status'] != 'ok'
            if query['status'] != other[qid]['status']:
                raise ValueError('native refinement changed registered support status')
            a, b = [torch.load(root / 'uqis/gaussians' / (qid + '.pt'), weights_only=True) for root in [p, f]]
            for key in ['membership', 'selected', 'candidates', 'candidate_weights']:
                if not torch.equal(a[key], b[key]):
                    raise ValueError('native refinement changed registered UQIS prediction')
    for job in cohorts[ARMS[1]]['nvos']:
        name = job['name']; result['nvos_queries'] += 1
        a, b = [torch.load(root / 'nvos' / name / name / 'predictions.pt', weights_only=True) for root in [p, f]]
        for key in ['membership', 'selected']:
            if not torch.equal(a[key], b[key]):
                raise ValueError('native refinement changed registered NVOS prediction')
    if result['uqis_queries'] != 134 or result['nvos_queries'] != 8:
        raise ValueError('registered query denominator changed')
    return result


def main():
    design = frozen_design(); read(ROOT / 'scoring_complete.json')
    common.ROOT = ROOT; common.ARMS = ARMS
    common.verify_predictions(design)
    previous = read(PREVIOUS / 'summary.json'); check(design['previous_summary'])
    data = {k: previous['per_scene'][a] for k, a in [('parent', 'parent'), ('previous_common', 'common_program')]}
    cohorts = {a: read(design['arms'][a]['path']) for a in ARMS}
    evidence = [ROOT / f for f in ['design.json', 'construction_complete.json', 'predictions_complete.json', 'scoring_complete.json']]
    diagnostics = {}
    for arm, cohort in cohorts.items():
        data[arm], paths = collect(cohort); evidence.extend(paths)
        for job in cohort['scannet']:
            metrics = read(Path(cohort['root']) / 'predictions/scannet' / job['name'] / 'evaluation.json')['metrics']
            for split in ['19', '15', '10']:
                data[arm].setdefault('ScanNet typed mIoU' + split, {})[job['name']] = metrics['typed'][split]['miou']
        uqis = read(Path(cohort['root']) / 'predictions/uqis/evaluation.json')['explicit_selection']
        diagnostics[arm] = dict(modalities={m: dict(scene_macro=r['scene_macro'], empty_selections=r['empty_selections'])
                                          for m, r in uqis['modalities'].items()},
                                dominant_selected_identity=uqis['identity_diagnosis'])
    macro = {a: {k: float(np.mean(list(v.values()))) for k, v in row.items()} for a, row in data.items()}
    comparisons = [('previous_common', ARMS[0]), (ARMS[0], ARMS[1]), (ARMS[1], ARMS[2]),
                   ('previous_common', ARMS[1]), ('previous_common', ARMS[2]), ('parent', ARMS[2])]
    differences = {a + '__to__' + b: {k: paired(data[a][k], data[b][k], k) for k in data[a]} for a, b in comparisons}
    training = []; total_bytes = {a: 0 for a in ARMS}
    for job in design['fields']:
        for arm in ARMS:
            out = ROOT / 'fits' / arm / job['key']
            report = read(out / 'training.json'); manifest = read(out / 'field/manifest.json')
            evidence.extend([out / 'training.json', out / 'field/manifest.json'])
            training.append(dict(arm=arm, key=job['key'], family=job['family'],
                                 **{k: v for k, v in report.items() if k not in {'arm', 'history', 'trace'}}))
            total_bytes[arm] += manifest['tensor_bytes']
    native = [r for r in training if r['arm'] == 'functional_native']
    source_function = {c: dict(
        parent_kl=float(np.mean([r['source_development']['metrics'][c]['parent_kl'] for r in native])),
        refined_kl=float(np.mean([r['source_development']['metrics'][c]['refined_kl'] for r in native])),
        scenes_improved=sum(r['source_development']['metrics'][c]['refined_kl'] < r['source_development']['metrics'][c]['parent_kl'] for r in native),
        mean_direction_drift=float(np.mean([r['mean_direction_drift'][c] for r in native]))) for c in ['semantic', 'local']}
    runtime_path = ROOT / 'validation/runtime/report.json'; runtime = read(runtime_path); evidence.append(runtime_path)
    if any(not all(r['exact_replay'].values()) or r['forbidden_open_attempts'] for r in runtime['records']):
        raise ValueError('runtime replay failed')
    report = dict(complete=True, fields=30, exports=90, primary_arm='functional_native',
        arms=list(data), macro=macro, per_scene=data, differences=differences,
        training=training, source_development=source_function, export_tensor_bytes=total_bytes,
        registered_query_invariance=point_invariance(cohorts), runtime=runtime, uqis_diagnostics=diagnostics,
        v10_targets=previous['v10_targets'], source_only=True, source_access_at_query=False,
        model_promotion=False, independent_final_test=False, per_task_model_selection=False,
        evidence=[receipt(p) for p in sorted(set(evidence), key=str)])
    write(ROOT / 'summary.json', report)
    write('docs/experiments/2026-10-08-protected-native-results.json', report)
    for metric in previous['common_program_primary_metrics']:
        print(metric, {a: round(100 * macro[a][metric], 4) for a in macro}, flush=True)
    print('source development', source_function)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
