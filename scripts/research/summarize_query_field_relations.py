"""Full-cohort relation results with matched controls and typed category ablation."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.query_field.storage import sha256
from scripts.research.run_query_field_relations import ROOT, ARMS, read, write, receipt, frozen_design, verify_predictions


def collect(cohort, parent=False):
    root = Path(cohort['root'])
    rows = {}
    files = []
    def add(metric, scene, value): rows.setdefault(metric, {})[scene] = value
    for j in cohort['lerf']:
        path = Path(j['predictions']) / 'evaluation/evaluation.json'
        r = read(path)['metrics']; m = r['shared_membership']; files.append(path)
        add('LERF2D IoU', j['scene'], m['iou'])
        add('LERF2D AP', j['scene'], m['ap'])
        add('LERF actual subset projection IoU', j['scene'], m['selected']['miou'])
        add('LERF native IoU', j['scene'], r['native_generic_relevance']['iou'])
        from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
        precision, recall = [], []
        for item in m['selected']['objects']:
            if item['missing']: raise ValueError('missing LERF projected prediction')
            target = load_binary_mask(Path(item['gt_path']), threshold=10)
            selected = load_binary_mask(Path(item['pred_path']), threshold=10, grayscale=True)
            tp = int((target & selected).sum()); union = int((target | selected).sum())
            if abs(tp / max(union, 1) - item['iou']) > 1e-12: raise ValueError('LERF selection replay differs')
            precision.append(tp / max(int(selected.sum()), 1)); recall.append(tp / max(int(target.sum()), 1))
        add('LERF actual subset precision', j['scene'], float(np.mean(precision)))
        add('LERF actual subset recall', j['scene'], float(np.mean(recall)))
    for j in cohort['nvos']:
        path = root / 'predictions/nvos' / j['name'] / 'evaluation.json'
        r = read(path)['results'][0]; files.append(path)
        add('NVOS response IoU', j['name'], r['retained_response']['foreground_iou'])
        add('NVOS AP', j['name'], r['retained_response']['average_precision'])
        add('NVOS actual subset IoU', j['name'], r['foreground_iou'])
        add('NVOS actual subset precision', j['name'], r['precision'])
        add('NVOS actual subset recall', j['name'], r['recall'])
    for j in cohort['scannet']:
        path = root / 'predictions/scannet' / j['name'] / 'evaluation.json'
        r = read(path)['metrics']; files.append(path)
        for split in ['19', '15', '10']:
            add('ScanNet member mIoU' + split, j['name'], r['member'][split]['miou'])
            add('ScanNet native mIoU' + split, j['name'], r['native'][split]['miou'])
    path = root / 'predictions/uqis/evaluation.json'; r = read(path); files.append(path)
    if r['query_count'] != 268 or r['target_count'] != 67: raise ValueError('UQIS denominator differs')
    for modality in ['text', 'image', 'point_2d', 'point_3d']:
        for scene, value in r['modalities'][modality]['per_scene'].items():
            add('UQIS ' + modality + ' AP', scene, value['average_precision'])
        for scene, value in r['explicit_selection']['modalities'][modality]['per_scene'].items():
            add('UQIS ' + modality + ' IoU', scene, value['iou'])
    return rows, files


def paired(reference, candidate, key):
    if set(reference) != set(candidate): raise ValueError('paired scene cohort differs')
    # NVOS has two correlated prompts from one horns scene. Resample physical
    # scenes as clusters while retaining the case-weighted primary estimand.
    clusters = {}
    for scene in sorted(reference):
        cluster = 'horns' if key.startswith('NVOS') and scene.startswith('horns') else scene
        clusters.setdefault(cluster, []).append(candidate[scene] - reference[scene])
    values = list(clusters.values()); rng = np.random.default_rng(20261008)
    boot = []
    for indices in rng.integers(0, len(values), size=(2000, len(values))):
        boot.append(float(np.mean([v for i in indices for v in values[i]])))
    return dict(delta=float(np.mean([candidate[s] - reference[s] for s in reference])),
                lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975)),
                improved=sum(candidate[s] > reference[s] for s in reference),
                regressed=sum(candidate[s] < reference[s] for s in reference),
                cases=len(reference), clusters=len(values), bootstrap_samples=2000)


def compare_inputs(cohorts):
    """Separate changed member responses from changed native identity inputs."""
    paths = {a: Path(c['root']) / 'predictions/uqis' for a, c in cohorts.items()}
    reports = {a: dict(queries=0, same_status=0, exact_candidates=0,
                      candidate_weight_max_error=0., native_score_max_error=0., changed_native_presence=0)
               for a in ARMS}
    for job in cohorts['parent']['uqis']:
        receipts = {a: {r['query_id']: r for r in read(p / 'receipts' / (job['name'] + '.json'))['queries']}
                    for a, p in paths.items()}
        for qid, original in receipts['parent'].items():
            parent = torch.load(paths['parent'] / 'gaussians' / (qid + '.pt'), map_location='cpu', weights_only=True)
            for a in ARMS:
                values = torch.load(paths[a] / 'gaussians' / (qid + '.pt'), map_location='cpu', weights_only=True)
                r = reports[a]; r['queries'] += 1
                r['same_status'] += original['status'] == receipts[a][qid]['status']
                r['exact_candidates'] += torch.equal(parent['candidates'], values['candidates'])
                if parent['candidate_weights'].shape == values['candidate_weights'].shape and len(parent['candidate_weights']):
                    r['candidate_weight_max_error'] = max(r['candidate_weight_max_error'],
                        float((parent['candidate_weights'] - values['candidate_weights']).abs().max()))
                p, v = parent['native_scores'], values['native_scores']
                if (p is None) != (v is None): r['changed_native_presence'] += 1
                elif p is not None:
                    r['native_score_max_error'] = max(r['native_score_max_error'], float((p - v).abs().max()))
    return reports


def main():
    design = frozen_design(); read(ROOT / 'scoring_complete.json'); verify_predictions(design)
    parent_cohort = read(design['parent']['path'])
    cohorts = {'parent': parent_cohort, **{a: read(design['arms'][a]['path']) for a in ARMS}}
    data, diagnostics, evidence = {}, {}, [ROOT / 'design.json', ROOT / 'construction_complete.json',
                         ROOT / 'predictions_complete.json', ROOT / 'scoring_complete.json']
    for a, c in cohorts.items():
        data[a], files = collect(c, a == 'parent'); evidence.extend(files)
        uqis = read(Path(c['root']) / 'predictions/uqis/evaluation.json')['explicit_selection']
        diagnostics[a] = dict(modalities={m: dict(scene_macro=r['scene_macro'], empty_selections=r['empty_selections'])
                                        for m, r in uqis['modalities'].items()},
                              dominant_selected_identity=uqis['identity_diagnosis'])
    categories = read(ROOT / 'category_evidence/evaluation.json')
    evidence.extend([ROOT / 'category_evidence/evaluation.json', ROOT / 'category_evidence/design.json',
                     ROOT / 'category_evidence/predictions_complete.json'])
    for r in categories['records']:
        for split in ['19', '15', '10']:
            data[r['arm']].setdefault('ScanNet typed mIoU' + split, {})[r['scene']] = r['metrics']['evidence'][split]['miou']
    program_root = ROOT / 'common_program'
    program_barrier = read(program_root / 'predictions_complete.json')
    from scripts.research.run_query_field_program import verify
    verify(program_barrier['records'])
    program = read(program_root / 'uqis/evaluation.json')
    diagnostics['common_program'] = dict(modalities={m: dict(scene_macro=r['scene_macro'], empty_selections=r['empty_selections'])
        for m, r in program['explicit_selection']['modalities'].items()},
        dominant_selected_identity=program['explicit_selection']['identity_diagnosis'])
    if program['query_count'] != 268 or program['target_count'] != 67:
        raise ValueError('integrated program denominator differs')
    data['common_program'] = copy.deepcopy(data[ARMS[1]])
    for modality in ['text', 'image', 'point_2d', 'point_3d']:
        data['common_program']['UQIS ' + modality + ' AP'] = {
            scene: value['average_precision'] for scene, value in program['modalities'][modality]['per_scene'].items()}
        data['common_program']['UQIS ' + modality + ' IoU'] = {
            scene: value['iou'] for scene, value in program['explicit_selection']['modalities'][modality]['per_scene'].items()}
    evidence.extend([program_root / 'design.json', program_root / 'predictions_complete.json',
                     program_root / 'uqis/evaluation.json'])
    macro = {a: {k: float(np.mean(list(v.values()))) for k, v in metrics.items()} for a, metrics in data.items()}
    matched = {k: paired(data[ARMS[0]][k], data[ARMS[1]][k], k) for k in data['parent']}
    versus_parent = {a: {k: paired(data['parent'][k], data[a][k], k) for k in data['parent']} for a in ARMS}
    program_difference = {k: paired(data[ARMS[1]][k], data['common_program'][k], k)
                          for k in data['parent'] if k.startswith('UQIS')}
    native_errors = {a: max(abs(macro[a][k] - macro['parent'][k]) for k in macro[a] if 'native' in k) for a in ARMS}
    training = []
    for job in design['fields']:
        pair = []
        for a in ARMS:
            path = ROOT / 'fits' / a / job['key'] / 'training.json'; r = read(path); evidence.append(path)
            pair.append(r)
            training.append({k: v for k, v in dict(key=job['key'], family=job['family'], **r).items() if k not in {'trace', 'history'}})
        if pair[0]['trace'] != pair[1]['trace']: raise ValueError('training pairs differ')
    baseline = read('output/query_field_20261007/frozen_v10/baseline_a.json')['macro']
    target = {'LERF2D IoU': baseline['lerf']['semantic_iou'],
              'LERF actual subset projection IoU': baseline['lerf']['selected_projection_iou'],
              'NVOS response IoU': baseline['nvos']['iou']}
    target.update({'ScanNet typed mIoU' + s: baseline['scannet']['miou_' + s] for s in ['19', '15', '10']})
    target.update({'UQIS ' + m + ' IoU': baseline['uqis'][m + '_selected_iou']
                   for m in ['text', 'image', 'point_2d', 'point_3d']})
    source_support = {family: {key: sum(r[key] for r in training if r['family'] == family and r['arm'] == ARMS[1])
                              for key in ['training_regions', 'steps', 'cross_view_steps', 'distinct_anchor_steps', 'ambiguity_steps']}
                      for family in ['lerf', 'nvos', 'scannet', 'uqis']}
    runtime = read(ROOT / 'validation/runtime/report.json')
    evidence.append(ROOT / 'validation/runtime/report.json')
    if any(not all(r['exact_replay'].values()) or r['forbidden_open_attempts'] for r in runtime['records']):
        raise ValueError('standalone runtime validation failed')
    inputs = compare_inputs(cohorts)
    report = dict(complete=True, fields=30, fits=60, arms=list(data), macro=macro,
        per_scene=data, matched_difference=matched, versus_parent=versus_parent,
        common_program_difference_from_cross_view=program_difference,
        common_program_primary_metrics=['LERF2D IoU', 'LERF actual subset projection IoU',
            'NVOS response IoU', 'ScanNet typed mIoU19', 'ScanNet typed mIoU15', 'ScanNet typed mIoU10',
            *['UQIS ' + m + ' IoU' for m in ['text', 'image', 'point_2d', 'point_3d']]],
        common_program_rule='cross_view_anchor for every scene; image v2 and bounded category reader fixed before scores; other query paths identical',
        native_macro_max_abs_error=native_errors, training=training, source_support=source_support,
        uqis_diagnostics=diagnostics, input_invariance=inputs, runtime=runtime, v10_targets=target,
        additional_3d_labels=False, source_access_at_query=False, independent_final_test=False,
        accuracy_promotion=False, evidence=[receipt(p) for p in sorted(set(evidence), key=str)])
    write(ROOT / 'summary.json', report)
    write(Path('docs/experiments/2026-10-08-cross-view-relation-results.json'), report)
    print(json.dumps(dict(macro=macro, matched=matched, native_errors=native_errors), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
