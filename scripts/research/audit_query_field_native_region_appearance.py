"""Replay actual masks and attribute a frozen appearance-only intervention."""
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_native_region_appearance import ROOT, PARENT, VARIANTS, CONTROLS, ARMS, design
from scripts.research.run_query_field_relations import read, write, receipt, check
from scripts.research.audit_query_field_extension_quality import compare_masks
from scripts.research.summarize_query_field_relations import paired


def main():
    d = design(); cfg = read(d['configuration']['path']); summary = read(ROOT / 'summary.json'); scenes = {}
    for r in read(ROOT / 'predictions_complete.json')['records']: check(r)
    for j in cfg['jobs']:
        out = ROOT / 'predictions' / j['name']; r = read(out / 'prediction.json')
        check(dict(path=str(out / 'predictions.pt'), sha256=r['predictions_sha256']))
        data = torch.load(out / 'predictions.pt', weights_only=True)
        previous = torch.load(PARENT / 'predictions' / j['name'] / 'predictions.pt', weights_only=True)
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a], data['membership_by_control'][a] > .5)
            assert torch.equal(data['scores'][a], previous['scores']['semantic_initial'])
        for a in CONTROLS: assert torch.equal(data['selected_by_control'][a], previous['selected_by_control'][a])
        initial = data['selected_by_control']['frozen_visibility']
        for variant in VARIANTS:
            assert not bool((initial & ~data['selected_by_control'][variant + '_addition']).any())
        assert not r['target_labels_opened'] and not r['source_access']
        e = read(out / 'evaluation/evaluation.json'); metrics = {}; objects = {}; masks = {}
        for a in ARMS:
            rows = e['metrics'][a]['selected']['objects']; objects[a] = {(x['frame'], x['query']): x for x in rows}
            assert len(objects[a]) == len(rows)
            ious = []; precision = []; recall = []; masks[a] = {}
            for key, x in objects[a].items():
                assert not x['missing']
                gt = load_binary_mask(Path(x['gt_path']), threshold=10)
                pred = load_binary_mask(Path(x['pred_path']), threshold=10, grayscale=True)
                tp = int((gt & pred).sum()); iou = tp / max(int((gt | pred).sum()), 1)
                assert abs(iou - x['iou']) < 1e-12
                ious.append(iou); precision.append(tp / max(int(pred.sum()), 1)); recall.append(tp / max(int(gt.sum()), 1))
                masks[a][key] = pred
            assert abs(float(np.mean(ious)) - summary['per_scene'][a][j['scene']]) < 1e-12
            metrics[a] = dict(iou=float(np.mean(ious)), precision=float(np.mean(precision)), recall=float(np.mean(recall)))
        extension = {}
        for baseline in ['frozen_visibility', 'seed_addition', 'conditional_addition']:
            for variant in VARIANTS:
                arm = variant + '_addition'; rows = []
                for key, x in objects[baseline].items():
                    assert x['gt_path'] == objects[arm][key]['gt_path']
                    gt = load_binary_mask(Path(x['gt_path']), threshold=10)
                    rows.append(dict(frame=key[0], query=key[1], **compare_masks(masks[baseline][key], masks[arm][key], gt)))
                tp = sum(x['added_tp'] for x in rows); fp = sum(x['added_fp'] for x in rows)
                extension[arm + '_vs_' + baseline] = dict(added_tp=tp, added_fp=fp, added_precision=tp / max(tp + fp, 1),
                    removed_pixels=sum(x['removed_pixels'] for x in rows), beneficial_cases=sum(x['beneficial'] for x in rows),
                    harmful_cases=sum(x['harmful'] for x in rows), rows=rows)
        old_domain = read(PARENT / 'domain_diagnostics' / (j['scene'] + '.json'))['distributions']['all_pairs']
        old_query_details = read(PARENT / 'predictions' / j['name'] / 'prediction.json')['diagnostics']
        feature_checks = {}
        for variant in VARIANTS:
            detail = r['diagnostics'][variant]
            native = [v for x in detail['queries'] for v in x['admitted_native_probabilities']]
            assert detail['feature_pairs'] == old_domain['pairs']
            for key, value in detail['feature_medians'].items():
                if key != 'native_appearance_cosine':
                    assert abs(value - old_domain['features'][key]['actual_median']) < 1e-8, key
            feature_checks[variant] = dict(other_feature_medians_exact=True,
                appearance_median=detail['feature_medians']['native_appearance_cosine'],
                previous_appearance_median=old_domain['features']['native_appearance_cosine']['actual_median'],
                training_appearance_median=old_domain['features']['native_appearance_cosine']['training_median'],
                applied=sum(x['applied'] for x in detail['queries']),
                admitted_regions=len(native), native_nonpositive_admitted_regions=sum(v <= .5 for v in native),
                changed_admission_queries=sum(x.get('admitted', []) != y.get('admitted', []) for x, y in zip(detail['queries'], old_query_details)))
        scenes[j['scene']] = dict(metrics=metrics, extension=extension, feature_checks=feature_checks,
            added_members={a: int((data['selected_by_control'][a] & ~initial).sum()) for a in ARMS[2:]},
            removed_members={a: int((initial & ~data['selected_by_control'][a]).sum()) for a in ARMS[2:]})
        print('audited', j['scene'], flush=True)
    macro = {a: {k: sum(s['metrics'][a][k] for s in scenes.values()) / 4 for k in ['iou', 'precision', 'recall']} for a in ARMS}
    comparisons = {a + '_vs_' + base: paired({s: v['metrics'][base]['iou'] for s, v in scenes.items()},
        {s: v['metrics'][a]['iou'] for s, v in scenes.items()}, 'actual selected-set IoU')
        for a in ARMS[4:] for base in ['conditional_addition', 'seed_addition']}
    report = dict(passed=True, macro=macro, scenes=scenes, comparisons=comparisons, design=receipt(ROOT / 'design.json'),
        code=receipt(__file__), all_ious_replayed=True, posthoc_extension_diagnosis=True, default_promoted=False, whole_five_task=False)
    write(ROOT / 'audit.json', report); write('docs/experiments/2026-10-11-native-region-appearance-audit.json', report)
    print(macro, flush=True); print(comparisons, flush=True)


if __name__ == '__main__': torch.set_num_threads(4); main()
