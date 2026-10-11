"""Audit the single sampling intervention and replay all actual output masks."""
from pathlib import Path
import numpy as np
import torch
from scripts.research.run_query_field_canonical_sampling import ROOT, PARENT, ARMS, design, sampling_mask
from scripts.research.run_query_field_relations import read, write, receipt, check
from scripts.research.summarize_query_field_relations import paired
from scripts.research.audit_query_field_extension_quality import compare_masks
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask


def main():
    d = design(); cfg = read(d['configuration']['path']); summary = read(ROOT / 'summary.json')
    assert read(PARENT / 'audit.json')['passed']
    for r in read(ROOT / 'predictions_complete.json')['records']: check(r)
    scenes = {}
    for j in cfg['jobs']:
        out = ROOT / 'predictions' / j['name']; parent = PARENT / 'predictions' / j['name']
        record = read(out / 'prediction.json'); check(dict(path=str(out / 'predictions.pt'), sha256=record['predictions_sha256']))
        data = torch.load(out / 'predictions.pt', weights_only=True); old = torch.load(parent / 'predictions.pt', weights_only=True)
        for k, v in data['geometry'].items(): assert torch.equal(v, old['geometry'][k])
        assert torch.equal(data['observed'], old['observed'])
        assert torch.equal(data['selected_by_control']['seed_addition'], old['selected_by_control']['seed_addition'])
        assert torch.equal(data['selected_by_control']['dense_joint'], old['selected_by_control']['admitted_joint'])
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a], data['membership_by_control'][a] > .5)
            assert torch.equal(data['scores'][a], old['scores']['semantic_initial'])
            assert not data['selected_by_control'][a][~data['observed']].any()
        assert not record['target_labels_opened'] and record['initialization_cost_confidence_view_order_exact']
        state = torch.load(out / 'optimization_state.pt', weights_only=True)
        original = torch.load(parent / 'admitted_optimization_state.pt', weights_only=True)
        assert state.keys() == original.keys()
        dense_negative = sparse_negative = 0
        for qi, s in state.items():
            r = original[qi]
            for k in ['start_membership', 'support_cost']: assert torch.equal(s[k], r[k])
            for k in ['confidence', 'view_indices']: assert s[k] == r[k]
            for vi, a, b in zip(s['view_indices'], s['negative'], r['negative']):
                phase = vi % 4; mask = sampling_mask(vi)[phase//2::2, phase%2::2].flatten()
                assert torch.equal(a, b & mask) and not (a & ~b).any()
                dense_negative += int(b.sum()); sparse_negative += int(a.sum())
        fit = read(record['fit']['path']); old_fit = read(parent / 'admitted' / j['scene'] / 'fit.json')
        assert [r['views'] for r in fit['history']] == [r['views'] for r in old_fit['history']]
        for k in ['source_frames', 'steps', 'source_raster', 'sampled_raster', 'pixel_loss', 'support_prior_weight']:
            assert fit[k] == old_fit[k]
        for qi in range(len(data['texts'])):
            if qi not in state: assert torch.equal(data['membership_by_control']['sparse_joint'][:, qi], old['membership_by_control']['semantic_initial'][:, qi])
        e = read(out / 'evaluation/evaluation.json'); metrics = {}; masks = {}; objects = {}
        for a in ARMS:
            rows = e['metrics'][a]['selected']['objects']; objects[a] = {(r['frame'], r['query']): r for r in rows}
            assert len(rows) == len(objects[a]); masks[a] = {}; ious = []; precisions = []; recalls = []
            for key, r in objects[a].items():
                assert not r['missing']
                gt = load_binary_mask(Path(r['gt_path']), threshold=10)
                pred = load_binary_mask(Path(r['pred_path']), threshold=10, grayscale=True)
                tp = int((gt & pred).sum()); iou = tp / max(int((gt | pred).sum()), 1)
                assert abs(iou - r['iou']) < 1e-12
                ious.append(iou); precisions.append(tp / max(int(pred.sum()), 1)); recalls.append(tp / max(int(gt.sum()), 1))
                masks[a][key] = pred
            assert abs(float(np.mean(ious)) - summary['per_scene'][a][j['scene']]) < 1e-12
            metrics[a] = dict(iou=float(np.mean(ious)), precision=float(np.mean(precisions)), recall=float(np.mean(recalls)))
        rows = []
        assert objects['dense_joint'].keys() == objects['sparse_joint'].keys()
        for key, r in objects['dense_joint'].items():
            assert r['gt_path'] == objects['sparse_joint'][key]['gt_path']
            gt = load_binary_mask(Path(r['gt_path']), threshold=10)
            rows.append(dict(frame=key[0], query=key[1], **compare_masks(masks['dense_joint'][key], masks['sparse_joint'][key], gt)))
        totals = {k: sum(r[k] for r in rows) for k in ['added_tp', 'added_fp', 'removed_tp', 'removed_fp']}
        scenes[j['scene']] = dict(metrics=metrics, initializer_cost_confidence_view_order_exact=True,
            active_queries=len(state), active_query_views=sum(r['views'] for r in fit['history']),
            dense_negative_samples=dense_negative, sparse_negative_samples=sparse_negative,
            changes=dict(**totals, beneficial_cases=sum(r['beneficial'] for r in rows), harmful_cases=sum(r['harmful'] for r in rows), rows=rows))
        print('audited', j['scene'], flush=True)
    macro = {a: {k: sum(r['metrics'][a][k] for r in scenes.values()) / 4 for k in ['iou', 'precision', 'recall']} for a in ARMS}
    comparisons = {base: paired({s: r['metrics'][base]['iou'] for s, r in scenes.items()},
        {s: r['metrics']['sparse_joint']['iou'] for s, r in scenes.items()}, 'actual selected-set IoU') for base in ['dense_joint', 'seed_addition']}
    report = dict(passed=True, macro=macro, scenes=scenes, comparisons=comparisons,
        design=receipt(ROOT / 'design.json'), code=receipt(__file__), all_ious_replayed=True,
        source_assisted=True, adaptive_development=True, default_promoted=False, whole_five_task=False)
    write(ROOT / 'audit.json', report); write('docs/experiments/2026-10-11-canonical-sampling-audit.json', report)
    print(macro, flush=True); print(comparisons, flush=True)


if __name__ == '__main__': torch.set_num_threads(4); main()
