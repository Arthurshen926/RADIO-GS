"""Actual-mask audit and complete two-factor attribution; no inference changes."""
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_identity_scope import ROOT, PARENT, ARMS, design
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
        sets = data['selected_by_control']; seed = sets['seed_addition']
        for a in ARMS:
            assert torch.equal(sets[a], data['membership_by_control'][a] > .5)
            assert torch.equal(data['scores'][a], previous['scores']['semantic_initial'])
        for a, b in [('frozen_visibility', 'frozen_visibility'), ('seed_addition', 'seed_addition'), ('unqualified_addition', 'native_448_addition')]:
            assert torch.equal(sets[a], previous['selected_by_control'][b])
        assert torch.equal(sets['unqualified_seeded'], seed | sets['unqualified_addition'])
        assert torch.equal(sets['native_seeded'], seed | sets['native_addition'])
        assert r['public_entry_sets_exact'] and r['public_score_max_error'] <= 2e-6
        assert not r['target_labels_opened'] and not r['source_access']
        e = read(out / 'evaluation/evaluation.json'); metrics = {}; masks = {}; objects = {}
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
                ious.append(iou); precision.append(tp / max(int(pred.sum()), 1)); recall.append(tp / max(int(gt.sum()), 1)); masks[a][key] = pred
            assert abs(float(np.mean(ious)) - summary['per_scene'][a][j['scene']]) < 1e-12
            metrics[a] = dict(iou=float(np.mean(ious)), precision=float(np.mean(precision)), recall=float(np.mean(recall)))
        extensions = {}
        for arm in ARMS[2:]:
            rows = []
            for key, x in objects['seed_addition'].items():
                assert x['gt_path'] == objects[arm][key]['gt_path']
                gt = load_binary_mask(Path(x['gt_path']), threshold=10)
                rows.append(dict(frame=key[0], query=key[1], **compare_masks(masks['seed_addition'][key], masks[arm][key], gt)))
            tp = sum(x['added_tp'] for x in rows); fp = sum(x['added_fp'] for x in rows)
            extensions[arm] = dict(added_tp=tp, added_fp=fp, added_precision=tp / max(tp + fp, 1),
                removed_pixels=sum(x['removed_pixels'] for x in rows), beneficial_cases=sum(x['beneficial'] for x in rows),
                harmful_cases=sum(x['harmful'] for x in rows), rows=rows)
        scenes[j['scene']] = dict(metrics=metrics, extension_vs_seed=extensions,
            applied=sum(x['applied'] for x in r['diagnostics']),
            candidate_pairs=sum(x.get('candidate_pairs', 0) for x in r['diagnostics']),
            native_rejected_pairs=sum(x.get('native_rejected_pairs', 0) for x in r['diagnostics']),
            added_members_vs_seed={a: int((sets[a] & ~seed).sum()) for a in ARMS[2:]},
            removed_members_vs_seed={a: int((seed & ~sets[a]).sum()) for a in ARMS[2:]})
        print('audited', j['scene'], flush=True)
    macro = {a: {k: sum(s['metrics'][a][k] for s in scenes.values()) / 4 for k in ['iou', 'precision', 'recall']} for a in ARMS}
    contrasts = [('unqualified_addition', 'unqualified_seeded'), ('native_addition', 'native_seeded'),
                 ('unqualified_addition', 'native_addition'), ('unqualified_seeded', 'native_seeded'), ('seed_addition', 'native_seeded')]
    comparisons = {b + '_vs_' + a: paired({s: v['metrics'][a]['iou'] for s, v in scenes.items()},
        {s: v['metrics'][b]['iou'] for s, v in scenes.items()}, 'actual selected-set IoU') for a, b in contrasts}
    interaction = {s: v['metrics']['native_seeded']['iou'] - v['metrics']['native_addition']['iou'] -
        v['metrics']['unqualified_seeded']['iou'] + v['metrics']['unqualified_addition']['iou'] for s, v in scenes.items()}
    report = dict(passed=True, macro=macro, scenes=scenes, comparisons=comparisons, interaction_by_scene=interaction,
        interaction_macro=sum(interaction.values()) / 4, design=receipt(ROOT / 'design.json'), code=receipt(__file__),
        all_ious_replayed=True, posthoc_extension_diagnosis=True, default_promoted=False, whole_five_task=False)
    write(ROOT / 'audit.json', report); write('docs/experiments/2026-10-11-query-identity-scope-audit.json', report)
    print(macro, flush=True); print(comparisons, flush=True)


if __name__ == '__main__': torch.set_num_threads(4); main()
