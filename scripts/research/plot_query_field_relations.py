"""Post-score illustrations of the largest LERF selection loss and gain.

Case selection is explicitly diagnostic. It changes no model, prediction or
reported full-cohort denominator.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_relations import ROOT, ARMS, read, write, receipt


def main():
    summary = read(ROOT / 'summary.json')
    cohorts = {'parent': read('configs/query_field/five_benchmark_v1/cohort.json'),
               **{a: read(ROOT / a / 'cohort.json') for a in ARMS}}
    rows = {}
    for a, c in cohorts.items():
        rows[a] = {}
        for job in c['lerf']:
            result = read(Path(job['predictions']) / 'evaluation/evaluation.json')
            for r in result['metrics']['shared_membership']['selected']['objects']:
                rows[a][(job['scene'], r['frame'], r['query'])] = r
    keys = sorted(rows['parent'], key=lambda k: rows[ARMS[1]][k]['iou'] - rows['parent'][k]['iou'])
    chosen = [keys[0], keys[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    records = []
    labels = ['Ground truth', 'Parent', 'Matched local control', 'Cross-view common program']
    for i, key in enumerate(chosen):
        truth_path = Path(rows['parent'][key]['gt_path'])
        target = load_binary_mask(truth_path, threshold=10)
        predictions = {a: load_binary_mask(Path(rows[a][key]['pred_path']), threshold=10, grayscale=True)
                       for a in cohorts}
        support = target.copy()
        for selected in predictions.values(): support |= selected
        y, x = np.where(support)
        box = (slice(max(0, y.min()-10), min(target.shape[0], y.max()+11)),
               slice(max(0, x.min()-10), min(target.shape[1], x.max()+11)))
        axes[i, 0].imshow(target[box], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(labels[0] + '\n' + ' / '.join(key), fontsize=9)
        metrics = {}
        for col, a in enumerate(cohorts, 1):
            selected = predictions[a]; tp = target & selected; fp = selected & ~target; fn = target & ~selected
            image = np.full((*target.shape, 3), .08)
            image[tp] = [.23, .76, .49]; image[fp] = [.96, .35, .30]; image[fn] = [.30, .53, .95]
            precision = tp.sum() / max(selected.sum(), 1); recall = tp.sum() / max(target.sum(), 1)
            axes[i, col].imshow(image[box])
            axes[i, col].set_title(labels[col] + f"\nIoU {rows[a][key]['iou']*100:.1f} | P {precision*100:.1f} | R {recall*100:.1f}", fontsize=9)
            job = next(j for j in cohorts[a]['lerf'] if j['scene'] == key[0])
            payload = torch.load(Path(job['predictions']) / 'predictions.pt', map_location='cpu', weights_only=True)
            selected_gaussians = int(payload['selected'][:, payload['texts'].index(key[2])].sum())
            metrics[a] = dict(iou=rows[a][key]['iou'], precision=float(precision), recall=float(recall),
                              selected_gaussians=selected_gaussians,
                              prediction=receipt(Path(rows[a][key]['pred_path'])))
        for ax in axes[i]: ax.axis('off')
        records.append(dict(scene=key[0], frame=key[1], query=key[2], metrics=metrics, truth=receipt(truth_path)))
    fig.suptitle('Actual selected-subset projection: green = TP, red = FP, blue = FN\n'
                 'Largest decline and gain among all 208 query-views; chosen after scoring, no prediction changes', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .93])
    out = ROOT / 'diagnostics'; out.mkdir(exist_ok=True)
    fig.savefig(out / 'lerf_selection_errors.png', dpi=150); plt.close(fig)
    write(out / 'lerf_selection_errors.json', dict(records=records, selection_rule='full-cohort min/max IoU change',
        summary=receipt(ROOT / 'summary.json'), figure=receipt(out / 'lerf_selection_errors.png'),
        code=receipt(Path(__file__)), posthoc_diagnosis=True, model_changed=False))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
