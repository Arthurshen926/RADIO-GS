"""Full-task comparison of retained partial-relation functions."""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.research.run_query_field_paired_functions import ROOT, ARMS, read, write, receipt


def main():
    r = read(ROOT / 'summary.json'); out = ROOT / 'diagnostics'; out.mkdir(exist_ok=True)
    metrics = [('LERF2D IoU', 'LERF 2D response'), ('LERF actual subset projection IoU', 'LERF selected-set projection'),
        ('NVOS response IoU', 'NVOS response'), ('ScanNet typed mIoU19', 'ScanNet 19 classes'),
        ('UQIS text IoU', 'UQIS text'), ('UQIS image IoU', 'UQIS image'),
        ('UQIS point_2d IoU', 'UQIS 2D point'), ('UQIS point_3d IoU', 'UQIS 3D point')]
    arms = ['protected_training', 'conditional_relation', 'joint_conditional_relation', ARMS[0]]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True); table = []
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [100*r['macro'][a][metric] for a in arms]; target = 100*r['v10_targets'][metric]
        ax.bar(range(4), values, color=['#286b9d', '#d9863d', '#8c9198', '#448767'], width=.7)
        ax.axhline(target, color='#333333', linestyle='--', linewidth=1.1)
        for i, v in enumerate(values): ax.text(i, v+1, f'{v:.2f}', ha='center', fontsize=8)
        ax.set_xticks(range(4), ['Render', 'Cond', 'Joint', 'Paired'])
        ax.set_ylim(0, min(100, max(target, *values)+12)); ax.set_title(title, fontsize=10); ax.set_ylabel('IoU (%)')
        ax.spines[['top', 'right']].set_visible(False)
        table.append(dict(metric=metric, **dict(zip(arms, values)), v10_reference=target))
    fig.suptitle('Preserving partial support-to-Gaussian functions: complete development cohort', fontsize=13)
    fig.supxlabel('Paired: uncompressed relation function reference with a fixed rendered prior; no new training\n'
                  'Dashed: v10 with different source/supervision contracts. LERF selected-set metric is a 2D projection.', fontsize=9)
    for suffix in ['png', 'pdf']: fig.savefig(out / ('full_cohort_metrics.'+suffix), dpi=180)
    plt.close(fig)
    with (out / 'full_cohort_metrics.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    write(out / 'full_cohort_metrics.json', dict(source=receipt(ROOT / 'summary.json'), table=table,
        scene_selection=False, query_selection=False,
        artifacts=[receipt(out / ('full_cohort_metrics.'+s)) for s in ['png', 'pdf', 'csv']]))


if __name__ == '__main__': main()
