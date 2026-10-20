"""N11 fixed-policy and paired full-cohort feedback."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read, require
from radio_gs.response_study.instance_extent_study import check_instance_extent_study, ARMS

REPO = Path(__file__).resolve().parents[2]; PAPER = REPO / 'paper/tmm_response'
LABELS = ['v10', 'Static image', 'Static native', 'Unrestricted image', 'Unrestricted native',
          'v10-anchor image', 'v10-anchor native', 'Scribble-geometry image', 'Scribble-geometry native (primary)']


def main():
    reg = read(REPO / 'configs/research/tmm_response.json'); item = next(i for i in reg['reference_studies'] if i['id'] == 'N11')
    r = check_instance_extent_study(REPO, REPO / reg['study_root'], item, read(REPO / reg['recipe'])['jobs'])
    require(r['complete'], 'N11 incomplete')
    pairs = [('scribble_anchor_native', 'static_native'), ('scribble_anchor_native', 'union_native'),
             ('scribble_anchor_native', 'prior_anchor_native'), ('prior_anchor_native', 'static_native'),
             ('union_native', 'static_native'), ('scribble_anchor_native', 'scribble_anchor_image')]
    r['effects'] = {t: {a + '_vs_' + b: dict(delta_pp=100 * mean(row[a][t]['iou']-row[b][t]['iou'] for row in r['rows'].values()),
        average_precision_delta_pp=100 * mean(row[a][t]['average_precision']-row[b][t]['average_precision'] for row in r['rows'].values()),
        improved=sum(row[a][t]['iou'] > row[b][t]['iou'] + 1e-8 for row in r['rows'].values()),
        harmed=sum(row[a][t]['iou'] < row[b][t]['iou'] - 1e-8 for row in r['rows'].values()),
        per_task_delta_pp={n: 100*(row[a][t]['iou']-row[b][t]['iou']) for n, row in r['rows'].items()}) for a, b in pairs} for t in r['macro']}
    r['source_teacher_macro'] = {n: {a: mean(v['confidence_weighted_mean_huber'][a] for v in rows)
        for a in ['v10', 'static_native', 'union_native', 'prior_anchor_native', 'scribble_anchor_native']} for n, rows in r['source_teacher_checks'].items()}
    (PAPER / 'nvos_instance_extent_feedback.json').write_text(json.dumps(r, indent=2)+'\n')
    rows = [dict(task=n, arm=a, gt_track=t, **v) for n, row in r['rows'].items() for a, tracks in row.items() for t, v in tracks.items()]
    with (PAPER / 'artifacts/nvos_instance_extent.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Variant & IoU & AP & Precision & Recall & LUDVIG GT IoU \\\midrule']
    for a, label in zip(ARMS, LABELS):
        vals = [100*r['macro']['retained_gt_gt127'][a][m] for m in ['iou', 'average_precision', 'precision', 'recall']] + [100*r['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label + ' & ' + ' & '.join(f'{v:.2f}' for v in vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']; (PAPER / 'tables/nvos_instance_extent.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\section{Instance-constrained additive extent recovery}',
        'N11 retains the exact N09 static and signed-SAM image evidence. Three matched policies '
        'accept all proposal components, use the v10 response as signed identity evidence, or '
        'use original signed scribbles transported through fixed geometry (primary). Unmarked '
        'reference coverage has zero authority in the geometry branch. Eight-connected proposal '
        'regions require positive evidence mass greater than negative mass; empty/tied evidence '
        'retains the static response. Admitted pixels take the maximum of static and proposal '
        'probability. This image update is additive, but downstream 3D fitting is not guaranteed '
        'monotone. Every branch uses the same native rays and 120-step fit. '
        'DINOv2, SAM3, target RGB and official negative scribbles are explicitly retained inputs. '
        'All 72 predictions precede GT; other benchmarks remain unchanged references.',
        r'\begin{center}\small\resizebox{\linewidth}{!}{\input{tables/nvos_instance_extent.tex}}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule', r'Task & Static & Unrestricted & v10 anchor & Scribble geometry \\\midrule']
    for n, row in r['rows'].items():
        lines.append(n.replace('_', r'\_') + ' & ' + ' & '.join(f"{100*row[a]['retained_gt_gt127']['iou']:.2f}" for a in ['static_native', 'union_native', 'prior_anchor_native', 'scribble_anchor_native']) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    for name, v in r['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_', ' ') + f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    if r['spatial'] is not None:
        lines += [r'\begin{figure}[ht]\centering\includegraphics[width=.85\linewidth,height=.75\textheight,keepaspectratio]{../../output/tmm_response_20261016/study/nvos_instance_extent_spatial_v1/errors.png}',
            r'\caption{All eight N11 tasks. Green TP, red FP, blue FN; native-grid display resizing only. Scores use original GT resolution.}\end{figure}']
    (PAPER / 'sections/nvos_instance_extent.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=r['macro'], effects=r['effects']), indent=2))


if __name__ == '__main__': main()
