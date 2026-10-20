"""N14 fixed reference completion and paired full-cohort feedback."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read, require
from radio_gs.response_study.reference_admission_study import check_reference_admission_study, ARMS

REPO = Path(__file__).resolve().parents[2]; PAPER = REPO / 'paper/tmm_response'
LABELS = ['v10', 'Static image', 'Static native', 'Feasibility image', 'Feasibility native (primary)',
          'Source quality image', 'Source quality native (secondary)']


def main():
    reg = read(REPO / 'configs/research/tmm_response.json'); item = next(i for i in reg['reference_studies'] if i['id'] == 'N14')
    r = check_reference_admission_study(REPO, REPO / item['study_root'], item, read(REPO / reg['recipe'])['jobs'])
    require(r['complete'], 'N14 incomplete')
    pairs = [('feasible_native', 'static_native'), ('quality_native', 'static_native'),
             ('quality_native', 'feasible_native'), ('feasible_native', 'feasible_image'),
             ('quality_native', 'quality_image')]
    r['effects'] = {t: {a + '_vs_' + b: dict(delta_pp=100 * mean(row[a][t]['iou']-row[b][t]['iou'] for row in r['rows'].values()),
        average_precision_delta_pp=100 * mean(row[a][t]['average_precision']-row[b][t]['average_precision'] for row in r['rows'].values()),
        improved=sum(row[a][t]['iou'] > row[b][t]['iou'] + 1e-8 for row in r['rows'].values()),
        harmed=sum(row[a][t]['iou'] < row[b][t]['iou'] - 1e-8 for row in r['rows'].values()),
        per_task_delta_pp={n: 100*(row[a][t]['iou']-row[b][t]['iou']) for n, row in r['rows'].items()}) for a, b in pairs} for t in r['macro']}
    r['previous_candidate_effects'] = {t: dict(delta_pp=100*mean(row['feasible_native'][t]['iou']-
        r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou'] for n,row in r['rows'].items()),
        improved=sum(row['feasible_native'][t]['iou'] > r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou']+1e-8 for n,row in r['rows'].items()),
        harmed=sum(row['feasible_native'][t]['iou'] < r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou']-1e-8 for n,row in r['rows'].items())) for t in r['macro']}
    r['source_teacher_macro'] = {n: {a: mean(v['confidence_weighted_mean_huber'][a] for v in rows)
        for a in ['v10', 'static_native', 'feasible_native', 'quality_native']} for n, rows in r['source_teacher_checks'].items()}
    (PAPER / 'nvos_reference_admission_feedback.json').write_text(json.dumps(r, indent=2)+'\n')
    rows = [dict(task=n, arm=a, gt_track=t, **v) for n, row in r['rows'].items() for a, tracks in row.items() for t, v in tracks.items()]
    with (PAPER / 'artifacts/nvos_reference_admission.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Variant & IoU & AP & Precision & Recall & LUDVIG GT IoU \\\midrule']
    for a, label in zip(ARMS, LABELS):
        vals = [100*r['macro']['retained_gt_gt127'][a][m] for m in ['iou', 'average_precision', 'precision', 'recall']] + [100*r['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label + ' & ' + ' & '.join(f'{v:.2f}' for v in vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']; (PAPER / 'tables/nvos_reference_admission.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\section{Source-conditioned reference admission}',
        'N14 is an adaptive development study designed after N12 analysis. A fixed feasibility '
        'policy retains semantic completion when it satisfies every original signed annotation, '
        'otherwise using constrained completion. A source-quality control instead chooses the '
        'larger SAM-predicted reference mask quality, with deterministic semantic-first ties. '
        'Both use only reference evidence, without scene names or target scores. Each selected '
        'target prediction is freshly replayed and hash-checked against N12, then actually fitted '
        'and re-rendered in 3D. All 56 predictions precede scoring. This is not an independent '
        'final test or a task-wise oracle. Other four benchmarks remain unchanged references.',
        r'\begin{center}\small\resizebox{\linewidth}{!}{\input{tables/nvos_reference_admission.tex}}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule', r'Task & v10 & Static & Feasibility & Source quality \\\midrule']
    for n, row in r['rows'].items():
        lines.append(n.replace('_', r'\_') + ' & ' + ' & '.join(f"{100*row[a]['retained_gt_gt127']['iou']:.2f}" for a in ['v10', 'static_native', 'feasible_native', 'quality_native']) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    for name, v in r['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_', ' ') + f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    if r['spatial'] is not None:
        lines += [r'\begin{figure}[ht]\centering\includegraphics[width=.85\linewidth,height=.75\textheight,keepaspectratio]{../../output/tmm_response_20261018/study/nvos_reference_admission_spatial_v1/errors.png}',
            r'\caption{All eight N14 tasks. Green TP, red FP, blue FN; native-grid display resizing only. Scores use original GT resolution.}\end{figure}']
    (PAPER / 'sections/nvos_reference_admission.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=r['macro'], effects=r['effects']), indent=2))


if __name__ == '__main__': main()
