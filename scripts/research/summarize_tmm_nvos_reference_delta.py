"""N13 fixed reference completion and paired full-cohort feedback."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read, require
from radio_gs.response_study.reference_delta_study import check_reference_delta_study, ARMS

REPO = Path(__file__).resolve().parents[2]; PAPER = REPO / 'paper/tmm_response'
LABELS = ['v10', 'Static native', 'Full replacement native', 'Probability delta image',
          'Probability delta native (primary)', 'Log-odds delta image', 'Log-odds delta native']


def main():
    reg = read(REPO / 'configs/research/tmm_response.json'); item = next(i for i in reg['reference_studies'] if i['id'] == 'N13')
    r = check_reference_delta_study(REPO, REPO / item['study_root'], item, read(REPO / reg['recipe'])['jobs'])
    require(r['complete'], 'N13 incomplete')
    pairs = [('probability_native', 'static_native'), ('probability_native', 'constrained_memory_native'),
             ('probability_native', 'log_odds_native'), ('probability_native', 'probability_image')]
    r['effects'] = {t: {a + '_vs_' + b: dict(delta_pp=100 * mean(row[a][t]['iou']-row[b][t]['iou'] for row in r['rows'].values()),
        average_precision_delta_pp=100 * mean(row[a][t]['average_precision']-row[b][t]['average_precision'] for row in r['rows'].values()),
        improved=sum(row[a][t]['iou'] > row[b][t]['iou'] + 1e-8 for row in r['rows'].values()),
        harmed=sum(row[a][t]['iou'] < row[b][t]['iou'] - 1e-8 for row in r['rows'].values()),
        per_task_delta_pp={n: 100*(row[a][t]['iou']-row[b][t]['iou']) for n, row in r['rows'].items()}) for a, b in pairs} for t in r['macro']}
    r['previous_candidate_effects'] = {t: dict(delta_pp=100*mean(row['probability_native'][t]['iou']-
        r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou'] for n,row in r['rows'].items()),
        improved=sum(row['probability_native'][t]['iou'] > r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou']+1e-8 for n,row in r['rows'].items()),
        harmed=sum(row['probability_native'][t]['iou'] < r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou']-1e-8 for n,row in r['rows'].items())) for t in r['macro']}
    r['source_teacher_macro'] = {n: {a: mean(v['confidence_weighted_mean_huber'][a] for v in rows)
        for a in ['v10', 'probability_native', 'log_odds_native']} for n, rows in r['source_teacher_checks'].items()}
    (PAPER / 'nvos_reference_delta_feedback.json').write_text(json.dumps(r, indent=2)+'\n')
    rows = [dict(task=n, arm=a, gt_track=t, **v) for n, row in r['rows'].items() for a, tracks in row.items() for t, v in tracks.items()]
    with (PAPER / 'artifacts/nvos_reference_delta.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Variant & IoU & AP & Precision & Recall & LUDVIG GT IoU \\\midrule']
    for a, label in zip(ARMS, LABELS):
        vals = [100*r['macro']['retained_gt_gt127'][a][m] for m in ['iou', 'average_precision', 'precision', 'recall']] + [100*r['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label + ' & ' + ' & '.join(f'{v:.2f}' for v in vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']; (PAPER / 'tables/nvos_reference_delta.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\section{Matched reference-intervention residuals}',
        'N13 was fixed before N12 target scores. It reuses the exact static response b and '
        'paired original/completed-reference memory responses p and q. The primary response '
        'is clip(b+q-p,0,1); a fixed secondary uses the equivalent log-odds difference. '
        'Unchanged paired teacher values retain b exactly. The intervention may add or remove '
        'foreground and does not guarantee correctness. All branches undergo actual native-ray '
        '3D fitting; 56 predictions precede target GT. Full N12 replacement is the matched '
        'control; earlier N11 is a separate candidate, not an additive causal comparison.',
        r'\begin{center}\small\resizebox{\linewidth}{!}{\input{tables/nvos_reference_delta.tex}}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule', r'Task & Static & Replacement & Probability delta & Log-odds delta \\\midrule']
    for n, row in r['rows'].items():
        lines.append(n.replace('_', r'\_') + ' & ' + ' & '.join(f"{100*row[a]['retained_gt_gt127']['iou']:.2f}" for a in ['static_native', 'constrained_memory_native', 'probability_native', 'log_odds_native']) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    for name, v in r['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_', ' ') + f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    if r['spatial'] is not None:
        lines += [r'\begin{figure}[ht]\centering\includegraphics[width=.85\linewidth,height=.75\textheight,keepaspectratio]{../../output/tmm_response_20261018/study/nvos_reference_delta_spatial_v1/errors.png}',
            r'\caption{All eight N13 tasks. Green TP, red FP, blue FN; native-grid display resizing only. Scores use original GT resolution.}\end{figure}']
    (PAPER / 'sections/nvos_reference_delta.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=r['macro'], effects=r['effects']), indent=2))


if __name__ == '__main__': main()
