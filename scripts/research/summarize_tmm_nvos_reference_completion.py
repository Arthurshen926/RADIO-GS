"""N12 fixed reference completion and paired full-cohort feedback."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read, require
from radio_gs.response_study.reference_completion_study import check_reference_completion_study, ARMS

REPO = Path(__file__).resolve().parents[2]; PAPER = REPO / 'paper/tmm_response'
LABELS = ['v10', 'Static image', 'Static native', 'Legacy memory image', 'Legacy memory native',
          'Semantic memory image', 'Semantic memory native', 'Constrained memory image', 'Constrained memory native (primary)']


def main():
    reg = read(REPO / 'configs/research/tmm_response.json'); item = next(i for i in reg['reference_studies'] if i['id'] == 'N12')
    r = check_reference_completion_study(REPO, REPO / item['study_root'], item, read(REPO / reg['recipe'])['jobs'])
    require(r['complete'], 'N12 incomplete')
    pairs = [('constrained_memory_native', 'static_native'), ('constrained_memory_native', 'legacy_memory_native'),
             ('constrained_memory_native', 'semantic_memory_native'), ('semantic_memory_native', 'legacy_memory_native'),
             ('constrained_memory_native', 'constrained_memory_image')]
    r['effects'] = {t: {a + '_vs_' + b: dict(delta_pp=100 * mean(row[a][t]['iou']-row[b][t]['iou'] for row in r['rows'].values()),
        average_precision_delta_pp=100 * mean(row[a][t]['average_precision']-row[b][t]['average_precision'] for row in r['rows'].values()),
        improved=sum(row[a][t]['iou'] > row[b][t]['iou'] + 1e-8 for row in r['rows'].values()),
        harmed=sum(row[a][t]['iou'] < row[b][t]['iou'] - 1e-8 for row in r['rows'].values()),
        per_task_delta_pp={n: 100*(row[a][t]['iou']-row[b][t]['iou']) for n, row in r['rows'].items()}) for a, b in pairs} for t in r['macro']}
    r['previous_candidate_effects'] = {t: dict(delta_pp=100*mean(row['constrained_memory_native'][t]['iou']-
        r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou'] for n,row in r['rows'].items()),
        improved=sum(row['constrained_memory_native'][t]['iou'] > r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou']+1e-8 for n,row in r['rows'].items()),
        harmed=sum(row['constrained_memory_native'][t]['iou'] < r['previous_candidate_rows'][n]['scribble_anchor_native'][t]['iou']-1e-8 for n,row in r['rows'].items())) for t in r['macro']}
    r['source_teacher_macro'] = {n: {a: mean(v['confidence_weighted_mean_huber'][a] for v in rows)
        for a in ['v10', 'static_native', 'legacy_memory_native', 'semantic_memory_native', 'constrained_memory_native']} for n, rows in r['source_teacher_checks'].items()}
    (PAPER / 'nvos_reference_completion_feedback.json').write_text(json.dumps(r, indent=2)+'\n')
    rows = [dict(task=n, arm=a, gt_track=t, **v) for n, row in r['rows'].items() for a, tracks in row.items() for t, v in tracks.items()]
    with (PAPER / 'artifacts/nvos_reference_completion.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Variant & IoU & AP & Precision & Recall & LUDVIG GT IoU \\\midrule']
    for a, label in zip(ARMS, LABELS):
        vals = [100*r['macro']['retained_gt_gt127'][a][m] for m in ['iou', 'average_precision', 'precision', 'recall']] + [100*r['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label + ' & ' + ' & '.join(f'{v:.2f}' for v in vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']; (PAPER / 'tables/nvos_reference_completion.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\section{Signed reference completion before instance transport}',
        'N12 compares the original reference mask, a frozen-DINO signed-affinity response decoded '
        'on the reference image, and the same completion with original positive/negative scribble '
        'points plus exact annotation constraints. All three use identical frozen two-frame SAM3 '
        'memory and native-ray 3D fitting. The primary constrained branch is fixed for all eight '
        'tasks; all 72 predictions precede either target GT convention. Reference annotation '
        'agreement is not complete-object ground truth. Original negative scribbles and target RGB '
        'are explicitly declared inputs. Other four benchmarks remain unchanged references.',
        r'\begin{center}\small\resizebox{\linewidth}{!}{\input{tables/nvos_reference_completion.tex}}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule', r'Task & Static & Legacy memory & Semantic memory & Constrained memory \\\midrule']
    for n, row in r['rows'].items():
        lines.append(n.replace('_', r'\_') + ' & ' + ' & '.join(f"{100*row[a]['retained_gt_gt127']['iou']:.2f}" for a in ['static_native', 'legacy_memory_native', 'semantic_memory_native', 'constrained_memory_native']) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    for name, v in r['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_', ' ') + f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    if r['spatial'] is not None:
        lines += [r'\begin{figure}[ht]\centering\includegraphics[width=.85\linewidth,height=.75\textheight,keepaspectratio]{../../output/tmm_response_20261018/study/nvos_reference_completion_spatial_v1/errors.png}',
            r'\caption{All eight N12 tasks. Green TP, red FP, blue FN; native-grid display resizing only. Scores use original GT resolution.}\end{figure}']
    (PAPER / 'sections/nvos_reference_completion.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=r['macro'], effects=r['effects']), indent=2))


if __name__ == '__main__': main()
