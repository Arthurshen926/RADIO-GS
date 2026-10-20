"""N15 full-cohort comparison and explicit predeclared stopping rule."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read, require, sha
from radio_gs.response_study.multiview_completion_study import check_multiview_completion_study, ARMS

REPO = Path(__file__).resolve().parents[2]; PAPER = REPO / 'paper/tmm_response'
LABELS = ['v10', 'N14 image', 'N14 native', 'Original source fusion',
          'Completed source fusion', 'Completed joint fusion (primary)']


def main():
    reg = read(REPO / 'configs/research/tmm_response.json')
    item = next(i for i in reg['reference_studies'] if i['id'] == 'N15')
    r = check_multiview_completion_study(REPO, REPO/item['study_root'], item, read(REPO/reg['recipe'])['jobs'])
    require(r['complete'], 'N15 incomplete')
    pairs = [('completed_source_native', 'legacy_source_native'),
             ('completed_joint_native', 'completed_source_native'),
             ('completed_joint_native', 'quality_native'), ('completed_source_native', 'quality_native')]
    r['effects'] = {t: {a+'_vs_'+b: dict(
        delta_pp=100*mean(row[a][t]['iou']-row[b][t]['iou'] for row in r['rows'].values()),
        average_precision_delta_pp=100*mean(row[a][t]['average_precision']-row[b][t]['average_precision'] for row in r['rows'].values()),
        improved=sum(row[a][t]['iou'] > row[b][t]['iou']+1e-8 for row in r['rows'].values()),
        harmed=sum(row[a][t]['iou'] < row[b][t]['iou']-1e-8 for row in r['rows'].values()),
        per_task_delta_pp={n: 100*(row[a][t]['iou']-row[b][t]['iou']) for n,row in r['rows'].items()})
        for a,b in pairs} for t in r['macro']}
    def improves(a):
        return all(r['macro'][t][a]['iou'] > r['macro'][t]['quality_native']['iou'] and
                   r['macro'][t][a]['average_precision'] >= r['macro'][t]['quality_native']['average_precision']
                   for t in r['macro'])
    accepted = [a for a in ARMS[3:] if improves(a)]
    r['stopping_decision'] = dict(rule='mean IoU improves and mean AP does not fall on both GT tracks',
        qualifying_arms=accepted, stop_this_branch=not accepted, universal_upper_bound_established=False,
        independent_test=False, default_changed=False)
    # Horns queries share one scene; leave a whole scene out, never one query
    # from that correlated pair. These are descriptive sensitivity statistics.
    clusters = {n: ('horns' if n.startswith('horns_') else n) for n in r['rows']}
    r['leave_one_scene_out_diagnostic'] = {t: {a: {
        omitted: 100*mean(row[a][t]['iou']-row['quality_native'][t]['iou']
                          for n,row in r['rows'].items() if clusters[n] != omitted)
        for omitted in sorted(set(clusters.values()))} for a in ARMS[3:]} for t in r['macro']}
    r['source_consistency_macro'] = {n: {branch: {a: mean(x[branch][a] for x in views)
        for a in views[0][branch]} for branch in ['legacy','completed']} for n,views in r['source_consistency'].items()}
    r['source_consistency_cohort'] = {branch: {a: mean(x[branch][a] for x in r['source_consistency_macro'].values())
        for a in next(iter(r['source_consistency_macro'].values()))[branch]} for branch in ['legacy','completed']}
    r['matched_source_completion_leave_one_scene_out'] = {t: {
        omitted: 100*mean(row['completed_source_native'][t]['iou']-row['legacy_source_native'][t]['iou']
                          for n,row in r['rows'].items() if clusters[n] != omitted)
        for omitted in sorted(set(clusters.values()))} for t in r['macro']}
    historical_path = PAPER/'nvos_coverage_feedback.json'
    historical = read(historical_path)
    r['historical_source_only_context'] = dict(reference='N01 coverage_high',
        source_sha256=sha(historical_path), matched_mechanism_control=False,
        added_inputs='official negative scribbles and DINOv2; no target image evidence',
        delta_iou_pp={t: 100*(r['macro'][t]['completed_source_native']['iou']-
                             historical['macro'][t]['coverage_high']) for t in r['macro']})
    r['receipts'][str(historical_path)] = sha(historical_path)
    (PAPER/'nvos_multiview_completion_feedback.json').write_text(json.dumps(r,indent=2)+'\n')
    rows = [dict(task=n,arm=a,gt_track=t,**v) for n,row in r['rows'].items() for a,tracks in row.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_multiview_completion.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Variant & IoU & AP & LUDVIG GT IoU \\\midrule']
    for a,label in zip(ARMS,LABELS):
        v=[100*r['macro']['retained_gt_gt127'][a][m] for m in ['iou','average_precision']]+[100*r['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label+' & '+' & '.join(f'{x:.2f}' for x in v)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/nvos_multiview_completion.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Completed-reference multiview fusion}',
        'N15 freezes N14 reference-quality selection and independently transports original and completed '
        'reference masks to every authorized source view with the same two-frame memory. Source-only '
        'fusion excludes target image evidence; the joint arm balances the average normalized source '
        'objective with one native target objective. All three columns use identical geometry, initial '
        'responses, observed elements and 120 Huber iterations. Source evidence is integrated at '
        '$512\\times384$, with native $1008\\times756$ target rays and rendering. There is no '
        'view-subset or fusion-weight search. All 48 predictions precede both GT scoring conventions. '
        'The original eight tasks are development data. Other four benchmarks remain unchanged references.',
        r'\begin{center}\small\resizebox{\linewidth}{!}{\input{tables/nvos_multiview_completion.tex}}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',
        r'Task & N14 & Original source & Complete source & Complete joint \\\midrule']
    for n,row in r['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*row[a]['retained_gt_gt127']['iou']:.2f}" for a in ARMS[2:])+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    for name,v in r['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_',' ')+f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    sensitivity=r['matched_source_completion_leave_one_scene_out']['retained_gt_gt127']
    lines.append('The matched source-only comparison isolates reference completion under the same '
        'transport and fusion, with zero target-image evidence in either field. Its gain remains '
        f"{min(sensitivity.values()):+.4f} to {max(sensitivity.values()):+.4f} points when leaving out one whole scene. "
        'This is descriptive cohort sensitivity, not independent validation. The historical N01 '
        'comparison additionally changes negative-scribble and DINOv2 access and is not a matched control.')
    losses=r['source_consistency_cohort']['completed']
    macro=r['macro']['retained_gt_gt127']
    lines.append('Against N14, joint fusion reduces mean completed-source Huber residual from '
        f"{losses['quality_native']:.6f} to {losses['completed_joint_native']:.6f}. "
        f"Mean precision changes by {100*(macro['completed_joint_native']['precision']-macro['quality_native']['precision']):+.4f} "
        f"points and recall by {100*(macro['completed_joint_native']['recall']-macro['quality_native']['recall']):+.4f}. "
        'Teacher agreement therefore does not by itself certify target accuracy. These teacher '
        'residuals are not source-view ground-truth scores.')
    lines.append('The predeclared stopping criterion requires higher mean IoU without lower AP. '+
        ('No arm qualifies; this fusion branch is stopped and N14 is retained. This is a local negative result, not a universal performance bound.'
         if not accepted else 'At least one fixed arm qualifies; cross-view evidence remains a development candidate, without independent SOTA certification.'))
    if r['spatial'] is not None:
        lines += [r'\begin{figure}[ht]\centering\includegraphics[width=.85\linewidth,height=.75\textheight,keepaspectratio]{../../output/tmm_response_20261018/study/nvos_multiview_completion_spatial_v1/errors.png}',
                  r'\caption{All eight N15 tasks: green TP, red FP, blue FN. Display resizing only; metrics use original GT resolution.}\end{figure}']
    (PAPER/'sections/nvos_multiview_completion.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=r['macro'],effects=r['effects'],stopping_decision=r['stopping_decision']),indent=2))


if __name__ == '__main__': main()
