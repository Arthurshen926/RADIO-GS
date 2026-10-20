"""N10: does SAM preserve fixed signed affinity evidence before actual 3D fit?"""
import csv,json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read,require
from radio_gs.response_study.direct_affinity_study import check_direct_affinity_study,ARMS
REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
LABELS=['v10','Static SAM native','Signed + SAM native','Pseudo + SAM native','Signed direct native (primary)','Pseudo direct native']


def main():
    reg=read(REPO/'configs/research/tmm_response.json');item=next(i for i in reg['reference_studies'] if i['id']=='N10')
    record=check_direct_affinity_study(REPO,REPO/reg['study_root'],item,read(REPO/reg['recipe'])['jobs']);require(record['complete'],'N10 incomplete')
    pairs=[('signed_direct_native','static_native'),('signed_direct_native','signed_native'),('pseudo_direct_native','pseudo_native'),('signed_direct_native','pseudo_direct_native')]
    record['effects']={t:{a+'_vs_'+b:dict(delta_pp=100*mean(r[a][t]['iou']-r[b][t]['iou'] for r in record['rows'].values()),
        average_precision_delta_pp=100*mean(r[a][t]['average_precision']-r[b][t]['average_precision'] for r in record['rows'].values()),
        improved=sum(r[a][t]['iou']>r[b][t]['iou']+1e-8 for r in record['rows'].values()),
        harmed=sum(r[a][t]['iou']<r[b][t]['iou']-1e-8 for r in record['rows'].values()),
        per_task_delta_pp={n:100*(r[a][t]['iou']-r[b][t]['iou']) for n,r in record['rows'].items()}) for a,b in pairs} for t in record['macro']}
    record['source_teacher_macro']={n:{a:mean(v['confidence_weighted_mean_huber'][a] for v in rows) for a in ['v10','signed_direct_native','pseudo_direct_native']} for n,rows in record['source_teacher_checks'].items()}
    (PAPER/'nvos_direct_affinity_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[dict(task=n,arm=a,gt_track=t,**v) for n,r in record['rows'].items() for a,tracks in r.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_direct_affinity.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Variant & IoU & AP & Precision & Recall & LUDVIG GT IoU \\\midrule']
    for a,label in zip(ARMS,LABELS):
        vals=[100*record['macro']['retained_gt_gt127'][a][m] for m in ['iou','average_precision','precision','recall']]+[100*record['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(PAPER/'tables/nvos_direct_affinity.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Direct affinity writeback without a second segmentation}',
        'N10 was fixed before N09 scoring. Exact signed and pseudo-mask affinity probabilities from N09 '
        'are fitted directly into the same observed Gaussians using the identical native-ray operator, '
        'confidence and 120-step objective. The four parent controls retain exact continuous-array hashes '
        'and scores. No teacher calls, new prompts, geometry or target-selected parameters are introduced. '
        'All 48 responses precede GT. Original negative reference scribbles and observed target RGB '
        'remain explicit input extensions; the result is view-conditioned development evidence.',
        r'\begin{center}\small\resizebox{\linewidth}{!}{\input{tables/nvos_direct_affinity.tex}}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Task & Static & Signed + SAM & Signed direct & Pseudo direct \\\midrule']
    for n,r in record['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*r[a]['retained_gt_gt127']['iou']:.2f}" for a in ['static_native','signed_native','signed_direct_native','pseudo_direct_native'])+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    for name,v in record['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_',' ')+f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    (PAPER/'sections/nvos_direct_affinity.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=record['effects']),indent=2))


if __name__=='__main__':main()
