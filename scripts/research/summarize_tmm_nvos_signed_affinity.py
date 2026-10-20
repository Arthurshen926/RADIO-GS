"""N09 signed reference transport: all-arm, all-task mechanism feedback."""
import csv,json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read,require
from radio_gs.response_study.signed_affinity_study import check_signed_affinity_study,ARMS
REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
LABELS=['v10','Static image','Static native','Signed affinity','Pseudo affinity','Signed + SAM','Pseudo + SAM','Signed native (primary)','Pseudo native']


def main():
    reg=read(REPO/'configs/research/tmm_response.json');item=next(i for i in reg['reference_studies'] if i['id']=='N09')
    record=check_signed_affinity_study(REPO,REPO/reg['study_root'],item,read(REPO/reg['recipe'])['jobs']);require(record['complete'],'N09 incomplete')
    comparisons=[('signed_native','static_native'),('signed_native','pseudo_native'),('signed_affinity','pseudo_affinity'),
                 ('signed_image','signed_affinity'),('pseudo_image','pseudo_affinity'),('signed_native','signed_image'),('pseudo_native','pseudo_image')]
    record['effects']={t:{a+'_vs_'+b:dict(delta_pp=100*mean(r[a][t]['iou']-r[b][t]['iou'] for r in record['rows'].values()),
        average_precision_delta_pp=100*mean(r[a][t]['average_precision']-r[b][t]['average_precision'] for r in record['rows'].values()),
        improved=sum(r[a][t]['iou']>r[b][t]['iou']+1e-8 for r in record['rows'].values()),
        harmed=sum(r[a][t]['iou']<r[b][t]['iou']-1e-8 for r in record['rows'].values()),
        per_task_delta_pp={n:100*(r[a][t]['iou']-r[b][t]['iou']) for n,r in record['rows'].items()}) for a,b in comparisons} for t in record['macro']}
    record['source_teacher_macro']={n:{a:mean(v['confidence_weighted_mean_huber'][a] for v in rows) for a in ['v10','static_native','signed_native','pseudo_native']} for n,rows in record['source_teacher_checks'].items()}
    (PAPER/'nvos_signed_affinity_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[dict(task=n,arm=a,gt_track=t,**v) for n,r in record['rows'].items() for a,tracks in r.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_signed_affinity.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Variant & IoU & AP & Precision & Recall & LUDVIG GT IoU \\\midrule']
    for a,label in zip(ARMS,LABELS):
        vals=[100*record['macro']['retained_gt_gt127'][a][m] for m in ['iou','average_precision','precision','recall']]+[100*record['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(PAPER/'tables/nvos_signed_affinity.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Signed reference evidence without an extent gate}',
        'N09 uses frozen DINOv2 ViT-B/14 descriptors on the original reference and target RGB. '
        'The primary branch gives authority only to original positive and negative scribbles; '
        'unlabelled reference pixels are unknown. The matched pseudo-mask control adds the '
        'retained reference SAM mask/complement as labels and applies the same scribble overrides. '
        'Both branches use identical teachers, query inputs and downstream operations. This '
        'explicitly extends the positive-only v10 input contract, without feature-field training. '
        'DINOv2 and SAM are external teachers, not new contributions of this paper.',
        r'For normalized reference descriptors $z_j$, define '
        r'$m_i=\max_{j\in P}z_i^\top z_j-\max_{j\in N}z_i^\top z_j$ and '
        r'$p_i=\sigma(m_i/\tau)$. The scale $\tau$ is the median signed leave-one-out '
        r'reference margin divided by $\log 9$. Mixed-sign patches are unknown. '
        'Missing support or a nonpositive median disables the edit and uses the static response. '
        'The frozen SAM decoder receives the transferred probability and its own extent box. '
        'Native-ray 3D writeback uses the same 120-step solver and observed domain as N08. '
        'No target-selected threshold, per-task winner or new user annotation is used.',
        'All 72 responses across eight tasks precede GT. Both GT conventions and AP are scored. '
        'Other benchmarks are unchanged references; outputs remain view-conditioned development evidence.',
        r'\begin{center}\small\resizebox{\linewidth}{!}{\input{tables/nvos_signed_affinity.tex}}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Task & Static native & Signed native & Pseudo native & Signed direct affinity \\\midrule']
    for n,r in record['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*r[a]['retained_gt_gt127']['iou']:.2f}" for a in ['static_native','signed_native','pseudo_native','signed_affinity'])+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    for name,v in record['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_',' ')+f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    lines += [r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{../../output/tmm_response_20261016/study/nvos_signed_affinity_v1/errors.png}',
              r'\caption{N09 full cohort: green TP, red FP, blue FN.}\end{figure}']
    (PAPER/'sections/nvos_signed_affinity.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=record['effects']),indent=2))


if __name__=='__main__':main()
