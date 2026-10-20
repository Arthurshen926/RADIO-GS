"""N05 all-task response-permission effects; never substitute AP for IoU."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.image_response_study import check_image_response_study,PARTIAL_ARMS
from radio_gs.response_study.reference_study import read,sha,require,check_binary_metrics

REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
LABELS=['v10','Signed + points','Positive + points','Points only','Signed + points + 3D','Positive + points + 3D','Points only + 3D']


def main():
    registry=read(REPO/'configs/research/tmm_response.json');study=REPO/registry['study_root']
    item=next(r for r in registry['reference_studies'] if r['id']=='N05')
    record=check_image_response_study(REPO,study,item,read(REPO/registry['recipe'])['jobs'])
    require(record['complete'],'N05 incomplete')
    pairs=[('positive_core_lift','v10'),('positive_core_lift','signed_core_lift'),
           ('positive_core_lift','sparse_core_lift'),('positive_core','signed_core'),
           ('sparse_core','signed_core'),('positive_core_lift','positive_core')]
    record['effects']={t:{a+'_vs_'+b:dict(delta_pp=100*mean(r[a][t]['iou']-r[b][t]['iou'] for r in record['rows'].values()),
        improved=sum(r[a][t]['iou']>r[b][t]['iou']+1e-8 for r in record['rows'].values()),
        harmed=sum(r[a][t]['iou']<r[b][t]['iou']-1e-8 for r in record['rows'].values()),
        per_task_delta_pp={n:100*(r[a][t]['iou']-r[b][t]['iou']) for n,r in record['rows'].items()}) for a,b in pairs} for t in record['macro']}
    source={n:{a:mean(r['confidence_weighted_mean_huber'][a] for r in rows)
               for a in ('v10','signed_core_lift','positive_core_lift','sparse_core_lift')} for n,rows in record['source_teacher_checks'].items()}
    record['source_teacher_macro']=source
    audit=study/'nvos_partial_image_spatial_v1'
    if (audit/'summary.json').exists():
        s,b=read(audit/'summary.json'),read(audit/'prediction_binding.json')
        require(s['complete'] and s['diagnostic_only'] and b['parent_summary_sha256']==sha(study/item['run']/'summary.json')
                and b['parent_seal_sha256']==sha(study/item['run']/'predictions_complete.json')
                and s['prediction_binding_sha256']==sha(audit/'prediction_binding.json')
                and s['figure_sha256']==sha(audit/'errors.png'),'N05 spatial binding changed')
        for n,r in s['rows'].items():
            for a,v in r['metrics'].items():
                check_binary_metrics(v);require(all(v[k]==record['rows'][n][a]['retained_gt_gt127'][k] for k in v),'N05 spatial counts changed')
        record['spatial_diagnosis']=s['rows'];record['receipts'].update({str(p):sha(p) for p in audit.iterdir() if p.is_file()})
    (PAPER/'nvos_partial_response_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    flat=[dict(task=n,arm=a,gt_track=t,**v) for n,r in record['rows'].items() for a,tracks in r.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_partial_response.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Arm & IoU & Precision & Recall & Accuracy & LUDVIG GT IoU \\\midrule']
    for a,label in zip(PARTIAL_ARMS,LABELS):
        vals=[100*record['macro']['retained_gt_gt127'][a][m] for m in ('iou','precision','recall','accuracy')]
        vals.append(100*record['macro']['ludvig_gt_maximum'][a]['iou'])
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(PAPER/'tables/nvos_partial_response.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Positive identity and uncertain exclusion in image prompts}',
        'N05 keeps the same response-derived box and eight deterministic inferred positive points '
        'for three image prompts: signed dense log-odds, positive-only dense log-odds (negative values '
        'become zero), and no dense prior. Point selection uses confidence, foreground interior distance '
        'and distance to previous points. These are estimated identity hints, not extra annotated clicks. '
        'The positive-only prompt with 3D writeback is the predeclared primary candidate. '
        'Neutral dense values are an operational SAM input, not a calibrated uncertainty estimate.',
        r'\begin{center}\small\input{tables/nvos_partial_response.tex}\end{center}',
        'All eight tasks use observed target RGB and the same frozen SAM3, original geometry, '
        '512-by-384 lifting grid, 120 Huber steps and 0.5 threshold. Every prediction precedes scoring. '
        'N05 measures IoU, precision, recall and pixel accuracy under both GT conventions; '
        'AP is not measured. Results do not establish query-global multi-view consistency.']
    for key,label in [('positive_core_lift_vs_v10','v10'),('positive_core_lift_vs_signed_core_lift','the signed dense control'),('positive_core_lift_vs_sparse_core_lift','the sparse-only control')]:
        e=record['effects']['retained_gt_gt127'][key]
        lines.append(f"The primary candidate changes mean IoU by {e['delta_pp']:+.4f} points relative to {label} "
                     f"({e['improved']} improved and {e['harmed']} harmed tasks).")
    lines += [r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Task & v10 & Signed + 3D & Positive + 3D & Sparse + 3D \\\midrule']
    for n,r in record['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*r[a]['retained_gt_gt127']['iou']:.2f}" for a in ('v10','signed_core_lift','positive_core_lift','sparse_core_lift'))+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        r'\IfFileExists{../../output/tmm_response_20261016/study/nvos_partial_image_spatial_v1/errors.png}{',
        r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{../../output/tmm_response_20261016/study/nvos_partial_image_spatial_v1/errors.png}',
        r'\caption{All eight N05 tasks. Green: true positive; red: false positive; blue: false negative. No task-specific prompt switch is used.}\end{figure}}{}']
    (PAPER/'sections/nvos_partial_response.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=record['effects'],source=source),indent=2))


if __name__=='__main__':main()
