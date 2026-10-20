"""N06 all-task response-permission effects; never substitute AP for IoU."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.image_response_study import check_image_response_study,UNBOXED_ARMS
from radio_gs.response_study.reference_study import read,sha,require,check_binary_metrics

REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
LABELS=['v10','Box + points','Points, no box','Signed + points, no box','Box + points + 3D','Points, no box + 3D','Signed, no box + 3D']


def main():
    registry=read(REPO/'configs/research/tmm_response.json');study=REPO/registry['study_root']
    item=next(r for r in registry['reference_studies'] if r['id']=='N06')
    record=check_image_response_study(REPO,study,item,read(REPO/registry['recipe'])['jobs'])
    require(record['complete'],'N06 incomplete')
    pairs=[('unboxed_sparse_lift','v10'),('unboxed_sparse_lift','boxed_sparse_lift'),
           ('unboxed_sparse_lift','unboxed_signed_lift'),('unboxed_sparse','boxed_sparse'),
           ('unboxed_signed','boxed_sparse'),('unboxed_sparse_lift','unboxed_sparse')]
    record['effects']={t:{a+'_vs_'+b:dict(delta_pp=100*mean(r[a][t]['iou']-r[b][t]['iou'] for r in record['rows'].values()),
        improved=sum(r[a][t]['iou']>r[b][t]['iou']+1e-8 for r in record['rows'].values()),
        harmed=sum(r[a][t]['iou']<r[b][t]['iou']-1e-8 for r in record['rows'].values()),
        per_task_delta_pp={n:100*(r[a][t]['iou']-r[b][t]['iou']) for n,r in record['rows'].items()}) for a,b in pairs} for t in record['macro']}
    source={n:{a:mean(r['confidence_weighted_mean_huber'][a] for r in rows)
               for a in ('v10','boxed_sparse_lift','unboxed_sparse_lift','unboxed_signed_lift')} for n,rows in record['source_teacher_checks'].items()}
    record['source_teacher_macro']=source
    audit=study/'nvos_unboxed_image_spatial_v1'
    if (audit/'summary.json').exists():
        s,b=read(audit/'summary.json'),read(audit/'prediction_binding.json')
        require(s['complete'] and s['diagnostic_only'] and b['parent_summary_sha256']==sha(study/item['run']/'summary.json')
                and b['parent_seal_sha256']==sha(study/item['run']/'predictions_complete.json')
                and s['prediction_binding_sha256']==sha(audit/'prediction_binding.json')
                and s['figure_sha256']==sha(audit/'errors.png'),'N06 spatial binding changed')
        for n,r in s['rows'].items():
            for a,v in r['metrics'].items():
                check_binary_metrics(v);require(all(v[k]==record['rows'][n][a]['retained_gt_gt127'][k] for k in v),'N06 spatial counts changed')
        record['spatial_diagnosis']=s['rows'];record['receipts'].update({str(p):sha(p) for p in audit.iterdir() if p.is_file()})
    previous=read(study/'nvos_partial_image_response_v1/summary.json')
    control={n:{t:abs(r['boxed_sparse'][t]['iou']-previous['rows'][n]['sparse_core'][t]['iou'])
                for t in record['macro']} for n,r in record['rows'].items()}
    require(max(d for row in control.values() for d in row.values())<1e-12, 'N06 boxed direct control changed')
    record['boxed_control_iou_differences']=control
    record['receipts'][str(study/'nvos_partial_image_response_v1/summary.json')]=sha(study/'nvos_partial_image_response_v1/summary.json')
    audit_path=study/item['run']/'source_prompt_contract.json'
    audit=read(audit_path)
    require(audit['complete'] and audit['jobs']==8 and not audit['target_gt_read'] and audit['unchanged_inference']
            and set(audit['rows'])==set(record['rows'])
            and hashlib.sha256(audit['source_text'].encode()).hexdigest()==audit['source_sha256'], 'source prompt audit changed')
    for p,h in audit['input_hashes'].items():
        require(sha(p)==h, 'source prompt input changed');record['receipts'][p]=h
    record['source_prompt_contract']=audit['rows']
    record['receipts'][str(audit_path)]=sha(audit_path)
    (PAPER/'nvos_unboxed_response_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    flat=[dict(task=n,arm=a,gt_track=t,**v) for n,r in record['rows'].items() for a,tracks in r.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_unboxed_response.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Arm & IoU & Precision & Recall & Accuracy & LUDVIG GT IoU \\\midrule']
    for a,label in zip(UNBOXED_ARMS,LABELS):
        vals=[100*record['macro']['retained_gt_gt127'][a][m] for m in ('iou','precision','recall','accuracy')]
        vals.append(100*record['macro']['ludvig_gt_maximum'][a]['iou'])
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(PAPER/'tables/nvos_unboxed_response.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Query identity without inherited extent}',
        'N06 keeps identical response-derived positive points and compares a box prompt, '
        'unboxed points, and unboxed points with signed dense log-odds. The primary candidate '
        'is unboxed sparse guidance with actual 3D writeback. It tests whether the old extent '
        'constrains completion. No new annotated clicks or target-dependent switch is used.',
        r'\begin{center}\small\input{tables/nvos_unboxed_response.tex}\end{center}',
        'All eight tasks use the same observed RGB, frozen SAM3, original Gaussian geometry, '
        '120 Huber steps, 512-by-384 lifting grid and 0.5 threshold. All 56 predictions precede '
        'scoring. Both GT conventions are reported; AP is not measured. This is a development '
        'comparison and view-conditioned update, not a query-global field.']
    for key,label in [('unboxed_sparse_lift_vs_v10','v10'),('unboxed_sparse_lift_vs_boxed_sparse_lift','the boxed sparse control'),('unboxed_sparse_lift_vs_unboxed_signed_lift','the unboxed signed control')]:
        e=record['effects']['retained_gt_gt127'][key]
        lines.append(f"The primary candidate changes mean IoU by {e['delta_pp']:+.4f} points relative to {label} "
                     f"({e['improved']} improved and {e['harmed']} harmed tasks).")
    lines += [r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Task & v10 & Box + 3D & No box + 3D & Signed, no box + 3D \\\midrule']
    for n,r in record['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*r[a]['retained_gt_gt127']['iou']:.2f}" for a in ('v10','boxed_sparse_lift','unboxed_sparse_lift','unboxed_signed_lift'))+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        r'\IfFileExists{../../output/tmm_response_20261016/study/nvos_unboxed_image_spatial_v1/errors.png}{',
        r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{../../output/tmm_response_20261016/study/nvos_unboxed_image_spatial_v1/errors.png}',
        r'\caption{All eight N06 tasks. Green: true positive; red: false positive; blue: false negative. No task-specific prompt switch is used.}\end{figure}}{}']
    (PAPER/'sections/nvos_unboxed_response.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=record['effects'],source=source),indent=2))


if __name__=='__main__':main()
