"""N07 all-task response-permission effects; never substitute AP for IoU."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.image_response_study import check_image_response_study,MEMORY_ARMS
from radio_gs.response_study.reference_study import read,sha,require,check_binary_metrics

REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
LABELS=['v10','Static image','Pair memory','Sequence memory','Static image + 3D','Pair memory + 3D','Sequence memory + 3D']


def main():
    registry=read(REPO/'configs/research/tmm_response.json');study=REPO/registry['study_root']
    item=next(r for r in registry['reference_studies'] if r['id']=='N07')
    record=check_image_response_study(REPO,study,item,read(REPO/registry['recipe'])['jobs'])
    require(record['complete'],'N07 incomplete')
    pairs=[('sequence_memory_lift','v10'),('sequence_memory_lift','static_image_lift'),
           ('sequence_memory_lift','pair_memory_lift'),('pair_memory','static_image'),
           ('sequence_memory','pair_memory'),('sequence_memory_lift','sequence_memory')]
    record['effects']={t:{a+'_vs_'+b:dict(delta_pp=100*mean(r[a][t]['iou']-r[b][t]['iou'] for r in record['rows'].values()),
        improved=sum(r[a][t]['iou']>r[b][t]['iou']+1e-8 for r in record['rows'].values()),
        harmed=sum(r[a][t]['iou']<r[b][t]['iou']-1e-8 for r in record['rows'].values()),
        per_task_delta_pp={n:100*(r[a][t]['iou']-r[b][t]['iou']) for n,r in record['rows'].items()}) for a,b in pairs} for t in record['macro']}
    source={n:{a:mean(r['confidence_weighted_mean_huber'][a] for r in rows)
               for a in ('v10','static_image_lift','pair_memory_lift','sequence_memory_lift')} for n,rows in record['source_teacher_checks'].items()}
    record['source_teacher_macro']=source
    audit=study/'nvos_memory_spatial_v1'
    if (audit/'summary.json').exists():
        s,b=read(audit/'summary.json'),read(audit/'prediction_binding.json')
        require(s['complete'] and s['diagnostic_only'] and b['parent_summary_sha256']==sha(study/item['run']/'summary.json')
                and b['parent_seal_sha256']==sha(study/item['run']/'predictions_complete.json')
                and s['prediction_binding_sha256']==sha(audit/'prediction_binding.json')
                and s['figure_sha256']==sha(audit/'errors.png'),'N07 spatial binding changed')
        for n,r in s['rows'].items():
            for a,v in r['metrics'].items():
                check_binary_metrics(v);require(all(v[k]==record['rows'][n][a]['retained_gt_gt127'][k] for k in v),'N07 spatial counts changed')
        record['spatial_diagnosis']=s['rows'];record['receipts'].update({str(p):sha(p) for p in audit.iterdir() if p.is_file()})
    previous_path=study/'nvos_image_response_v3/summary.json'
    previous=read(previous_path)
    control={n:{t:abs(r['static_image'][t]['iou']-previous['rows'][n]['rgb_mask'][t]['iou'])
                for t in record['macro']} for n,r in record['rows'].items()}
    require(max(d for row in control.values() for d in row.values())<1e-12, 'N07 static direct control changed')
    record['static_control_iou_differences']=control
    record['receipts'][str(previous_path)]=sha(previous_path)
    old_seal=study/'nvos_image_response_v3/predictions_complete.json'
    old_seals=read(old_seal)['predictions'];new_seals=read(study/item['run']/'predictions_complete.json')['predictions']
    record['static_control_array_exact']={n:new_seals[n]['static_image']['sha256']==old_seals[n]['rgb_mask']['sha256'] for n in record['rows']}
    record['receipts'][str(old_seal)]=sha(old_seal)
    (PAPER/'nvos_memory_response_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    flat=[dict(task=n,arm=a,gt_track=t,**v) for n,r in record['rows'].items() for a,tracks in r.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_memory_response.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Arm & IoU & Precision & Recall & Accuracy & LUDVIG GT IoU \\\midrule']
    for a,label in zip(MEMORY_ARMS,LABELS):
        vals=[100*record['macro']['retained_gt_gt127'][a][m] for m in ('iou','precision','recall','accuracy')]
        vals.append(100*record['macro']['ludvig_gt_maximum'][a]['iou'])
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(PAPER/'tables/nvos_memory_response.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Reference memory response transport}',
        'N07 carries the original positive-only reference mask through a frozen SAM3 tracker. '
        'Two-frame memory uses the reference and observed target image; sequence memory also '
        'uses the ordered source views between them. The primary arm is sequence memory with '
        'actual 3D writeback. The source mask is fallible, and temporal memory is an existing '
        'external teacher capability, not a claimed new tracker.',
        r'\begin{center}\small\input{tables/nvos_memory_response.tex}\end{center}',
        'The static N04 image response is freshly replayed. Each arm uses the same image '
        'checkpoint, image preprocessing, original Gaussian geometry, 120 Huber updates, '
        '512-by-384 lifting grid and 0.5 threshold. All 56 predictions precede both GT '
        'scorings. AP is not measured. No task-wise switching is allowed. These are development '
        'results with observed target RGB; the updated fields are view-conditioned, not query-global.']
    for key,label in [('sequence_memory_lift_vs_v10','v10'),('sequence_memory_lift_vs_static_image_lift','the static control'),('sequence_memory_lift_vs_pair_memory_lift','two-frame memory')]:
        e=record['effects']['retained_gt_gt127'][key]
        lines.append(f"The primary candidate changes mean IoU by {e['delta_pp']:+.4f} points relative to {label} "
                     f"({e['improved']} improved and {e['harmed']} harmed tasks).")
    lines += [r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Task & v10 & Static + 3D & Pair + 3D & Sequence + 3D \\\midrule']
    for n,r in record['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*r[a]['retained_gt_gt127']['iou']:.2f}" for a in ('v10','static_image_lift','pair_memory_lift','sequence_memory_lift'))+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        r'\IfFileExists{../../output/tmm_response_20261016/study/nvos_memory_spatial_v1/errors.png}{',
        r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{../../output/tmm_response_20261016/study/nvos_memory_spatial_v1/errors.png}',
        r'\caption{All eight N07 tasks. Green: true positive; red: false positive; blue: false negative. Original reference identity is transferred with the same rule across tasks.}\end{figure}}{}']
    (PAPER/'sections/nvos_memory_response.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=record['effects'],source=source),indent=2))


if __name__=='__main__':main()
