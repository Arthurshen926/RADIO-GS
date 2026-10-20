"""N08 complete-cohort resolution transport effects on fixed N07 evidence."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.native_writeback_study import check_native_writeback_study,ARMS,PROVIDERS
from radio_gs.response_study.reference_study import read,sha,require,check_binary_metrics
REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
LABELS=['v10','Static, coarse','Pair, coarse','Sequence, coarse','Static, native','Pair, native','Sequence, native']


def main():
    registry=read(REPO/'configs/research/tmm_response.json');study=REPO/registry['study_root'];item=next(r for r in registry['reference_studies'] if r['id']=='N08')
    record=check_native_writeback_study(REPO,study,item,read(REPO/registry['recipe'])['jobs']);require(record['complete'],'N08 incomplete')
    pairs=[(p+'_native',p+'_lift') for p in PROVIDERS]+[('sequence_memory_native','v10'),('sequence_memory_native','static_image_native'),('sequence_memory_native','pair_memory_native')]
    record['effects']={t:{a+'_vs_'+b:dict(delta_pp=100*mean(r[a][t]['iou']-r[b][t]['iou'] for r in record['rows'].values()),
        improved=sum(r[a][t]['iou']>r[b][t]['iou']+1e-8 for r in record['rows'].values()),
        harmed=sum(r[a][t]['iou']<r[b][t]['iou']-1e-8 for r in record['rows'].values()),
        per_task_delta_pp={n:100*(r[a][t]['iou']-r[b][t]['iou']) for n,r in record['rows'].items()}) for a,b in pairs} for t in record['macro']}
    record['source_teacher_macro']={n:{a:mean(r['confidence_weighted_mean_huber'][a] for r in rows) for a in ('v10',*[p+'_native' for p in PROVIDERS])} for n,rows in record['source_teacher_checks'].items()}
    audit=study/'nvos_native_spatial_v2';s,b=read(audit/'summary.json'),read(audit/'prediction_binding.json')
    require(s['complete'] and s['diagnostic_only'] and b['parent_summary_sha256']==sha(study/item['run']/'summary.json')
            and b['parent_seal_sha256']==sha(study/item['run']/'predictions_complete.json')
            and s['prediction_binding_sha256']==sha(audit/'prediction_binding.json') and s['figure_sha256']==sha(audit/'errors.png'),'N08 spatial binding changed')
    require(set(s['rows'])==set(record['rows']),'N08 spatial cohort incomplete')
    for n,r in s['rows'].items():
        for a,v in r['metrics'].items():check_binary_metrics(v);require(all(v[k]==record['rows'][n][a]['retained_gt_gt127'][k] for k in v),'N08 spatial counts changed')
    record['spatial_diagnosis']=s['rows'];record['receipts'].update({str(p):sha(p) for p in audit.iterdir() if p.is_file()})
    diagnosis=study/'nvos_native_writeback_v2/operator_diagnosis.json'
    record['operator_correction']=read(diagnosis)
    require(record['operator_correction']['complete'] and not record['operator_correction']['target_gt_read']
            and record['operator_correction']['cuda_max_alpha_error']<1e-4, 'N08 correction audit changed')
    record['receipts'][str(diagnosis)]=sha(diagnosis)
    replay=study/'nvos_memory_response_v2/replay_check.json'
    record['parent_replay']=read(replay)
    require(record['parent_replay']['replay_summary_sha256']==read(study/item['run']/'summary.json')['parent_summary_sha256'], 'N08 parent replay changed')
    record['receipts'][str(replay)]=sha(replay)
    control=study/'nvos_coarse_backend_v1';c=read(control/'summary.json')
    require(c['complete'] and c['diagnostic_only'] and not c['gt_selection']
            and c['prediction_seal_sha256']==sha(control/'predictions_complete.json')
            and c['parent_summary_sha256']==read(study/item['run']/'summary.json')['parent_summary_sha256'], 'N08 backend control changed')
    require(set(c['rows'])==set(record['rows']),'N08 backend control incomplete')
    parent=read(study/read(study/item['run']/'manifest.json')['config']['parent_run']/'summary.json')
    for name,arms in c['rows'].items():
        require(set(arms)==set(PROVIDERS),'N08 backend control missing provider')
        for provider,value in arms.items():
            require(set(value['metrics'])==set(record['macro']),'N08 backend control missing GT track')
            for track,metrics in value['metrics'].items():
                check_binary_metrics(metrics)
                before=parent['rows'][name][provider+'_lift'][track]['iou']
                require(metrics['original_iou']==before and abs(metrics['delta_pp']-100*(metrics['iou']-before))<1e-12,
                        'N08 backend control comparison changed')
    maximum=max(abs(mean(r[p]['metrics'][t]['delta_pp'] for r in c['rows'].values())) for p in PROVIDERS for t in record['macro'])
    require(abs(maximum-c['maximum_cohort_iou_delta_pp'])<1e-12 and c['all_scoring_masks_identical']==
            all(v['scoring_binary_changed']==0 for r in c['rows'].values() for v in r.values()),'N08 backend control aggregate changed')
    record['coarse_backend_control']=c
    record['receipts'].update({str(p):sha(p) for p in control.iterdir() if p.is_file()})
    from radio_gs.response_study.reference_study import check_exact_readout_capacity
    capacity=study/'nvos_readout_capacity_v1'
    record['readout_capacity']=check_exact_readout_capacity(study)
    record['receipts'].update({str(p):sha(p) for p in capacity.iterdir() if p.is_file()})
    (PAPER/'nvos_native_writeback_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    flat=[dict(task=n,arm=a,gt_track=t,**v) for n,r in record['rows'].items() for a,tracks in r.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_native_writeback.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Writeback & IoU & Precision & Recall & Accuracy & LUDVIG GT IoU \\\midrule']
    for a,label in zip(ARMS,LABELS):
        vals=[100*record['macro']['retained_gt_gt127'][a][m] for m in ('iou','precision','recall','accuracy')]+[100*record['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(PAPER/'tables/nvos_native_writeback.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Resolution-consistent response transport}',
        'N08 keeps every image probability from N07 fixed and changes the writeback ray grid '
        'from 512-by-384 to the native 1008-by-756 output grid. The Gaussian geometry and '
        'observed domain, initial v10 field, absolute-margin confidence, 120 Huber steps and '
        '0.5 threshold are unchanged. The three original coarse writebacks are exact frozen '
        'controls. All 56 responses precede target scoring; both GT conventions are reported. '
        'The primary arm is sequence-memory native writeback. This design was registered '
        'before any N07 scores. AP is not measured. Results remain view-conditioned development evidence.',
        r'\begin{center}\small\input{tables/nvos_native_writeback.tex}\end{center}']
    for provider in PROVIDERS:
        e=record['effects']['retained_gt_gt127'][provider+'_native_vs_'+provider+'_lift']
        lines.append(provider.replace('_',' ')+f": native writeback changes IoU by {e['delta_pp']:+.4f} points ({e['improved']} improved, {e['harmed']} harmed tasks).")
    lines += [r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Task & v10 & Static native & Pair native & Sequence native \\\midrule']
    for n,r in record['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*r[a]['retained_gt_gt127']['iou']:.2f}" for a in ('v10','static_image_native','pair_memory_native','sequence_memory_native'))+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{../../output/tmm_response_20261016/study/nvos_native_spatial_v2/errors.png}',
        r'\caption{N08 complete cohort: green = TP, red = FP, blue = FN. The sequence coarse column uses the exact N07 response.}\end{figure}']
    cap=read(capacity/'summary.json')
    lines += [r'\paragraph{Readout capacity, not a method result.}',
        'An exhaustive threshold curve refines the earlier A15 grid diagnosis. Equal scores are '
        'kept together. Even one GT-optimal threshold per task is a nondeployable upper bound '
        'on scalar-threshold decisions of unchanged v10 responses. It is not an upper bound '
        'on new spatially conditioned responses, and no oracle threshold is exported.',
        r'\begin{center}\small\begin{tabular}{lrrr}\toprule',
        r'Task & Fixed v10 IoU & GT-oracle IoU & AP \\\midrule']
    for n,r in cap['rows'].items():
        v=r['retained_gt_gt127']
        lines.append(n.replace('_',r'\_')+f" & {100*v['fixed_half']['iou']:.2f} & {100*v['oracle_iou']:.2f} & {100*v['average_precision']:.2f}"+r' \\')
    v=cap['macro']['retained_gt_gt127']
    lines += [f"Mean & {100*v['fixed_half_iou']:.2f} & {100*v['oracle_iou']:.2f} & {100*v['average_precision']:.2f}"+r' \\',r'\bottomrule\end{tabular}\end{center}']
    lines.append('The optional CUDA footprint backend corrects eager arithmetic cancellation '
                 'on thin conics; this is an implementation correction, not a new segmentation '
                 'mechanism. The matched coarse-grid backend control changes cohort IoU by at '
                 f"most {c['maximum_cohort_iou_delta_pp']:.7f} percentage points. "
                 'The full native run retains the original renderer-consistency tolerance.')
    (PAPER/'sections/nvos_native_writeback.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=record['effects']),indent=2))


if __name__=='__main__':main()
