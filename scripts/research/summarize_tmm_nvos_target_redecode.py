"""N16 matched target-decoder effects and fixed stopping decision."""
import csv
import json
from pathlib import Path
from statistics import mean
from radio_gs.response_study.reference_study import read, require
from radio_gs.response_study.target_redecode_study import check_target_redecode_study, ARMS

REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
LABELS=['v10','N14 image','N14 native','Full-frame image','Full-frame native','Local image','Local native (primary)']


def main():
    reg=read(REPO/'configs/research/tmm_response.json');item=next(i for i in reg['reference_studies'] if i['id']=='N16')
    r=check_target_redecode_study(REPO,REPO/item['study_root'],item,read(REPO/reg['recipe'])['jobs']);require(r['complete'],'N16 incomplete')
    pairs=[('full_native','quality_native'),('crop_native','quality_native'),('crop_native','full_native'),
           ('full_image','quality_image'),('crop_image','quality_image')]
    r['effects']={t:{a+'_vs_'+b:dict(
        delta_pp=100*mean(row[a][t]['iou']-row[b][t]['iou'] for row in r['rows'].values()),
        average_precision_delta_pp=100*mean(row[a][t]['average_precision']-row[b][t]['average_precision'] for row in r['rows'].values()),
        improved=sum(row[a][t]['iou']>row[b][t]['iou']+1e-8 for row in r['rows'].values()),
        harmed=sum(row[a][t]['iou']<row[b][t]['iou']-1e-8 for row in r['rows'].values()),
        per_task_delta_pp={n:100*(row[a][t]['iou']-row[b][t]['iou']) for n,row in r['rows'].items()})
        for a,b in pairs} for t in r['macro']}
    accepted=[a for a in ['full_native','crop_native'] if all(
        m[a]['iou']>m['quality_native']['iou'] and m[a]['average_precision']>=m['quality_native']['average_precision']
        for m in r['macro'].values())]
    r['stopping_decision']=dict(rule='mean IoU improves and mean AP does not fall on both GT tracks',
        qualifying_arms=accepted,stop_this_branch=not accepted,universal_upper_bound_established=False,
        independent_test=False,default_changed=False)
    clusters={n:('horns' if n.startswith('horns_') else n) for n in r['rows']}
    r['leave_one_scene_out_diagnostic']={t:{a:{omitted:100*mean(
        row[a][t]['iou']-row['quality_native'][t]['iou'] for n,row in r['rows'].items() if clusters[n]!=omitted)
        for omitted in sorted(set(clusters.values()))} for a in ['full_native','crop_native']} for t in r['macro']}
    r['source_residual_macro']={a:mean(mean(v['confidence_weighted_mean_huber'][a] for v in rows)
        for rows in r['source_teacher_checks'].values()) for a in ['v10','quality_native','full_native','crop_native']}
    r['crop_geometry']={}
    for n,d in r['decoder'].items():
        box=d['refinement']['crop']['box']
        r['crop_geometry'][n]=dict(box=box,full_frame=box==[0,0,1008,756],
            area_fraction=None if box is None else (box[2]-box[0])*(box[3]-box[1])/(1008*756))
    (PAPER/'nvos_target_redecode_feedback.json').write_text(json.dumps(r,indent=2)+'\n')
    rows=[dict(task=n,arm=a,gt_track=t,**v) for n,row in r['rows'].items() for a,tracks in row.items() for t,v in tracks.items()]
    with (PAPER/'artifacts/nvos_target_redecode.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Variant & IoU & AP & LUDVIG GT IoU \\\midrule']
    for a,label in zip(ARMS,LABELS):
        values=[100*r['macro']['retained_gt_gt127'][a][m] for m in ['iou','average_precision']]+[100*r['macro']['ludvig_gt_maximum'][a]['iou']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/nvos_target_redecode.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\section{Complete-query target re-decoding}',
        'N16 replays N14 reference-quality completion and transport. The full-frame control '
        'uses that complete target response as one additional frozen SAM3 dense-logit and box prompt. '
        'The local primary applies the same operation to the response extent with 25 percent '
        'context on each side. RGB and prior share the same crop, and all outside probabilities '
        'are preserved before identical native 3D fitting. No extra annotations, candidate selection, '
        'recursive decoding or crop-size search is used. All 56 predictions precede both GT tracks. '
        'This is development on the opened eight-task cohort, with the other four benchmarks unchanged.',
        r'\begin{center}\small\input{tables/nvos_target_redecode.tex}\end{center}',
        r'\begin{center}\small\begin{tabular}{lrrr}\toprule',r'Task & N14 native & Full native & Local native \\\midrule']
    for n,row in r['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*row[a]['retained_gt_gt127']['iou']:.2f}" for a in ['quality_native','full_native','crop_native'])+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    full_names=[n.replace('_',r'\_') for n,v in r['crop_geometry'].items() if v['full_frame']]
    if full_names:
        lines.append('The fixed window covers the full image on '+', '.join(full_names)+
            '; local and full-frame image predictions there have identical hashes. These tasks do not test an image-scale change.')
    for name,v in r['effects']['retained_gt_gt127'].items():
        lines.append(name.replace('_',' ')+f": {v['delta_pp']:+.4f} IoU points, {v['average_precision_delta_pp']:+.4f} AP points; {v['improved']} improved and {v['harmed']} harmed tasks.")
    lines.append('The fixed stopping criterion requires higher mean IoU without lower AP on both GT tracks. '+
        ('No arm qualifies; this target-refinement branch is stopped and N14 is retained. This does not establish a universal performance ceiling.'
         if not accepted else 'At least one fixed native arm qualifies as a development candidate; the local primary and full-frame control retain their original roles.'))
    if r['spatial'] is not None:
        lines += [r'\begin{figure}[ht]\centering\includegraphics[width=.85\linewidth,height=.75\textheight,keepaspectratio]{../../output/tmm_response_20261018/study/nvos_target_redecode_spatial_v1/errors.png}',
            r'\caption{All eight N16 tasks: green TP, red FP, blue FN. Display resizing only; metrics use original GT resolution.}\end{figure}']
    (PAPER/'sections/nvos_target_redecode.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=r['macro'],effects=r['effects'],stopping_decision=r['stopping_decision']),indent=2))


if __name__=='__main__':main()
