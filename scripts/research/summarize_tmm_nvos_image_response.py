"""Generate N04 tables and paired effects only after independent receipt checks."""
import csv
import json
from pathlib import Path
from statistics import mean

from radio_gs.response_study.image_response_study import check_image_response_study, ARMS
from radio_gs.response_study.reference_study import read, sha, require, check_binary_metrics

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper/tmm_response'
LABELS = ['v10', 'Render + box', 'RGB + box', 'Render + mask', 'RGB + mask', 'Render + mask + 3D', 'RGB + mask + 3D']


def main():
    registry = read(REPO/'configs/research/tmm_response.json')
    item = next(r for r in registry['reference_studies'] if r['id']=='N04')
    study = REPO/registry['study_root']
    record = check_image_response_study(REPO, study, item, read(REPO/registry['recipe'])['jobs'])
    require(record['complete'], 'N04 incomplete')
    import numpy as np
    pairs = [('rgb_mask_lift','v10'), ('rgb_mask_lift','render_mask_lift'),
             ('rgb_mask_lift','rgb_mask'), ('rgb_mask','render_mask'), ('rgb_mask','rgb_box')]
    effects = {}
    for track in record['macro']:
        effects[track] = {}
        for arm, base in pairs:
            delta = {n:100*(v[arm][track]['iou']-v[base][track]['iou']) for n,v in record['rows'].items()}
            groups = {n:'horns' if n.startswith('horns_') else n for n in delta}
            unique = sorted(set(groups.values()))
            samples = np.random.default_rng(0).choice(unique, size=(5000,7), replace=True)
            bootstrap = [mean(d for g in sample for n,d in delta.items() if groups[n]==g) for sample in samples]
            effects[track][arm+'_vs_'+base] = dict(delta_pp=mean(delta.values()), per_task_delta_pp=delta,
                improved=sum(d>1e-6 for d in delta.values()), harmed=sum(d< -1e-6 for d in delta.values()),
                descriptive_scene_bootstrap95_pp=np.quantile(bootstrap,[.025,.975]).tolist())
    record['effects'] = effects
    source = {n:{a:mean(v['confidence_weighted_mean_huber'][a] for v in rows)
                 for a in ('v10','render_mask_lift','rgb_mask_lift')} for n,rows in record['source_teacher_checks'].items()}
    record['source_teacher_macro'] = source
    replay_path=study/item['run']/'replay_comparison.json'
    replay=read(replay_path)
    prior=study/'nvos_image_response_v2'
    require(replay['images']==len(replay['rows'])==56 and not replay['target_labels_used']
            and replay['canonical_seal_sha256']==sha(study/item['run']/'predictions_complete.json')
            and replay['prior_seal_sha256']==sha(prior/'predictions_complete.json')
            and replay['native_binary_changed']==replay['scoring_grid_binary_changed']==0
            and replay['max_float_error']<1e-5, 'N04 replay differs')
    require(all(v['native_binary_changed']==v['scoring_grid_binary_changed']==0 for v in replay['rows'].values()),
            'N04 binary replay differs')
    record['replay_comparison']={k:v for k,v in replay.items() if k not in ('rows','source_text')}
    record['receipts'].update({str(p):sha(p) for p in (replay_path,prior/'predictions_complete.json',prior/'provenance_exception.json')})
    audit = study/'nvos_image_response_spatial_v2'
    if (audit/'summary.json').exists():
        s, b = read(audit/'summary.json'), read(audit/'prediction_binding.json')
        require(s['complete'] and s['diagnostic_only'] and not s['default_changed'] and s['durable_dense_prediction_bytes']==0,
                'N04 spatial audit scope changed')
        require(b['parent_summary_sha256']==sha(study/item['run']/'summary.json')
                and b['parent_seal_sha256']==sha(study/item['run']/'predictions_complete.json')
                and s['prediction_binding_sha256']==sha(audit/'prediction_binding.json')
                and s['figure_sha256']==sha(audit/'errors.png'), 'N04 spatial audit binding changed')
        for n,row in s['rows'].items():
            for a,metrics in row['metrics'].items():
                check_binary_metrics(metrics)
                require(all(metrics[k]==record['rows'][n][a]['retained_gt_gt127'][k] for k in metrics), 'N04 spatial counts changed')
        record['spatial_diagnosis'] = s['rows']
        record['receipts'].update({str(p):sha(p) for p in audit.iterdir() if p.is_file()})
    (PAPER/'nvos_image_response_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    flat = [dict(task=n,arm=a,gt_track=t,**m) for n,arms in record['rows'].items() for a,tracks in arms.items() for t,m in tracks.items()]
    with (PAPER/'artifacts/nvos_image_response.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    lines = [r'\begin{tabular}{lrrrrr}',r'\toprule',r'Arm & IoU & Precision & Recall & AP & LUDVIG GT IoU \\',r'\midrule']
    for arm,label in zip(ARMS,LABELS):
        values = [100*record['macro']['retained_gt_gt127'][arm][m] for m in ('iou','precision','recall','average_precision')]
        values.append(100*record['macro']['ludvig_gt_maximum'][arm]['iou'])
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/nvos_image_response.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\section{Image-conditioned query responses and 3D writeback}',
        'N04 pairs a fixed v10 response with reconstructed RGB or observed RGB at the same camera. '
        'The former uses only source-trained appearance; the latter explicitly changes input access. '
        'SAM3 receives a response-derived box, optionally with dense log-odds. '
        'Single-mask decoding has no ground-truth candidate selection. Both dense-prior outputs '
        'are additionally written to the original observed Gaussian domain by 120 Huber updates '
        'and rendered again. Unsupported elements retain v10. The primary arm, observed RGB with '
        'dense prior and 3D writeback, was specified before scoring.',
        r'\begin{center}\small\input{tables/nvos_image_response.tex}\end{center}',
        'All 56 predictions precede target scoring. This is the opened development cohort, '
        'not an independent test. Neither target RGB nor the inferred image mask is target GT. '
        'A view-conditioned 3D update is not evidence of one query-global field or improved '
        'cross-view consistency. Source losses below compare against fallible original teacher '
        'tracks and are distinct from target accuracy.']
    for key,label in [('rgb_mask_lift_vs_v10','v10'),('rgb_mask_lift_vs_render_mask_lift','the matched rendered-RGB arm'),('rgb_mask_lift_vs_rgb_mask','the direct image output')]:
        e = effects['retained_gt_gt127'][key]; lo,hi=e['descriptive_scene_bootstrap95_pp']
        lines.append(f"The primary arm changes mean IoU by {e['delta_pp']:+.4f} points versus {label} "
            f"({e['improved']} improved, {e['harmed']} harmed; descriptive seven-scene interval [{lo:+.4f},{hi:+.4f}]).")
    lines += [r'\begin{center}\small\begin{tabular}{lrrrrr}\toprule',
        r'Task & v10 IoU & RGB + 3D IoU & Source loss: v10 & Render + 3D & RGB + 3D \\\midrule']
    for n,v in record['rows'].items():
        vals=[100*v[a]['retained_gt_gt127']['iou'] for a in ('v10','rgb_mask_lift')]
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f'{x:.3f}' for x in vals)+' & '+
                     ' & '.join(f'{source[n][a]:.6f}' for a in ('v10','render_mask_lift','rgb_mask_lift'))+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        'Implementation checks reproduce v10 binary outputs exactly, compare the sparse compositor '
        'and normalized response operator with gsplat, and bind RGB parameters to the exact geometry. '
        'Both GT conventions use continuous-score bilinear resizing before thresholding. '
        'An independent confusion-matrix implementation and scikit-learn AP check all arms. '
        'Other benchmark outputs are unchanged references, not fresh inference.',
        r'\IfFileExists{../../output/tmm_response_20261016/study/nvos_image_response_spatial_v2/errors.png}{',
        r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth,height=.78\textheight,keepaspectratio]{../../output/tmm_response_20261016/study/nvos_image_response_spatial_v2/errors.png}',
        r'\caption{All eight N04 tasks. Green: true positive; red: false positive; blue: false negative. GT-conditioned diagnostics never enter inference.}\end{figure}}{}']
    (PAPER/'sections/nvos_image_response.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=effects,source=source),indent=2))


if __name__=='__main__':
    main()
