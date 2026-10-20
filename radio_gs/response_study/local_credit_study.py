"""Stdlib receipts, cohort, accounting and matched-budget checks for N03."""
import hashlib
import json
import math
from statistics import mean
from .reference_study import read, sha, require, check_binary_metrics

ARMS={'v10','coverage','coverage_anchor_domain','fit_local','held_local','held_uniform'}


def check_local_credit_study(repo,study,item,jobs):
    root=study/item['run']
    if not (root/'summary.json').exists():
        return dict(**item,complete=False,status='failed' if (root/'failure.json').exists()
                    else 'in_progress' if root.exists() else 'planned')
    manifest=read(root/'manifest.json');config=manifest['config'];summary=read(root/'summary.json')
    seal=read(root/'predictions_complete.json');snapshot=read(root/'source_snapshot.json')
    nvos=[j for j in jobs if j['benchmark']=='nvos'];names={j['name'] for j in nvos}
    require(manifest['jobs']==nvos and len(names)==8 and set(config['arms'])==ARMS,'N03 cohort changed')
    require(config['source_hw']==[384,512] and config['solver_steps']==120 and config['huber_clip']==.25
            and config['source_holdout_stride']==4 and config['source_holdout_phase']==0
            and config['threshold']==.5 and config['primary_arm']=='held_local','N03 design changed')
    require(summary['complete'] and not summary['target_gt_used_for_prediction']
            and not summary['target_rgb_used'] and not summary['default_changed']
            and not summary['independent_test'] and not summary['per_task_winners']
            and summary['durable_dense_prediction_bytes']==0,'N03 scientific scope changed')
    require(not seal['target_gt_read'] and seal['jobs']==8 and seal['images']==48 and seal['threshold']==.5
            and set(seal['report_hashes'])==set(seal['predictions'])==set(summary['rows'])==names,
            'N03 prediction/scoring subset changed')
    require(summary['prediction_seal_sha256']==sha(root/'predictions_complete.json'),'N03 seal changed')
    parent_path=study/config['parent']/'summary.json';parent=read(parent_path)
    require(config['parent']=='nvos_source_readout_v1' and manifest['parent_summary_sha256']==sha(parent_path),
            'N03 parent changed')
    receipts={str(root/p):sha(root/p) for p in ('manifest.json','source_snapshot.json','summary.json','predictions_complete.json')}
    receipts[str(parent_path)]=sha(parent_path)
    require(set(snapshot)==set(manifest['source_hashes']),'N03 source archive subset changed')
    for path,digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest==snapshot[path]['sha256'],
                'N03 source archive changed')
    reference_path=repo/read(repo/'configs/research/tmm_response.json')['reference_summary']
    reference={r['job']['name']:r['metrics'] for r in read(reference_path)['rows'] if r['job']['benchmark']=='nvos'}
    receipts[str(reference_path)]=sha(reference_path)
    require(config==json.loads(snapshot['configs/research/nvos_local_credit.json']['text']),
            'N03 manifest config differs from archived design')
    inputs=dict(summary['gt_hashes']);source={};edits={};supports={};objectives={}
    for name,digest in seal['report_hashes'].items():
        path=root/(name+'_prediction.json');report=read(path)
        require(sha(path)==digest and report['job'] in nvos and report['job']['name']==name,
                'N03 prediction report changed');receipts[str(path)]=digest
        require(not report['target_gt_read'] and not report['target_rgb_used'] and report['calibration_pixels_identical'],
                'N03 prediction permissions changed')
        require(report['predictions']==seal['predictions'][name] and set(report['predictions'])==ARMS,
                'N03 sealed images changed')
        fit=report['all_fit_frames'];held=report['calibration_frames'];split=report['split_fit_frames']
        require(held==sorted(fit)[::4] and set(split)==set(fit)-set(held) and set(split).isdisjoint(held)
                and {r['frame'] for r in report['source_rows']}==set(held),'N03 split changed')
        for arm,row in report['credit'].items():
            require(row['observations']==len(held) and all(math.isfinite(row[k]) for k in
                ('source_huber_gain','allocated_gain','gross_credit','conservation_error')),'N03 invalid credit')
            require(abs(row['allocated_gain']-row['source_huber_gain'])<1e-6*max(1,abs(row['source_huber_gain']))
                    and abs(row['conservation_error']-(row['allocated_gain']-row['source_huber_gain']))<1e-10,
                    'N03 source loss accounting failed')
            require(row['gross_credit']>=abs(row['allocated_gain'])-1e-8 and 0<=row['gate_mean']<=1
                    and 0<=row['full_gates']<=row['nonzero_gates'],'N03 invalid local gate accounting')
        require(set(report['credit'])=={'fit_local','held_local'},'N03 credit arms changed')
        for sign in ('positive_edit_mass','negative_edit_mass'):
            local=report['edits']['held_local'][sign];uniform=report['edits']['held_uniform'][sign]
            require(local>=0 and abs(local-uniform)<1e-5*max(1,local),'N03 matched edit budget failed')
        for p,h in report['input_hashes'].items():
            require(p not in inputs or inputs[p]==h,'N03 inconsistent shared input');inputs[p]=h
        source[name]=report['credit'];edits[name]=report['edits'];supports[name]=report['support_elements']
        objectives[name]=report['source_huber_objectives']
        require(set(objectives[name])==ARMS and all(math.isfinite(v) and v>=0 for v in objectives[name].values()),
                'N03 source objective invalid')
        path=root/(name+'_evaluation.json');row=summary['rows'][name]
        require(read(path)==row and set(row)==ARMS,'N03 evaluation record changed');receipts[str(path)]=sha(path)
        for arm,tracks in row.items():
            require(set(tracks)==set(config['gt_tracks']),'N03 GT tracks differ')
            for track,metrics in tracks.items():
                check_binary_metrics(metrics)
                require(math.isfinite(metrics['average_precision']) and 0<=metrics['average_precision']<=1,
                        'N03 invalid AP')
                require(metrics['tp']+metrics['fn']==row['v10'][track]['tp']+row['v10'][track]['fn'],
                        'N03 target foreground changed')
        for track in config['gt_tracks']:
            for key in ('iou','average_precision'):
                require(abs(row['coverage'][track][key]-parent['rows'][name]['fixed_half'][track][key])<1e-5,
                        'N03 coverage response changed')
        for key,old_key in [('iou','iou'),('average_precision','ap')]:
            require(abs(row['v10']['retained_gt_gt127'][key]-reference[name][old_key])<1e-5,
                    'N03 original v10 reproduction changed')
    for path,digest in inputs.items():
        require(sha(path)==digest,'N03 input changed: '+path);receipts[path]=digest
    require(set(summary['macro'])==set(config['gt_tracks']),'N03 aggregate tracks changed')
    for track,values in summary['macro'].items():
        require(set(values)==ARMS,'N03 aggregate arms changed')
        for arm,metrics in values.items():
            for key,value in metrics.items():
                require(abs(mean(r[arm][track][key] for r in summary['rows'].values())-value)<1e-12,
                        'N03 aggregate changed')
    require(set(summary['other_benchmarks'])=={j['benchmark']+'/'+j['name'] for j in jobs if j['benchmark']!='nvos'}
            and not summary['other_benchmarks_freshly_evaluated'],'N03 other cohort changed')
    for record in summary['other_benchmarks'].values():
        require(not record['intervention'] and sha(record['path'])==record['sha256'],'N03 other result changed')
        receipts[record['path']]=record['sha256']
    return dict(**item,complete=True,status='completed_not_promoted',fresh_prediction_jobs=8,
                images=48,other_jobs_unchanged=23,other_jobs_fresh_inference=False,
                macro=summary['macro'],rows=summary['rows'],source_credit=source,edits=edits,
                support_elements=supports,source_huber_objectives=objectives,receipts=receipts,
                default_changed=False,independent_test=False,primary_arm='held_local')


def check_local_credit_spatial(study):
    root=study/'nvos_local_credit_spatial_v1';parent=study/'nvos_local_credit_v1'
    manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    binding=read(root/'prediction_binding.json');snapshot=read(root/'source_snapshot.json')
    seal=read(parent/'predictions_complete.json');reference=read(parent/'summary.json')
    names=set(reference['rows']);arms={'v10','coverage','held_local','held_uniform'}
    require(summary['complete'] and summary['diagnostic_only'] and not summary['default_changed']
            and summary['durable_dense_prediction_bytes']==0,'A21 scope changed')
    require(len(names)==8 and set(summary['rows'])==names and binding['jobs']==8 and binding['images']==32
            and binding['all_parent_images']==48 and not binding['target_gt_read'],'A21 cohort changed')
    require(binding['parent_seal_sha256']==sha(parent/'predictions_complete.json')
            and binding['parent_summary_sha256']==sha(parent/'summary.json')
            and manifest['parent_manifest_sha256']==sha(parent/'manifest.json')
            and summary['prediction_binding_sha256']==sha(root/'prediction_binding.json')
            and summary['figure_sha256']==sha(root/'errors.png'),'A21 bindings changed')
    require(set(binding['exact_prediction_digests'])=={n+'_'+a for n in names for a in arms},
            'A21 exact prediction cohort changed')
    receipts={str(root/p):sha(root/p) for p in ('manifest.json','summary.json','source_snapshot.json','prediction_binding.json','errors.png')}
    for p in ('manifest.json','summary.json','predictions_complete.json'):receipts[str(parent/p)]=sha(parent/p)
    for path,digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest==snapshot[path]['sha256'],
                'A21 source archive changed')
    for name,row in summary['rows'].items():
        path=root/(name+'.json');require(read(path)==row,'A21 per-task record changed');receipts[str(path)]=sha(path)
        require(set(row['metrics'])==arms,'A21 arm set changed')
        for arm,metrics in row['metrics'].items():
            check_binary_metrics(metrics)
            require(metrics=={k:reference['rows'][name][arm]['retained_gt_gt127'][k] for k in metrics}
                    and binding['exact_prediction_digests'][name+'_'+arm]==seal['predictions'][name][arm]['sha256'],
                    'A21 metrics/digests differ from parent')
            for label in ('foreground','background'):
                hist=row['score_histograms'][label][arm]
                require(len(hist)==len(row['score_histogram_edges'])-1 and all(type(v) is int and v>=0 for v in hist),
                        'A21 invalid histogram')
                if label=='foreground':require(sum(hist)==metrics['tp']+metrics['fn'],'A21 foreground changed')
        require(set(row['transitions'])=={'held_local_vs_'+b for b in ('v10','coverage','held_uniform')},
                'A21 comparisons changed')
        for key,changes in row['transitions'].items():
            a,b=key.split('_vs_');before,after=row['metrics'][b],row['metrics'][a]
            require(all(type(v) is int and v>=0 for v in changes.values())
                    and after['tp']-before['tp']==changes['recovered_foreground']-changes['lost_foreground']
                    and after['fp']-before['fp']==changes['added_false_positive']-changes['removed_false_positive'],
                    'A21 transitions differ from confusion counts')
    require(summary['gt_hashes']==reference['gt_hashes'],'A21 GT cohort changed')
    for path,digest in summary['gt_hashes'].items():
        require(sha(path)==digest,'A21 target changed');receipts[path]=digest
    return dict(complete=True,rows=summary['rows'],receipts=receipts,diagnostic_only=True)


def check_reference_scope(study):
    root=study/'nvos_reference_scope_v1';manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    snapshot=read(root/'source_snapshot.json');jobs=manifest['jobs'];names={j['name'] for j in jobs}
    from pathlib import Path
    recipe=read(Path(__file__).resolve().parents[2]/'configs/pipeline/system_v10.json')
    require(jobs==[j for j in recipe['jobs'] if j['benchmark']=='nvos'],'A22 canonical cohort differs')
    require(summary['complete'] and len(names)==8 and set(summary['rows'])==names
            and not summary['reference_mask_is_gt'] and not summary['target_gt_read']
            and not summary['target_rgb_read'] and not summary['new_method_gain']
            and summary['unchanged_inference'] and summary['durable_dense_prediction_bytes']==0,'A22 scope changed')
    require(not manifest['target_gt_read'] and not manifest['target_rgb_read']
            and summary['figure_sha256']==sha(root/'reference_scope.jpg'),'A22 access or figure changed')
    receipts={str(root/p):sha(root/p) for p in ('manifest.json','summary.json','source_snapshot.json','reference_scope.jpg')}
    for path,digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest==snapshot[path]['sha256'],
                'A22 source archive changed')
    for job in jobs:
        name=job['name'];row=summary['rows'][name];path=root/(name+'.json')
        require(read(path)==row,'A22 per-task record changed');receipts[str(path)]=sha(path)
        check_binary_metrics(row['agreement']);counts=row['agreement']
        require(row['seed_pixels']==counts['tp']+counts['fn'] and row['track_pixels']==counts['tp']+counts['fp']
                and row['seed_only_pixels']==counts['fn'] and row['track_only_pixels']==counts['fp']
                and not row['reference_mask_is_gt'] and not row['target_gt_read'] and not row['target_rgb_read'],
                'A22 source agreement counts/access changed')
        query=read(job['queries']);spec=read(job['evaluation'])
        require(row['reference_frame']==query['reference_frame'] and row['target_frame']==spec['target_frame']
                and row['reference_frame']!=row['target_frame'] and not query['reference_mask_is_gt'],
                'A22 source/target query mismatch')
    for path,digest in summary['input_hashes'].items():
        require(sha(path)==digest,'A22 input changed: '+path);receipts[path]=digest
    # Bind the frozen fitted-query lineage as an additional independent check.
    from pathlib import Path
    fits=[Path(p) for p in summary['input_hashes'] if p.endswith('/query/reference/0/fit.json')]
    require(len(fits)==8,'A22 fitted query cohort changed')
    for path in fits:
        fit=read(path);fitted=read(fit['query']);scene=read(fit['scene']);name=path.parents[3].name
        job=next(j for j in jobs if j['name']==name);query=read(job['queries'])
        require(all(fitted.get(k)==query.get(k) for k in ('reference_frame','reference_rgb','reference_mask'))
                and query['reference_frame'] in scene['source_frames']
                and Path(query['reference_rgb']).resolve()==Path(scene['source_rgb'][str(query['reference_frame'])]).resolve(),
                'A22 legal-source/fitted-query lineage differs')
        receipts[fit['query']]=sha(fit['query'])
    return dict(complete=True,rows=summary['rows'],receipts=receipts,source_agreement_not_accuracy=True)
