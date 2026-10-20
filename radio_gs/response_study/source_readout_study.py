"""Independent stdlib verification of source-only threshold estimation."""
import hashlib
import math
from statistics import mean

from .reference_study import read, sha, require, check_binary_metrics


def verify_histogram(record):
    bins=record['bins'];positive=record['positive_histogram'];negative=record['negative_histogram']
    require(type(bins) is int and bins>=2 and bins%2==0, 'invalid calibration bin count')
    require(len(positive)==len(negative)==bins and
            all(math.isfinite(x) and x>=0 for x in positive+negative), 'invalid source histogram')
    total=math.fsum(positive);scores=[];tp=fp=0.
    for p,n in zip(reversed(positive),reversed(negative)):
        tp+=p;fp+=n;scores.append(tp/max(total+fp,1e-12))
    scores.reverse()
    if total<=1e-12:
        index=bins//2;reason='no_positive_source_mass'
    else:
        maximum=max(scores[1:])
        tied=[i for i in range(1,bins) if abs(scores[i]-maximum)<=1e-12]
        index=min(tied,key=lambda i:(abs(i-bins//2),i));reason='maximum_soft_source_iou'
    require(record['bin_index']==index and record['threshold']==index/bins
            and record['criterion']==reason, 'source threshold does not follow declared rule')
    for actual,expected in [(record['positive_mass'],total),(record['negative_mass'],math.fsum(negative)),
                            (record['soft_iou'],scores[index]),(record['soft_iou_at_half'],scores[bins//2])]:
        require(math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10), 'source histogram aggregate changed')
    return index/bins


def check_source_readout_study(repo,study,item,jobs):
    root=study/item['run']
    if not (root/'summary.json').exists():
        return dict(**item,complete=False,status='in_progress' if root.exists() else 'planned')
    manifest=read(root/'manifest.json');config=manifest['config'];summary=read(root/'summary.json')
    seal=read(root/'predictions_complete.json');snapshot=read(root/'source_snapshot.json')
    nvos=[j for j in jobs if j['benchmark']=='nvos'];names={j['name'] for j in nvos}
    arms={'fixed_half','source_fit','source_holdout'}
    require(manifest['jobs']==nvos and len(names)==8 and set(config['arms'])==arms, 'N02 cohort changed')
    require(config['histogram_bins']==512 and config['source_holdout_stride']==4
            and config['source_holdout_phase']==0 and config['source_hw']==[384,512]
            and config['solver_steps']==120 and config['huber_clip']==.25, 'N02 design changed')
    require(summary['complete'] and not summary['target_gt_used_for_calibration']
            and not summary['target_rgb_used'] and not summary['default_changed']
            and not summary['independent_test'] and not summary['per_task_winners']
            and summary['shared_final_response'] and summary['durable_dense_prediction_bytes']==0,
            'N02 scientific scope changed')
    require(not seal['target_gt_read'] and seal['jobs']==8 and seal['decisions']==24
            and set(seal['report_hashes'])==set(seal['predictions'])==set(seal['thresholds'])==names
            and set(summary['rows'])==names,'N02 prediction or scoring subset changed')
    require(summary['prediction_seal_sha256']==sha(root/'predictions_complete.json'), 'N02 seal changed')
    parent=study/config['parent']/'summary.json';parent_record=read(parent)
    require(config['parent']=='nvos_coverage_evidence_v2' and config['parent_arm']=='coverage_high'
            and manifest['parent_summary_sha256']==sha(parent), 'N02 parent changed')
    receipts={str(root/p):sha(root/p) for p in
              ('manifest.json','source_snapshot.json','summary.json','predictions_complete.json')}
    receipts[str(parent)]=sha(parent)
    ranking_path=study/'nvos_coverage_ranking_v2/summary.json';parent_ranking=read(ranking_path)
    receipts[str(ranking_path)]=sha(ranking_path)
    for path,digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest,'N02 source archive changed')
    inputs=dict(summary['gt_hashes']);thresholds={};source_calibration={}
    for name,digest in seal['report_hashes'].items():
        path=root/(name+'_prediction.json');report=read(path)
        require(sha(path)==digest,'N02 prediction report changed');receipts[str(path)]=digest
        require(report['job'] in nvos and not report['target_gt_read'] and not report['target_rgb_used']
                and report['shared_final_response'] and report['calibration_pixels_identical'],
                'N02 prediction permissions or shared response changed')
        require(report['prediction']==seal['predictions'][name]
                and report['thresholds']==seal['thresholds'][name]
                and set(report['thresholds'])==arms and report['thresholds']['fixed_half']==.5,
                'N02 sealed response or threshold changed')
        fit=report['all_fit_frames'];heldout=report['calibration_frames'];split=report['split_fit_frames']
        require(heldout==sorted(fit)[::4] and set(split)==set(fit)-set(heldout)
                and set(split).isdisjoint(heldout) and set(heldout)=={r['frame'] for r in report['source_rows']},
                'N02 reconstruction holdout changed')
        for arm,record in report['calibration'].items():
            require(record['bins']==512 and verify_histogram(record)==report['thresholds'][arm],
                    'N02 reported threshold differs from source evidence')
            require(record['observations']==sum(r['common_pixels'] for r in report['source_rows']),
                    'N02 calibrators use different pixels')
        for k in ('positive_mass','negative_mass'):
            require(math.isclose(report['calibration']['source_fit'][k],report['calibration']['source_holdout'][k],
                                 rel_tol=1e-10,abs_tol=1e-8),'N02 calibration target or weights changed')
        thresholds[name]=report['thresholds'];source_calibration[name]=report['calibration']
        for p,h in report['input_hashes'].items():
            require(p not in inputs or inputs[p]==h,'N02 inconsistent shared input');inputs[p]=h
        path=root/(name+'_evaluation.json');receipts[str(path)]=sha(path)
        row=summary['rows'][name]
        require(read(path)==row and set(row)==arms,'N02 evaluation record changed')
        for arm,tracks in row.items():
            require(set(tracks)==set(config['gt_tracks']),'N02 GT tracks differ')
            for track,metrics in tracks.items():
                check_binary_metrics(metrics)
                require(metrics['average_precision']==row['fixed_half'][track]['average_precision'],
                        'N02 changed continuous ranking')
                require(metrics['tp']+metrics['fn']==row['fixed_half'][track]['tp']+row['fixed_half'][track]['fn'],
                        'N02 changed GT foreground count')
        for track in config['gt_tracks']:
            require(abs(row['fixed_half'][track]['iou']-parent_record['rows'][name]['coverage_high'][track]['iou'])<1e-5,
                    'N02 raw response does not reproduce N01')
            require(abs(row['fixed_half'][track]['average_precision']-parent_ranking['rows'][name]['coverage_high'][track]['average_precision'])<1e-5,
                    'N02 raw ranking does not reproduce N01')
    for path,digest in inputs.items():
        require(sha(path)==digest,'N02 changed input: '+path);receipts[path]=digest
    require(set(summary['macro'])==set(config['gt_tracks']),'N02 aggregate GT tracks changed')
    for track,values in summary['macro'].items():
        require(set(values)==arms,'N02 aggregate arm subset changed')
        for arm,metrics in values.items():
            for key,value in metrics.items():
                require(abs(mean(r[arm][track][key] for r in summary['rows'].values())-value)<1e-12,
                        'N02 aggregate changed')
    require(set(summary['other_benchmarks'])=={j['benchmark']+'/'+j['name'] for j in jobs if j['benchmark']!='nvos'}
            and not summary['other_benchmarks_freshly_evaluated'],'N02 other cohort changed')
    for record in summary['other_benchmarks'].values():
        require(not record['intervention'] and sha(record['path'])==record['sha256'],'N02 other result changed')
        receipts[record['path']]=record['sha256']
    return dict(**item,complete=True,status='completed_not_promoted',fresh_prediction_jobs=8,
                decisions=24,other_jobs_unchanged=23,other_jobs_fresh_inference=False,
                macro=summary['macro'],rows=summary['rows'],thresholds=thresholds,
                source_calibration=source_calibration,receipts=receipts,
                default_changed=False,independent_test=False,shared_final_response=True)


def check_score_changes(study):
    root=study/'nvos_score_changes_v1';parent=study/'nvos_source_readout_v1'
    manifest=read(root/'manifest.json');summary=read(root/'summary.json')
    joint=read(root/'joint_prediction_seal.json');old=read(root/'old_predictions_complete.json')
    seal=read(parent/'predictions_complete.json');new=read(parent/'summary.json')
    snapshot=read(root/'source_snapshot.json');names=set(new['rows'])
    require(summary['complete'] and not summary['gt_hybrids_are_deployable']
            and summary['score_space_attribution_not_upstream_causality']
            and not summary['new_method_gain'] and not summary['default_changed']
            and summary['durable_dense_prediction_bytes']==0,'A20 scope changed')
    require(set(summary['rows'])==set(old['rows'])==set(joint['exact_new_response_digests'])==names
            and len(names)==old['jobs']==8 and not old['target_gt_read'] and not joint['target_gt_read'],
            'A20 prediction cohort changed')
    require(joint['old_seal_sha256']==sha(root/'old_predictions_complete.json')
            and joint['parent_seal_sha256']==sha(parent/'predictions_complete.json')
            and joint['parent_summary_sha256']==sha(parent/'summary.json')
            and summary['joint_prediction_seal_sha256']==sha(root/'joint_prediction_seal.json')
            and summary['figure_sha256']==sha(root/'errors.png')
            and manifest['parent_manifest_sha256']==sha(parent/'manifest.json'), 'A20 binding changed')
    receipts={str(root/p):sha(root/p) for p in ('manifest.json','summary.json','source_snapshot.json',
              'joint_prediction_seal.json','old_predictions_complete.json','errors.png')}
    for path,digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest,'A20 source archive changed')
    inputs=dict(summary['gt_hashes'])
    for name,row in summary['rows'].items():
        require(not row['gt_hybrids_are_deployable'] and read(root/(name+'.json'))==row,'A20 row changed')
        receipts[str(root/(name+'.json'))]=sha(root/(name+'.json'))
        require(joint['exact_new_response_digests'][name]==seal['predictions'][name]['sha256'],
                'A20 did not use the sealed N02 response')
        report=old['rows'][name]
        require(not report['target_gt_read'] and not report['target_rgb_used'],'A20 original prediction access changed')
        for p,h in report['input_hashes'].items():
            require(p not in inputs or inputs[p]==h,'A20 shared input differs');inputs[p]=h
        for metrics in row['metrics'].values():check_binary_metrics(metrics)
        for arm,key in [('coverage','fixed_half'),('calibrated','source_holdout')]:
            require(row['metrics'][arm]=={k:new['rows'][name][key]['retained_gt_gt127'][k] for k in row['metrics'][arm]},
                    'A20 target counts differ from N02')
            change=row['transitions'][arm];a=row['metrics']['v10'];b=row['metrics'][arm]
            require(all(type(v) is int and v>=0 for v in change.values())
                    and b['tp']-a['tp']==change['recovered_foreground']-change['lost_foreground']
                    and b['fp']-a['fp']==change['added_false_positive']-change['removed_false_positive'],
                    'A20 pixel transitions do not match counts')
        ap=row['average_precision'];a=ap['v10'];b=ap['coverage']
        positive=ap['gt_hybrid_positive_only'];negative=ap['gt_hybrid_negative_only']
        require(all(math.isfinite(v) and 0<=v<=1 for v in (a,b,positive,negative)), 'A20 invalid AP')
        require(abs(ap['foreground_change_pp']-50*((positive-a)+(b-negative)))<1e-10
                and abs(ap['background_change_pp']-50*((negative-a)+(b-positive)))<1e-10
                and abs(ap['additive_residual'])<1e-12,'A20 attribution identity failed')
        require(abs(b-new['rows'][name]['fixed_half']['retained_gt_gt127']['average_precision'])<1e-12,
                'A20 new AP differs from N02')
        require(row['calibrated_threshold']==seal['thresholds'][name]['source_holdout'],'A20 readout changed')
        for arm in ('v10','coverage'):
            require(sum(row['score_histograms']['foreground'][arm])==row['metrics'][arm]['tp']+row['metrics'][arm]['fn'],
                    'A20 score histogram changed foreground count')
    for path,digest in inputs.items():
        require(sha(path)==digest,'A20 changed input: '+path);receipts[path]=digest
    return dict(complete=True,rows=summary['rows'],receipts=receipts,new_method_gain=False,
                gt_hybrids_are_deployable=False,score_space_attribution_not_upstream_causality=True)
