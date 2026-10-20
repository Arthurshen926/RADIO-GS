"""Retained-record validation for targeted NVOS studies (no GT or inference)."""
import hashlib
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError('NVOS evidence study: '+message)


def check_binary_metrics(row):
    tp,fp,fn=[row[k] for k in ('tp','fp','fn')]
    require(all(type(x) is int and x>=0 for x in (tp,fp,fn)), 'invalid confusion counts')
    expected=dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                  precision=tp/(tp+fp) if tp+fp else 0.,
                  recall=tp/(tp+fn) if tp+fn else 0.)
    require(all(abs(row[k]-v)<1e-12 for k,v in expected.items()), 'counts and metrics differ')


def check_reference_study(repo, study, item, jobs):
    root = study/item['run']
    if not (root/'summary.json').exists():
        return dict(**item, complete=False, status='in_progress' if root.exists() else 'planned')
    manifest = read(root/'manifest.json')
    summary = read(root/'summary.json')
    seal = read(root/'predictions_complete.json')
    snapshot = read(root/'source_snapshot.json')
    expected = [j for j in jobs if j['benchmark']=='nvos']
    names = {j['name'] for j in expected}
    config = manifest['config']
    arms = set(config['arms'])
    require(len(names)==8 and manifest['jobs']==expected, 'task cohort differs')
    require(arms=={'legacy_low','coverage_low','legacy_high','coverage_high'}, 'factorial arms differ')
    require(summary['complete'] and summary['all_eight_completed'] and seal['jobs']==8
            and not seal['target_gt_read'] and set(seal['arms'])==arms, 'incomplete label-free seal')
    require(not summary['target_rgb_used'] and not summary['default_changed']
            and not summary['independent_test'] and not summary['other_benchmarks_freshly_evaluated']
            and summary['durable_dense_prediction_bytes']==0, 'scientific scope changed')
    require(summary['prediction_seal_sha256']==sha(root/'predictions_complete.json'), 'changed prediction seal')
    require(set(summary['rows'])==names and len(seal['predictions'])==32, 'partial scored cohort')
    require(config['threshold']==.5 and not config['per_scene_winner_selection'], 'readout changed')
    receipts = {str(root/p):sha(root/p) for p in
                ('manifest.json','summary.json','predictions_complete.json','source_snapshot.json')}
    for path, digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest,
                'archived source changed: '+path)
    input_hashes = {}
    maximum_error = 0.
    reference_path=repo/read(repo/'configs/research/tmm_response.json')['reference_summary']
    reference={r['job']['name']:r['metrics']['iou'] for r in read(reference_path)['rows']
               if r['job']['benchmark']=='nvos'}
    receipts[str(reference_path)]=sha(reference_path)
    for name in sorted(names):
        report = read(root/(name+'_prediction.json'))
        scored = read(root/(name+'_evaluation.json'))
        require(not report['target_gt_read'] and not report['target_rgb_used'], 'prediction access changed')
        require(report['job'] in expected and set(report['arms'])==arms, 'prediction task or arms changed')
        require(scored==summary['rows'][name], 'per-task scores changed')
        require(set(scored)==arms, 'scored arms changed')
        for values in scored.values():
            require(set(values)==set(config['gt_tracks']), 'scored GT tracks changed')
            for value in values.values():check_binary_metrics(value)
        require(abs(scored['legacy_low']['retained_gt_gt127']['iou']-reference[name])<1e-5,
                'baseline does not reproduce v10 task score')
        require(set(report['predictions'])=={name+'_'+a for a in arms}, 'prediction subset missing')
        for key,value in report['predictions'].items():
            require(value==seal['predictions'][key], 'prediction digest changed')
        compatibility = report['arms']['legacy_low']['compatibility']
        require(compatibility['maximum']<1e-4 and compatibility['mean']<1e-6, 'retained response mismatch')
        maximum_error=max(maximum_error,compatibility['maximum'])
        for p,h in report['input_hashes'].items():
            require(p not in input_hashes or input_hashes[p]==h, 'inconsistent shared input')
            input_hashes[p]=h
        for suffix in ('_prediction.json','_evaluation.json'):
            p=root/(name+suffix);receipts[str(p)]=sha(p)
    for path,digest in input_hashes.items():
        require(sha(path)==digest, 'changed input: '+path)
        receipts[path]=digest
    expected_other={j['benchmark']+'/'+j['name'] for j in jobs if j['benchmark']!='nvos'}
    require(set(summary['other_benchmarks'])==expected_other, 'other benchmark coverage changed')
    for record in summary['other_benchmarks'].values():
        require(not record['intervention'] and sha(record['path'])==record['sha256'], 'other benchmark changed')
        receipts[record['path']]=record['sha256']
    require(set(summary['macro'])==set(config['gt_tracks']), 'GT tracks changed')
    for track,values in summary['macro'].items():
        for arm,value in values.items():
            actual=mean(summary['rows'][name][arm][track]['iou'] for name in names)
            require(abs(actual-value)<1e-12, 'aggregate mismatch')
    return dict(**item,complete=True,status='completed_not_promoted',fresh_prediction_jobs=8,
                other_jobs_unchanged=23,other_jobs_fresh_inference=False,
                macro=summary['macro'],maximum_retained_field_error=maximum_error,
                receipts=receipts,independent_test=False)


def check_inherited_readouts(study,run_name='nvos_v10_readout_audit_v1'):
    root=study/run_name
    summary=read(root/'summary.json');manifest=read(root/'manifest.json')
    seal=read(root/'predictions_complete.json');snapshot=read(root/'source_snapshot.json')
    source=study/'nvos_coverage_evidence_v2'
    expected=set(read(source/'summary.json')['rows'])
    require(summary['complete'] and summary['inherited_output_audit']
            and not summary['new_method_gain'] and not summary['default_changed']
            and not summary['per_task_winners'] and summary['durable_dense_prediction_bytes']==0,
            'inherited-output scope changed')
    arms=set(manifest['config']['arms'])
    require(arms==({'semantic_response','independent_set_projection'} if run_name=='nvos_v10_readout_audit_v1'
                   else {'semantic_response','independent_set_projection','independent_response','semantic_set_projection'}),
            'inherited-output arm definitions changed')
    require(set(summary['rows'])==set(seal['report_hashes'])==expected and len(expected)==8
            and seal['jobs']==8 and len(seal['predictions'])==8*len(arms) and not seal['target_gt_read'],
            'inherited-output cohort or seal changed')
    require(summary['prediction_seal_sha256']==sha(root/'predictions_complete.json')
            and summary['joint_barrier_sha256']==sha(source/'predictions_complete.json'),
            'inherited-output prediction seal changed')
    receipts={str(root/p):sha(root/p) for p in
              ('manifest.json','summary.json','predictions_complete.json','source_snapshot.json')}
    for path,digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest,
                'inherited-output source changed')
    inputs=dict(summary['gt_hashes'])
    for name,digest in seal['report_hashes'].items():
        p=root/(name+'_prediction.json');require(sha(p)==digest,'inherited-output report changed')
        report=read(p);inputs.update(report['input_hashes']);receipts[str(p)]=digest
        require(not report['target_gt_read'] and not report['target_rgb_used'], 'inherited-output access changed')
        for key,value in report['predictions'].items():
            require(seal['predictions'][key]==value, 'inherited-output digest changed')
        p=root/(name+'_evaluation.json');receipts[str(p)]=sha(p)
        require(read(p)==summary['rows'][name], 'inherited-output score changed')
        require(set(summary['rows'][name])==arms,'inherited-output scored arm subset changed')
        for arm in summary['rows'][name].values():
            for metrics in arm.values():check_binary_metrics(metrics)
    for path,digest in inputs.items():
        require(sha(path)==digest,'inherited-output input changed: '+path);receipts[path]=digest
    for track,arms in summary['macro'].items():
        for arm,metrics in arms.items():
            for metric,value in metrics.items():
                actual=mean(r[arm][track][metric] for r in summary['rows'].values())
                require(abs(actual-value)<1e-12, 'inherited-output aggregate changed')
    return dict(complete=True,macro=summary['macro'],rows=summary['rows'],receipts=receipts,
                new_method_gain=False,output_operator_changed=True,metric='binary-mask IoU remains the same',
                independent_test=False)


def check_coverage_ranking(study):
    root=study/'nvos_coverage_ranking_v2';source=study/'nvos_coverage_evidence_v2'
    summary=read(root/'summary.json');manifest=read(root/'manifest.json');snapshot=read(root/'source_snapshot.json')
    require(summary['complete'] and summary['exact_prediction_digest_matches']==32
            and summary['all_confusion_counts_identical'] and not summary['new_predictions']
            and not summary['default_changed'] and summary['durable_dense_prediction_bytes']==0,
            'ranking scope changed')
    require(summary['source_summary_sha256']==sha(source/'summary.json')
            and summary['prediction_seal_sha256']==sha(source/'predictions_complete.json')
            and manifest['source_manifest_sha256']==sha(source/'manifest.json'), 'ranking source changed')
    require(set(summary['rows'])==set(read(source/'summary.json')['rows']), 'ranking task subset changed')
    receipts={str(root/p):sha(root/p) for p in ('manifest.json','summary.json','source_snapshot.json')}
    for path,digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest()==digest,'ranking source archive changed')
    for path,digest in summary['gt_hashes'].items():
        require(sha(path)==digest,'ranking GT changed');receipts[path]=digest
    for track,arms in summary['macro'].items():
        for arm,metrics in arms.items():
            for metric,value in metrics.items():
                actual=mean(r[arm][track][metric] for r in summary['rows'].values())
                require(abs(actual-value)<1e-12,'ranking aggregate changed')
    return dict(complete=True,macro=summary['macro'],receipts=receipts,new_predictions=False)


def check_exact_readout_capacity(study):
    root=study/'nvos_readout_capacity_v1';s=read(root/'summary.json');m=read(root/'manifest.json')
    baseline=read(study/'nvos_image_response_v3/summary.json');grid=read(study/'protocol_correctness_audit_v2/summary.json')
    require(s['complete'] and s['gt_oracle_diagnostic_only'] and not s['deployable'] and not s['thresholds_exported']
            and not s['default_changed'] and s['durable_dense_prediction_bytes']==0,'A23 oracle scope changed')
    require(hashlib.sha256(m['source_text'].encode()).hexdigest()==m['source_sha256'] and m['historical_predictions_only'],'A23 source changed')
    require(len(s['rows'])==8 and set(s['rows'])==set(baseline['rows']),'A23 incomplete cohort')
    for n,tracks in s['rows'].items():
        require(read(root/(n+'.json'))==tracks,'A23 task receipt changed')
        for t,r in tracks.items():
            check_binary_metrics(r['fixed_half']);old=baseline['rows'][n]['v10'][t]
            require(abs(r['fixed_half']['iou']-old['iou'])<1e-12 and abs(r['average_precision']-old['average_precision'])<1e-12
                    and r['fixed_half']['iou']<=r['oracle_iou']<=1 and type(r['unique_decisions']) is int and r['unique_decisions']>0,'A23 invalid curve/old-response match')
        require(tracks['retained_gt_gt127']['oracle_iou']+1e-12>=grid['nvos']['rows'][n]['diagnostic_grid_oracle']['iou'],'A23 exact curve below former grid')
    for t,r in s['macro'].items():
        for key in ['oracle_iou','average_precision']:
            require(abs(r[key]-mean(v[t][key] for v in s['rows'].values()))<1e-12,'A23 mean changed')
        require(abs(r['fixed_half_iou']-mean(v[t]['fixed_half']['iou'] for v in s['rows'].values()))<1e-12,'A23 fixed mean changed')
    for p,h in s['input_hashes'].items():require(sha(p)==h,'A23 input changed')
    return dict(exact_unique_score_thresholds=True,tie_groups_preserved=True,nondeployable_gt_oracle=True,macro=s['macro'])
