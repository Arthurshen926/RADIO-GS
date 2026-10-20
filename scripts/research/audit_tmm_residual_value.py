"""Diagnose residual utility on already-opened, sealed source/target results.

No parameter search or new benchmark decision. Candidate utility reuse requires
the exact candidate and target checks recorded by A12; all control scores are
then checked against their actual selected candidate, not inferred from a mean.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.querying.response_value import set_features
from radio_gs.querying.multitask_response_value import predict
from radio_gs.querying.response_study_adapters import read
from radio_gs.response_study.residual_diagnosis import decision_attribution
from scripts.research.train_tmm_multitask_value import load_all
from scripts.research.replay_tmm_response import STUDY,POLICY,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='residual_utility_audit_v1');args=p.parse_args()
    trained=STUDY/'response_value_residual_v1';transfer=STUDY/'response_value_residual_transfer_v1'
    diagnosis=STUDY/'response_value_residual_diagnosis_v1';history=STUDY/'response_replacement_capacity_v1'
    manifest=read(trained/'manifest.json');seal=read(trained/'model_seal.json')
    if sha(trained/'model.joblib')!=seal['model_sha256'] or sha(trained/'calibration.json')!=seal['calibration_sha256']:
        raise ValueError('sealed model/calibration changed')
    for path,digest in {**manifest['input_hashes'],**manifest['source_hashes']}.items():
        if sha(path)!=digest:raise ValueError('sealed training dependency changed: '+path)
    audit=read(diagnosis/'summary.json');audit_manifest=read(diagnosis/'manifest.json')
    if not audit['complete'] or not audit['uqis_cached_utilities_verified_against_exact_candidates_and_fresh_selected_scores']:
        raise ValueError('complete exact-candidate capacity audit required')
    for path,digest in audit_manifest['input_hashes'].items():
        if sha(path)!=digest:raise ValueError('exact-pool audit dependency changed: '+path)
    bundle=joblib.load(trained/'model.joblib');config=bundle['config'];arms=read(transfer/'manifest.json')['arms']
    if set(arms)!={'full','query_blind','permuted','unrestricted','unregularized'}:
        raise ValueError('all predeclared controls required')
    paths=[trained/n for n in ('model.joblib','model_seal.json','manifest.json','calibration.json','fit_diagnostics.json')]
    paths += [diagnosis/n for n in ('summary.json','manifest.json')]
    paths += [transfer/n for n in ('summary.json','manifest.json','predictions_complete.json')]
    paths += [history/'summary.json'];paths += list(map(Path,audit_manifest['input_hashes']))
    outputs={};null_behavior={}
    for split in ('calibration','validation'):
        reports=load_all(config,bundle['splits'][split]);outputs[split]={};null_behavior[split]={}
        for arm in arms:
            key=arm if arm in bundle['models'] else 'full'
            predictions=predict(reports,bundle['models'][key],blind=key=='query_blind',permuted=arm=='permuted')
            margin=0. if arm=='unrestricted' else bundle['margins'][key]
            outputs[split][arm]={task:decision_attribution([r for r in predictions if r['modality']==task],margin)
                                 for task in ('text','image','category')}
        for arm in ('full','unregularized'):
            null_behavior[split][arm]={}
            for task in ('text','image','category'):
                records=[]
                for report in reports:
                    if report['modality']!=task:continue
                    aligned=[];null=[];relative=[]
                    for row in report['rows']:
                        a=bundle['models'][arm].correction.predict(set_features(row['features']))
                        n=bundle['models'][arm].correction.predict(set_features(row['permuted_features']))
                        aligned.append(float(np.mean(a*a)));null.append(float(np.mean(n*n)))
                        relative.append(float(np.mean(((a-a[0])-(n-n[0]))**2)))
                    records.append(dict(scene=report['scene'],aligned_mse=float(np.mean(aligned)),
                        null_mse=float(np.mean(null)),correspondence_gain_mse=float(np.mean(relative))))
                null_behavior[split][arm][task]=dict(records=records,scene_macro={
                    k:float(np.mean([r[k] for r in records])) for k in ('aligned_mse','null_mse','correspondence_gain_mse')})
    transfer_summary=read(transfer/'summary.json');old={r['query_id']:r for r in read(history/'summary.json')['rows']}
    rows={a:[] for a in arms}
    for job in read(transfer/'manifest.json')['jobs']:
        path=transfer/(job['benchmark']+'_'+job['name']+'.json');paths.append(path);report=read(path)
        if job['benchmark']=='scannet':
            result=next(r for r in transfer_summary['category_rows'] if r['job']['name']==job['name'])
            values=[float(np.mean([result['metrics'][a][k] for k in ('miou_19','miou_15','miou_10')]))
                    for a in ('baseline','fixed_dual_native','fixed_source_readout')]
            for arm in arms:
                actual=float(np.mean([result['metrics'][arm][k] for k in ('miou_19','miou_15','miou_10')]))
                decision=report['arms'][arm]
                if abs(values[decision['candidate']]-actual)>1e-10:raise ValueError('category control differs from intact candidate')
                rows[arm].append(dict(scene=job['name'],modality='category',query_index=0,ious=values,
                    gains=decision['gains'],evidence_available=report['evidence']['verification_views']>0))
        else:
            actual={}
            for arm in arms:
                score_path=transfer/('uqis_'+arm+'_sets.json');paths.append(score_path)
                actual[arm]={r['query_id']:r['iou'] for r in read(score_path)['per_query']}
            for row in report['rows']:
                values=old[row['query_id']]['candidate_iou']
                for arm in arms:
                    decision=row['arms'][arm]
                    if abs(values[decision['candidate']]-actual[arm][row['query_id']])>1e-10:
                        raise ValueError('instance control differs from exact cached candidate')
                    rows[arm].append(dict(scene=job['name'],modality=row['modality'],query_index=row['query_id'],
                        ious=values,gains=decision['gains'],evidence_available=row['evidence']['verification_views']>0))
    outputs['benchmark']={}
    for arm in arms:
        key=arm if arm in bundle['models'] else 'full';margin=0. if arm=='unrestricted' else bundle['margins'][key]
        outputs['benchmark'][arm]={task:decision_attribution([r for r in rows[arm] if r['modality']==task],margin)
                                   for task in ('text','image','category')}
    sources=[__file__,'radio_gs/response_study/residual_diagnosis.py','radio_gs/response_study/utility_diagnosis.py',
             'radio_gs/querying/residual_response_value.py','radio_gs/querying/multitask_response_value.py',
             'radio_gs/querying/response_value.py','scripts/research/train_tmm_multitask_value.py']
    run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(2*2**20)
    run.json('manifest.json',dict(input_hashes={str(p):sha(p) for p in sorted(set(paths))},
        source_hashes={p:sha(p) for p in sources},parameters_changed=False,new_decisions=False,benchmark_labels_used=True))
    run.json('source_snapshot.json',{p:Path(p).read_text() for p in sources})
    run.json('summary.json',dict(complete=True,diagnostic_only=True,outputs=outputs,null_behavior=null_behavior,
        fit=read(trained/'fit_diagnostics.json'),all_control_utilities_checked_against_fresh_scores=True,
        decomposition='available = ranking regret + signed retention cost + actual gain',
        null_mse_is_not_accuracy=True,independent_final_test=False,default_changed=False,durable_dense_prediction_bytes=0))
    print({s:{a:{t:r['decomposition'] for t,r in tasks.items()} for a,tasks in arms_.items()} for s,arms_ in outputs.items()},flush=True)


if __name__=='__main__':main()
