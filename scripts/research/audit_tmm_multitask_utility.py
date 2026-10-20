"""Audit fixed utility gains and category-domain shifts after joint model seal.

No new model, margin, dense prediction or target-benchmark utility is used here.
Calibration/development utilities were already opened by the sealed training.
"""
import argparse
from pathlib import Path
import time
import joblib
import numpy as np
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.querying.response_replacement import FEATURE_NAMES,STRUCTURE_NAMES
from radio_gs.querying.response_value import training_values,set_features
from radio_gs.querying.multitask_response_value import predict
from radio_gs.response_study.utility_diagnosis import range_audit,gain_audit
from scripts.research.train_tmm_multitask_value import load_all
from scripts.research.replay_tmm_response import STUDY,POLICY,sha
from radio_gs.querying.response_study_adapters import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',default='response_value_multitask_v1')
    parser.add_argument('--prepared-run',default='category_transfer_inputs_v2')
    parser.add_argument('--run',default='multitask_utility_audit_v1')
    parser.add_argument('--wait-for-training',action='store_true');args=parser.parse_args()
    trained=STUDY/args.training_run
    while args.wait_for_training and not (trained/'summary.json').exists():
        if read(STUDY/'category_multitask_program_v1/progress.json')['status']=='failed':
            raise RuntimeError('joint data/training program failed')
        print('utility audit waiting for complete sealed training',flush=True);time.sleep(30)
    summary=read(trained/'summary.json');seal=read(trained/'model_seal.json');manifest=read(trained/'manifest.json')
    if (not summary['complete'] or sha(trained/'model.joblib')!=seal['model_sha256']
            or sha(trained/'calibration.json')!=seal['calibration_sha256']):
        raise ValueError('complete sealed model and margin required')
    for p,h in {**manifest['source_hashes'],**manifest['input_hashes']}.items():
        if sha(p)!=h:raise ValueError('sealed training input or implementation changed: '+p)
    bundle=joblib.load(trained/'model.joblib');config=bundle['config'];splits=bundle['splits']
    if bundle['training_manifest_sha256']!=sha(trained/'manifest.json'):
        raise ValueError('model/manifest binding differs')
    fit=load_all(config,splits['fit'],fit=True)
    prepared=STUDY/args.prepared_run;preseal=read(prepared/'summary.json')
    target=[];paths=[trained/p for p in ('summary.json','model_seal.json','manifest.json','model.joblib','calibration.json')]
    paths += [prepared/'summary.json']
    for name,digest in preseal['report_hashes'].items():
        p=prepared/name
        if sha(p)!=digest:raise ValueError('category signature seal changed')
        target.append(read(p));paths.append(p)
    if len(target)!=10:raise ValueError('complete target menu cohort required')
    fullmap=[r['features'] for s in splits['fit'] for r in read(STUDY/config['category_run']/(s+'.json'))['rows'] if r['fold']=='full']
    raw_fit=np.concatenate(fullmap);raw_target=np.concatenate([r['features'] for r in target])
    comparison=range_audit(raw_fit,raw_target,FEATURE_NAMES)
    x,_,_=training_values(fit)
    target_x=np.concatenate([set_features(r['features']) for r in target])
    set_names=[p+n for p in ('candidate/','set_mean/','centered/') for n in FEATURE_NAMES]
    allfit_range=range_audit(x,target_x,set_names)
    importance=np.asarray(bundle['models']['full'].feature_importances_).reshape(3,len(FEATURE_NAMES))
    influence={group:{'structural':float(v[:len(STRUCTURE_NAMES)].sum()),
                      'query':float(v[len(STRUCTURE_NAMES):].sum())}
               for group,v in zip(('candidate','set_mean','centered'),importance)}
    candidate_variation=[np.ptp(np.asarray(row['features']),axis=0) for report in fit for row in report['rows']]
    context_fraction=(np.asarray(candidate_variation)<=1e-6).mean(0)
    outputs={}
    for split in ('calibration','validation'):
        reports=load_all(config,splits[split]);outputs[split]={}
        for arm in ('full','query_blind','permuted','unrestricted'):
            key='query_blind' if arm=='query_blind' else 'full'
            predictions=predict(reports,bundle['models'][key],blind=key=='query_blind',permuted=arm=='permuted')
            margin=0. if arm=='unrestricted' else bundle['margins'][key]
            outputs[split][arm]={m:gain_audit([r for r in predictions if r['modality']==m],margin)
                                for m in ('text','image','category')}
    source=[__file__,'radio_gs/response_study/utility_diagnosis.py','radio_gs/querying/response_value.py',
        'radio_gs/querying/multitask_response_value.py','scripts/research/train_tmm_multitask_value.py']
    run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(2*2**20)
    run.json('manifest.json',dict(training_run=args.training_run,prepared_run=args.prepared_run,
        input_hashes={str(p):sha(p) for p in paths},source_hashes={p:sha(p) for p in source},
        diagnostic_only=True,target_benchmark_ground_truth_read=False,model_or_margin_changed=False))
    run.json('source_snapshot.json',{p:Path(p).read_text() for p in source})
    run.json('summary.json',dict(complete=True,diagnostic_only=True,default_changed=False,
        category_fullmap_range=comparison,all_task_learner_input_range=allfit_range,
        impurity_importance=influence,importance_is_not_causal_evidence=True,
        candidate_constant_fraction=dict(zip(FEATURE_NAMES,context_fraction.tolist())),
        independent_outputs=outputs,benchmark_labels_used=False,durable_dense_prediction_bytes=0,
        complete_menu_selected_fraction_interpretation='observed carrier fraction divided by 19; invariant across candidates',
        coverage_equivalence_not_established=True,independent_final_test=False))
    print('fixed utility diagnosis', {s:{a:{m:{k:v[k] for k in ('changed','improved','harmed','scene_macro')} for m,v in tasks.items()}
        for a,tasks in arms.items()} for s,arms in outputs.items()},flush=True)


if __name__=='__main__':main()
