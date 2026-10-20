"""Same frozen utility/margin on complete category menus and UQIS answers.

All predictions for both families are sealed before target scoring. LERF and
NVOS remain explicit historical references, never new transfer successes.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from radio_gs.query_field.bounded_experiment import BoundedExperiment,tree_bytes
from radio_gs.querying.response_study_adapters import SYSTEM,read,load,readout_path
from radio_gs.querying.response_replacement import FEATURE_NAMES,choose_response
from radio_gs.querying.response_value import relative_utilities
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from scripts.research.replay_tmm_response import STUDY,POLICY,sha
from scripts.research.tmm_ram_predictions import RamPredictionBank
from scripts.research.run_tmm_support_agreement import PredictionMemory,RAM_LIMIT
from scripts.research.evaluate_tmm_response_replacement import predict as predict_uqis,score as score_uqis,ARMS
from scripts.research.run_tmm_response_study import score_scannet
from scripts.research.prepare_tmm_category_transfer import PROVIDERS


def load_policy(training):
    manifest=read(training/'manifest.json'); seal=read(training/'model_seal.json')
    if (not read(training/'summary.json')['complete'] or sha(training/'model.joblib')!=seal['model_sha256']
            or sha(training/'calibration.json')!=seal['calibration_sha256']):
        raise ValueError('complete sealed joint training required')
    for p,h in manifest['source_hashes'].items():
        if sha(p)!=h:raise ValueError('trained implementation changed: '+p)
    for p,h in manifest['input_hashes'].items():
        if sha(p)!=h:raise ValueError('training input seal changed: '+p)
    bundle=joblib.load(training/'model.joblib')
    if bundle['training_manifest_sha256']!=sha(training/'manifest.json'):
        raise ValueError('model is not bound to this training manifest')
    calibration=read(training/'calibration.json')
    if any(bundle['margins'][k]!=v['selection']['margin'] for k,v in calibration.items()):
        raise ValueError('checkpoint and calibration margin differ')
    if bundle['feature_names']!=FEATURE_NAMES or bundle['decision_rule']!='set_value':
        raise ValueError('unexpected shared response policy')
    parent=read(bundle['config']['parent_config'])
    text_manifest=read(STUDY/bundle['config']['text_run']/'manifest.json')
    provider=parent['provider_checkpoint']
    if sha(provider)!=text_manifest['input_hashes'][provider]:
        raise ValueError('original member decoder changed')
    bundle['provider_sha256']=text_manifest['input_hashes'][provider]
    # The model keeps its own protocol. This adds only the inherited decoder
    # interface expected by the existing, unchanged UQIS candidate constructor.
    bundle['config']=dict(bundle['config'],candidate_budget=parent['candidate_budget'],provider_checkpoint=provider)
    return bundle,seal


def decide_menu(report,bundle,*,arms=ARMS):
    decisions={}
    for arm in arms:
        condition=arm if arm in ('query_blind','unregularized') else 'full'
        x=report['permuted_features' if arm=='permuted' else 'features']
        gains=relative_utilities(bundle['models'][condition],x,query_blind=condition=='query_blind')
        margin=0. if arm=='unrestricted' else bundle['margins'][condition]
        winner=choose_response(gains,margin,evidence_available=report['evidence']['verification_views']>0)
        decisions[arm]=dict(candidate=winner,provider=PROVIDERS[winner],gains=gains.tolist(),margin=margin)
    return decisions


def wait_for_inputs(training,prepared):
    while not (training/'summary.json').exists() or not (prepared/'summary.json').exists():
        status_path=STUDY/'category_multitask_program_v1/progress.json'
        if status_path.exists() and read(status_path)['status']=='failed':
            raise RuntimeError('upstream corpus/training program failed; no partial-cohort transfer')
        print('waiting for sealed joint training and all ten category pools',flush=True)
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',default='response_value_multitask_v1')
    p.add_argument('--prepared-run',default='category_transfer_inputs_v2')
    p.add_argument('--run',default='response_value_multitask_transfer_v1')
    p.add_argument('--include-no-null-control',action='store_true')
    p.add_argument('--wait-for-inputs',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    training=STUDY/args.training_run; prepared=STUDY/args.prepared_run
    if args.wait_for_inputs:wait_for_inputs(training,prepared)
    bundle,seal=load_policy(training); source_manifest=read(prepared/'manifest.json'); source_seal=read(prepared/'summary.json')
    arms=(*ARMS,'unregularized') if args.include_no_null_control else ARMS
    if args.include_no_null_control and 'unregularized' not in bundle['models']:
        raise ValueError('matched no-null control must be trained and calibrated in the sealed bundle')
    if not source_seal['complete'] or source_seal['jobs']!=10 or source_seal['ground_truth_read']:
        raise ValueError('ten category candidate pools must be sealed without labels')
    for path,digest in source_manifest['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('prepared candidate implementation changed: '+path)
    source=BoundedExperiment(STUDY,args.prepared_run,POLICY,resume=True)
    bank=RamPredictionBank(source_manifest['ram_directory'])
    run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(8*2**20)
    jobs=read('configs/pipeline/system_v10.json')['jobs']; category=[j for j in jobs if j['benchmark']=='scannet']
    uqis=[j for j in jobs if j['benchmark']=='uqis']
    if source_manifest['jobs']!=category or len(uqis)!=9:
        raise ValueError('fixed joint transfer cohort changed')
    paths=[__file__,'radio_gs/querying/category_response_transfer.py','radio_gs/querying/response_value.py',
        'radio_gs/querying/response_replacement.py','scripts/research/prepare_tmm_category_transfer.py',
        'scripts/research/evaluate_tmm_response_replacement.py','scripts/research/run_tmm_response_study.py',
        'radio_gs/benchmarks/uqis/evaluate.py','radio_gs/benchmarks/uqis/metrics.py',
        'radio_gs/querying/direct_anchor_membership.py','radio_gs/querying/instance_observation_scope.py',
        'radio_gs/querying/complete_response_verification.py','radio_gs/querying/registered_observation_query.py',
        'radio_gs/querying/anchor_observation_evidence.py','radio_gs/querying/observation_readout.py',
        'scripts/research/run_tmm_support_agreement.py','scripts/research/summarize_tmm_response_study.py',
        'radio_gs/scannet_constants.py','configs/pipeline/system_v10.json']
    if args.include_no_null_control:
        paths.append('radio_gs/querying/residual_response_value.py')
    run.json('manifest.json',dict(training_run=args.training_run,prepared_run=args.prepared_run,
        model_sha256=seal['model_sha256'],calibration_sha256=seal['calibration_sha256'],margins=bundle['margins'],
        training_summary_sha256=sha(training/'summary.json'),
        prepared_summary_sha256=sha(prepared/'summary.json'),jobs=category+uqis,arms=arms,
        matched_no_null_control=args.include_no_null_control,
        source_hashes={s:sha(s) for s in paths},one_shared_policy=True,benchmark_parameter_search=False,
        scope='ten category menus and 134 UQIS text/image interventions; all 268 UQIS scores',
        other_outputs='LERF semantic, selected projection and NVOS remain retained references',default_changed=False))
    run.json('source_snapshot.json',{s:Path(s).read_text() for s in paths})
    memory=PredictionMemory(limit=RAM_LIMIT-bank.bytes); menus=[]; instances=[]
    for job in category:
        name=job['name']+'.json'
        if sha(prepared/name)!=source_seal['report_hashes'][name]:
            raise ValueError('sealed category input report changed')
        report=read(prepared/name)
        for p,h in report['input_hashes'].items():
            if sha(p)!=h:raise ValueError('category input changed: '+p)
        values=[bank.read(source,job['name']+'_'+name,len(report['columns'])) for name in PROVIDERS]
        for name,value in zip(PROVIDERS,values):
            if bank.entries[bank.key(source,job['name']+'_'+name)]['sha256']!=report['prediction_arrays'][name]['sha256']:
                raise ValueError('candidate array differs from pre-model seal')
        data=load(readout_path(job)); observed=data['semantic_observed']
        for subset in SPLITS.values():
            ids=[report['class_ids'].index(c) for c in subset]
            if ((data['category_logits'][:,ids].argmax(1)!=values[0][:,ids].argmax(1))&observed).any():
                raise ValueError('incumbent probability round trip changed original menu labels')
        decisions=decide_menu(report,bundle,arms=arms)
        arrays={a:memory.add(run,job['name']+'_'+a,values[d['candidate']]) for a,d in decisions.items()}
        receipt=dict(job=job,columns=report['columns'],arms=decisions,evidence=report['evidence'],
            prepared_report_sha256=sha(prepared/(job['name']+'.json')),prediction_arrays=arrays,
            ground_truth_read=False,all_submenus_share_one_candidate=True)
        run.json('scannet_'+job['name']+'.json',receipt);menus.append(receipt);del values,data;gc.collect()
    for job in uqis:
        report=predict_uqis(job,run,memory,bundle,arms=arms);instances.append(report)
        run.json('uqis_'+job['name']+'.json',report);gc.collect()
    if sum(len(r['rows']) for r in instances)!=134:
        raise ValueError('complete UQIS text/image cohort changed')
    names=['scannet_'+j['name']+'.json' for j in category]+['uqis_'+j['name']+'.json' for j in uqis]
    run.json('predictions_complete.json',dict(category_jobs=10,uqis_intervention_queries=134,
        report_hashes={n:sha(run.root/n) for n in names},ground_truth_read=False,
        combined_transient_prediction_bytes=bank.bytes+memory.bytes))
    for report in instances:
        if any(sha(p)!=h for p,h in report['input_hashes'].items()):
            raise ValueError('UQIS inputs changed after prediction seal')
    if any(sha(p)!=h for p,h in read(run.root/'manifest.json')['source_hashes'].items()):
        raise ValueError('inference implementation changed before scoring')
    # Both families have complete predictions before this first target-label access.
    evaluation_paths=[Path(j['evaluation']) for j in category]
    evaluation_paths += [Path(read(p)['ground_truth']) for p in evaluation_paths.copy()]
    evaluation_hashes={str(p):sha(p) for p in evaluation_paths}
    run.json('evaluation_manifest.json',dict(prediction_seal_sha256=sha(run.root/'predictions_complete.json'),
        category_evaluator_hashes=evaluation_hashes,inputs_rechecked=True,
        uqis_evaluator_hashes_location='summary.json/uqis_evaluator_hashes'))
    results=[]
    fixed=('fixed_dual_native','fixed_source_readout')
    for job,report in zip(category,menus):
        def response_loader(_,arm,count):
            if arm in fixed:return bank.read(source,job['name']+'_'+arm.removeprefix('fixed_'),count)
            return memory.read(run,job['name']+'_'+arm,count)
        result=score_scannet(job,run,report,response_loader=response_loader,arms=(*arms,*fixed))
        run.json('scannet_'+job['name']+'_evaluation.json',result);results.append(dict(job=job,**result))
        print('multitask category scored',job['name'],result['metrics'],flush=True)
    uqis_result=score_uqis(uqis,instances,run,memory,arms=arms)
    if any(sha(p)!=h for p,h in evaluation_hashes.items()):
        raise ValueError('category scoring inputs changed')
    source_macro={a:{k:float(np.mean([r['metrics'][a][k] for r in results])) for k in results[0]['metrics'][a]}
                  for a in results[0]['metrics']}
    reference=read(SYSTEM/'summary.json')
    summary=dict(complete=True,category_macro=source_macro,category_rows=results,uqis_macro=uqis_result['uqis_macro'],
        uqis_effects=uqis_result['effects'],uqis_changed_queries=uqis_result['changed_queries'],
        uqis_evaluator_hashes=uqis_result['evaluator_hashes'],
        freshly_scored_category_scenes=10,freshly_scored_uqis_queries=268,uqis_point_outputs_exact=True,
        one_shared_checkpoint=True,one_shared_full_policy_margin=bundle['margins']['full'],
        other_benchmarks={k:dict(status='transfer_pending',fresh_inference=False,metrics=reference['macro'][k]) for k in ('lerf','nvos')},
        all_five_transfer_complete=False,independent_final_test=False,default_changed=False,durable_dense_prediction_bytes=0,
        study_bytes=tree_bytes(STUDY))
    run.json('summary.json',summary);memory.values.clear();gc.collect();bank.close()
    run.json('ram_released.json',dict(complete=True,category_bank_released=True,all_dense_arrays_released=True))
    print(json.dumps(dict(category=source_macro,uqis=summary['uqis_macro']),indent=2),flush=True)


if __name__=='__main__':main()
