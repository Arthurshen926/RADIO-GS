"""Matched response-generation interventions; full cohort before target scoring.

Three fixed solvers share admitted observations and native providers. UQIS has
no such raster reconstruction stage and is an explicitly unchanged control,
not evidence that this intervention improves all query types.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_study_adapters import SYSTEM, read, load, readout_path
from radio_gs.querying.response_stage_adapters import (
    lerf_problem, nvos_problem, scannet_problem, finish_response)
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.run_tmm_support_agreement import PredictionMemory, RAM_LIMIT
from scripts.research.run_tmm_response_study import score_lerf, score_nvos, score_scannet, score_uqis
from scripts.research.tmm_transient_runtime import run_with_gpu_retry
from scripts.research.tmm_ram_predictions import RamPredictionBank, public_query_files

ARMS=('diagonal','rendered_squared','rendered_huber')
EXPERIMENTAL_INPUT='frozen pre-fusion providers for source families; unchanged complete UQIS outputs'


@torch.no_grad()
def predict(job,run,memory):
    started=time.monotonic();paths=[job['scene']]
    report=dict(job=job,ground_truth_read=False,selection_heads_frozen=True,
        experimental_input=EXPERIMENTAL_INPUT,no_new_image_teacher_calls=True,
        prediction_retention='transient RAM; replay by recomputing from hashed inputs')
    if job['benchmark']=='uqis':
        from radio_gs.querying.registered_observation_query import public_queries
        public,_=public_queries(Path(job['queries']),job['name'])
        qpath=SYSTEM/'uqis'/job['name']/'query/registered/runs'/f"{job['name']}.json"
        details=read(qpath)['query_details'];paths.extend([qpath,*public_query_files(job['queries'])])
        if {q['query_id'] for q in public}!={d['query_id'] for d in details}:
            raise ValueError('public query identity changed')
        source=SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        columns=[];ids=[]
        for detail in details:
            qid=detail['query_id'];p=source/'predictions'/f'{qid}.npy'
            paths.extend([p,source/'selections'/f'{qid}.npy'])
            ids.append(qid);columns.append(np.load(p))
        array=torch.from_numpy(np.stack(columns,1));arrays={a:array for a in ARMS}
        report.update(columns=ids,intervention_applied=False,
            compatibility_only=True,reason='no raster response reconstruction in retained membership reader',
            source_rgb_read=False,source_depth_read=False,source_masks_read=False)
    else:
        path=readout_path(job);paths.append(path);data=load(path)
        factory={'lerf':lerf_problem,'nvos':nvos_problem,'scannet':scannet_problem}[job['benchmark']]
        print('assembling',job['benchmark'],job['name'],flush=True)
        problem,contract,evidence_paths=factory(job,data);paths.extend(evidence_paths)
        report.update(columns=data['texts'],intervention_applied=True,stage_contract=contract,arms={})
        arrays={}
        for arm in ARMS:
            progress=lambda iteration:print('solving',job['benchmark'],job['name'],arm,iteration,flush=True)
            value=(problem.diagonal() if arm=='diagonal' else problem.rendered(steps=120,
                residual_clip=None if arm=='rendered_squared' else .25,progress=progress))
            # Source losses diagnose fitting, never choose a per-scene winner.
            report['arms'][arm]=dict(source_squared_objective=problem.objective(value),
                source_huber_objective=problem.objective(value,.25),
                supported_entries=int((problem.mass>1e-8).sum()))
            arrays[arm]=finish_response(job,data,problem,value).cpu()
        original=(data['category_logits'].sigmoid() if job['benchmark']=='scannet' else data['scores'])
        reconstructed=arrays[contract['retained_solver']]
        error=float((reconstructed-original).abs().max())
        crossings=int(((reconstructed>.5)!=(original>.5)).sum())
        tolerance=1e-4 if job['benchmark']=='nvos' else 2e-6
        mean_error=float((reconstructed-original).abs().mean())
        report['compatibility']=dict(original_solver=contract['retained_solver'],
            maximum_response_error=error,half_threshold_crossings=crossings,
            mean_response_error=mean_error,tolerance=tolerance,
            passed=error<=tolerance and mean_error<=1e-6)
        if not report['compatibility']['passed']:
            run.json('failed_compatibility.json',report)
            raise AssertionError(f"pre-response replay failed {job['name']}: {error}")
    report.update(input_hashes={str(p):sha(p) for p in sorted(set(map(str,paths)))},
        seconds=time.monotonic()-started,
        prediction_arrays={a:memory.add(run,a,v) for a,v in arrays.items()})
    run.json('prediction.json',report)
    print('predicted',job['benchmark'],job['name'],'seconds',round(report['seconds']),
        'RAM MiB',round(memory.bytes/2**20),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='stage_reconstruction_v4')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    jobs=sorted(read('configs/pipeline/system_v10.json')['jobs'],
        key=lambda j:({'uqis':0,'nvos':1,'scannet':2,'lerf':3}[j['benchmark']],j['name']))
    if len(jobs)!=31:raise ValueError('full cohort changed')
    root=BoundedExperiment(STUDY,args.run,POLICY,resume=args.resume);root.preflight(20*2**20)
    if (root.root/'summary.json').exists():
        print('run already complete',root.root,flush=True);return
    sources=['radio_gs/querying/response_reconstruction.py','radio_gs/querying/response_stage_adapters.py',
        'radio_gs/querying/local_element_fusion.py','radio_gs/querying/typed_evidence.py',
        'radio_gs/querying/native_observation_prior.py','radio_gs/models/native_query_encoders.py',
        'scripts/research/run_tmm_stage_reconstruction.py','scripts/research/run_tmm_support_agreement.py',
        'scripts/research/run_tmm_response_study.py','scripts/research/fit_consolidated_response.py',
        'radio_gs/evaluation/lerf_readouts.py','configs/pipeline/system_v10.json',
        'scripts/research/replay_tmm_stage_inputs.py','radio_gs/querying/observation_reliability.py',
        'radio_gs/querying/observation_readout.py','radio_gs/models/pretrained_mask_adapter.py',
        'scripts/research/tmm_transient_runtime.py','scripts/research/tmm_ram_predictions.py']
    if args.resume:
        original=read(root.root/'manifest.json')
        if original['jobs']!=jobs or original['source_hashes']!={p:sha(p) for p in sources}:
            raise ValueError('cannot resume after source/protocol changes')
        memory=RamPredictionBank(original['prediction_ram_directory'])
    else:memory=RamPredictionBank()
    manifest=dict(jobs=jobs,arms=ARMS,source_hashes={p:sha(p) for p in sources},
        mechanism='admitted observation reconstruction at the original response stage',
        changed_variable='diagonal transport versus rendered squared fit; separate rendered Huber control',
        experimental_input=EXPERIMENTAL_INPUT,all_existing_admission_rules_frozen=True,
        uqis_intervention_applied=False,uqis_compatibility_only=True,
        steps=120,step_size=.95,huber_residual_clip=.25,parameters_inherited_from_v10_nvos=True,
        additional_supervision=False,independent_test=False,default_changed=False,
        no_new_image_teacher_calls=True,frozen_auxiliary_category_text_encoder_replayed=True,
        input_contract='source-assisted LERF/NVOS/ScanNet; unchanged map-only UQIS',
        prediction_storage='private volatile tmpfs bank; survives worker exit, removed after scoring',
        prediction_ram_directory=str(memory.root),transient_prediction_bytes_limit=RAM_LIMIT,
        previous_unsealed_failure=str(STUDY/'stage_reconstruction_v1_nvos_fern/failed_compatibility.json'),
        previous_resource_failure='stage_reconstruction_v2: CUDA0 free memory fell during LERF assembly; no prediction seal/GT scoring',
        resource_policy='choose visible GPU with >=16GiB free; retain completed CPU arrays and retry only CUDA-OOM job',
        source_only_numeric_audit=str(STUDY/'stage_input_numeric_preflight/summary.json'),
        retained_numeric_tolerance=dict(nvos=1e-4,lerf=2e-6,scannet=2e-6,mean=1e-6),
        tolerance_reason='old and shared NVOS solvers both differ from retained artifacts by up to 4.68e-5; fresh old/shared <=1.08e-6 without target scoring')
    if not args.resume:
        root.json('manifest.json',manifest)
        root.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    runs=[];reports=[]
    def execute(phase,job,operation,needs_gpu=True):
        def event(detail):
            record=dict(phase=phase,job=job['name'],family=job['benchmark'],
                completed_prediction_jobs=len(reports),transient_prediction_bytes=memory.bytes,**detail)
            root.json('progress.json',record,replace=True)
            if detail['state']=='retry_after_cuda_oom':
                root.json(f"{phase}_{job['benchmark']}_{job['name']}_oom_{detail['attempt']:02d}.json",record)
            print('runtime',phase,job['name'],detail['state'],detail.get('cuda_device'),flush=True)
        return run_with_gpu_retry(operation,event=event) if needs_gpu else operation()
    for job in jobs:
        name=f"{args.run}_{job['benchmark']}_{job['name']}"
        run=BoundedExperiment(STUDY,name,POLICY,resume=args.resume and (STUDY/name).exists())
        if (run.root/'prediction.json').exists():
            report=read(run.root/'prediction.json')
            if report['job']!=job:raise ValueError('retained job contract differs')
            for arm in ARMS:
                memory.read(run,arm,len(report['columns']))
                entry=memory.entries[memory.key(run,arm)]
                if entry['sha256']!=report['prediction_arrays'][arm]['sha256']:
                    raise ValueError('retained prediction differs from report')
            reports.append(report);runs.append(run);continue
        if args.resume:memory.discard_unsealed(run)
        reports.append(execute('predict',job,lambda:predict(job,run,memory),job['benchmark']!='uqis'));runs.append(run)
        gc.collect();torch.cuda.empty_cache()
    if not (root.root/'predictions_complete.json').exists():root.json('predictions_complete.json',dict(jobs=31,all_five_present=True,no_target_gt_read=True,
        reports={str(r.root/'prediction.json'):sha(r.root/'prediction.json') for r in runs},
        transient_prediction_bytes=memory.bytes,hashes_are_raw_C_order_float32=True))
    seal=read(root.root/'predictions_complete.json')
    if any(sha(p)!=h for p,h in seal['reports'].items()):raise ValueError('prediction report seal changed')
    for run,report in zip(runs,reports):
        for p,h in report['input_hashes'].items():
            if sha(p)!=h:raise ValueError('stage input changed after predictions')
        for arm in ARMS:memory.read(run,arm,len(report['columns']))
    if not (root.root/'evaluation_manifest.json').exists():root.json('evaluation_manifest.json',dict(prediction_seal_sha256=sha(root.root/'predictions_complete.json'),
        all_inputs_and_predictions_checked=True,
        scoring_source_hashes={p:sha(p) for p in ['scripts/research/run_tmm_response_study.py',
            'radio_gs/evaluation/lerf_readouts.py','radio_gs/benchmarks/uqis/evaluate.py','radio_gs/scannet_constants.py']}))
    functions={'lerf':score_lerf,'nvos':score_nvos,'scannet':score_scannet};results=[]
    for job,run,report in zip(jobs,runs,reports):
        if job['benchmark']=='uqis':continue
        if (run.root/'evaluation.json').exists():
            results.append(dict(job=job,**read(run.root/'evaluation.json')));continue
        result=execute('score',job,lambda:functions[job['benchmark']](job,run,report,
            response_loader=memory.read,arms=ARMS),job['benchmark'] in ['lerf','nvos'])
        run.json('evaluation.json',result);results.append(dict(job=job,**result))
        print('scored',job['benchmark'],job['name'],result['metrics'],flush=True)
        gc.collect();torch.cuda.empty_cache()
    indices=[i for i,j in enumerate(jobs) if j['benchmark']=='uqis']
    uqis=score_uqis([jobs[i] for i in indices],[runs[i] for i in indices],[reports[i] for i in indices],root,
        response_loader=memory.read,arms=ARMS)
    macro={}
    for family in functions:
        group=[r for r in results if r['job']['benchmark']==family]
        macro[family]={a:{k:sum(r['metrics'][a][k] for r in group)/len(group) for k in group[0]['metrics'][a]}
            for a in ('baseline',)+ARMS}
    macro['uqis']=uqis['metrics']
    root.json('summary.json',dict(complete=True,macro=macro,rows=results,uqis=uqis,
        all_prediction_jobs=31,independent_test=False,default_changed=False,selection_heads_frozen=True,
        actual_new_storage_bytes=tree_bytes(STUDY),new_end_to_end_system=False,
        experimental_input=EXPERIMENTAL_INPUT,uqis_compatibility_only=True,
        transient_prediction_bytes=memory.bytes,prediction_replay_requires_recomputation=True))
    print(json.dumps(macro,indent=2),flush=True)
    memory.close()
    root.json('ram_released.json',dict(complete=True,removed_own_tmpfs_bank=True,
        prediction_arrays_retained_on_persistent_disk=False))


if __name__=='__main__':main()
