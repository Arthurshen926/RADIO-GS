"""Full-cohort support-admission study with sealed, transient RAM predictions.

Two controls share evidence and per-query signed correction budgets. Preserve
code, input identities, prediction hashes and metrics; reproduce arrays by
rerunning inference instead of accumulating another 1.3 GB on disk.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.support_agreement import SupportAgreementRefiner
from radio_gs.querying.response_study_adapters import (
    SYSTEM, read, load, readout_path, source_observations, uqis_inputs, uqis_observations)
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.run_tmm_response_study import score_lerf, score_nvos, score_scannet, score_uqis

ARMS=('scope','matched')
RAM_LIMIT=2**31


def array_sha(array):
    if not array.flags.c_contiguous:raise ValueError('hash expects contiguous float32 matrix')
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


class PredictionMemory:
    def __init__(self, limit=RAM_LIMIT):
        self.values={};self.hashes={};self.bytes=0;self.limit=limit

    def add(self, run, arm, value):
        key=(str(run.root),arm)
        if key in self.values:raise ValueError('duplicate transient prediction')
        array=np.ascontiguousarray(value.cpu().numpy(),dtype=np.float32)
        if self.bytes+array.nbytes>self.limit:raise ValueError('transient prediction RAM budget exceeded')
        array.setflags(write=False)
        self.values[key]=array;self.hashes[key]=array_sha(array);self.bytes+=array.nbytes
        return dict(shape=list(array.shape),dtype='float32',sha256=self.hashes[key],bytes=array.nbytes)

    def read(self, run, arm, count):
        key=(str(run.root),arm);array=self.values[key]
        if array.shape[1]!=count or array_sha(array)!=self.hashes[key]:
            raise ValueError('sealed transient prediction changed')
        return torch.from_numpy(array.copy())


def source_guide(data, family):
    if family=='scannet':
        guide=torch.zeros_like(data['category_logits'],dtype=torch.bool)
        guide.scatter_(1,data['category_logits'].argmax(1)[:,None],True)
        return guide & data['semantic_observed'][:,None]
    if data['semantic_geometry']!=data['selection_geometry']:
        raise ValueError('extent and response carriers differ; no implicit remapping')
    guide=data['selection_scores']>.5
    observed=data.get('selection_observed_by_query',data['selection_observed'][:,None])
    return guide & observed


def infer(prior,guide,observations):
    refiner=SupportAgreementRefiner(prior,guide)
    for partial,_ in observations:refiner.add(partial)
    result=refiner.finish()
    for key in ['positive_step_sum','negative_step_sum']:
        if not np.allclose(result['scope'][1][key],result['matched'][1][key],atol=1e-5,rtol=1e-6):
            raise AssertionError('signed update budgets differ')
    return result


@torch.no_grad()
def predict(job, run, memory):
    started=time.monotonic();inputs={}
    report=dict(job=job,ground_truth_read=False,selection_heads_frozen=True,
        no_new_teacher_calls=True,experimental_input='retained frozen v10 response and extent',
        prediction_retention='transient RAM, hashed before scoring, recompute for replay')
    if job['benchmark']=='uqis':
        membership,visibility,frames,details=uqis_inputs(job)
        from radio_gs.querying.registered_observation_query import public_queries
        public,_=public_queries(Path(job['queries']),job['name'])
        if {d['query_id'] for d in details}!={q['query_id'] for q in public}:
            raise ValueError('frozen query identity differs from legal inputs')
        source=SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        columns={arm:[] for arm in ARMS};rows=[]
        for detail in details:
            qid=detail['query_id'];p=source/'predictions'/f'{qid}.npy';s=source/'selections'/f'{qid}.npy'
            inputs[str(p)]=sha(p);inputs[str(s)]=sha(s)
            prior=torch.from_numpy(np.load(p)[:,None]).cuda()
            guide=torch.from_numpy(np.load(s)[:,None]).cuda()
            result=infer(prior,guide,uqis_observations(membership,visibility,frames,detail))
            row=dict(query_id=qid,modality=detail['modality'],arms={})
            for arm,(value,stats) in result.items():
                columns[arm].append(value.cpu());row['arms'][arm]=stats
            rows.append(row)
        report.update(columns=[r['query_id'] for r in rows],queries=rows,
            guide_contract='frozen method-selected complete membership; no GT',
            source_rgb_read=False,source_depth_read=False,source_masks_read=False)
        arrays={a:torch.cat(v,1) for a,v in columns.items()}
    else:
        path=readout_path(job);inputs[str(path)]=sha(path);data=load(path)
        prior=(data['category_logits'].sigmoid() if job['benchmark']=='scannet' else data['scores']).cuda()
        guide=source_guide(data,job['benchmark']).cuda()
        result=infer(prior,guide,source_observations(job,data,prior))
        arrays={};report.update(columns=data['texts'],arms={},
            guide_contract=('frozen full-menu native category argmax' if job['benchmark']=='scannet'
                else 'frozen independent selected-set score >0.5 and original observed domain'))
        for arm,(value,stats) in result.items():
            value[~data['semantic_observed'].cuda()]=prior[~data['semantic_observed'].cuda()]
            arrays[arm]=value;report['arms'][arm]=stats
    inputs[str(Path(job['scene']))]=sha(job['scene'])
    report.update(input_hashes=inputs,seconds=time.monotonic()-started,
        prediction_arrays={a:memory.add(run,a,v) for a,v in arrays.items()})
    run.json('prediction.json',report)
    print('predicted',job['benchmark'],job['name'],'RAM MiB',round(memory.bytes/2**20),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='post_response_v3')
    args=parser.parse_args();torch.set_num_threads(4)
    jobs=sorted(read('configs/pipeline/system_v10.json')['jobs'],
        key=lambda j:({'lerf':0,'nvos':1,'scannet':2,'uqis':3}[j['benchmark']],j['name']))
    if len(jobs)!=31:raise ValueError('declared full cohort changed')
    root=BoundedExperiment(STUDY,args.run,POLICY);root.preflight(20*2**20)
    sources=['radio_gs/querying/support_agreement.py','radio_gs/querying/response_refinement.py',
        'radio_gs/querying/response_permissions.py','radio_gs/querying/response_study_adapters.py',
        'scripts/research/run_tmm_support_agreement.py','scripts/research/run_tmm_response_study.py',
        'scripts/research/fit_consolidated_response.py','radio_gs/evaluation/lerf_readouts.py',
        'configs/pipeline/system_v10.json']
    root.json('manifest.json',dict(jobs=jobs,arms=ARMS,source_hashes={p:sha(p) for p in sources},
        mechanism='whole-observation agreement with frozen extent, separate positive/exclusion confidence',
        control='same per-query positive/negative pre-clipping L1 step mass, uniformly attenuated',
        changed_variable='spatial allocation of a matched signed correction budget',
        prediction_storage='transient RAM; no durable response tensors; replay requires recomputation',
        transient_prediction_bytes_limit=RAM_LIMIT,selection_heads_frozen=True,
        input_contract='source-assisted LERF/NVOS/ScanNet; map-only UQIS',
        independent_test=False,default_changed=False,no_new_supervision=True))
    root.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    memory=PredictionMemory();runs=[];reports=[]
    for job in jobs:
        run=BoundedExperiment(STUDY,f"{args.run}_{job['benchmark']}_{job['name']}",POLICY)
        reports.append(predict(job,run,memory));runs.append(run)
        gc.collect();torch.cuda.empty_cache()
    root.json('predictions_complete.json',dict(jobs=31,all_five_present=True,no_target_gt_read=True,
        reports={str(r.root/'prediction.json'):sha(r.root/'prediction.json') for r in runs},
        transient_prediction_bytes=memory.bytes,hashes_are_raw_C_order_float32=True))
    for run,report in zip(runs,reports):
        for p,h in report['input_hashes'].items():
            if sha(p)!=h:raise ValueError('frozen input changed after prediction')
        for arm in ARMS:memory.read(run,arm,len(report['columns']))
    root.json('evaluation_manifest.json',dict(prediction_seal_sha256=sha(root.root/'predictions_complete.json'),
        scoring_source_hashes={p:sha(p) for p in ['scripts/research/run_tmm_response_study.py',
            'radio_gs/evaluation/lerf_readouts.py','radio_gs/benchmarks/uqis/evaluate.py','radio_gs/scannet_constants.py']},
        all_inputs_and_predictions_checked=True))
    functions={'lerf':score_lerf,'nvos':score_nvos,'scannet':score_scannet};results=[]
    for job,run,report in zip(jobs,runs,reports):
        if job['benchmark']=='uqis':continue
        result=functions[job['benchmark']](job,run,report,response_loader=memory.read,arms=ARMS)
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
        transient_prediction_bytes=memory.bytes,prediction_replay_requires_recomputation=True))
    print(json.dumps(macro,indent=2),flush=True)


if __name__=='__main__':main()
