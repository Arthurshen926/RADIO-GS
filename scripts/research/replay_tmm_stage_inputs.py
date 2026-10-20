"""Source-only replay of response-stage inputs and the pre-extraction solver."""
import argparse
import gc
import json
from pathlib import Path
import torch

from radio_gs.querying.response_study_adapters import read,load,readout_path
from radio_gs.querying.response_stage_adapters import nvos_problem,scannet_problem,finish_response
from scripts.research.replay_tmm_response import STUDY,POLICY,sha
from radio_gs.query_field.bounded_experiment import BoundedExperiment


@torch.no_grad()
def legacy_loop(problem):
    # Literal old loop on identical admitted terms; no new permissions/inputs.
    value=problem.initial.clone();mass=torch.zeros_like(value)
    for op,adj,target,known in problem.terms:mass+=torch.sparse.mm(adj,known)
    for _ in range(120):
        gradient=torch.zeros_like(value)
        for op,adj,target,known in problem.terms:
            gradient+=torch.sparse.mm(adj,(target-torch.sparse.mm(op,value)).clamp(-.25,.25)*known)
        value.add_(.95*gradient/mass.clamp_min(1e-8)).clamp_(0,1)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='stage_input_numeric_replay')
    args=parser.parse_args();torch.set_num_threads(4)
    run=BoundedExperiment(STUDY,args.run,POLICY)
    sources=['scripts/research/replay_tmm_stage_inputs.py','radio_gs/querying/response_stage_adapters.py',
        'radio_gs/querying/response_reconstruction.py','radio_gs/querying/local_element_fusion.py']
    run.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    rows=[]
    for job in read('configs/pipeline/system_v10.json')['jobs']:
        if job['benchmark'] not in ['nvos','scannet']:continue
        data=load(readout_path(job))
        factory=nvos_problem if job['benchmark']=='nvos' else scannet_problem
        problem,contract,paths=factory(job,data)
        expected=data['category_logits'].sigmoid() if job['benchmark']=='scannet' else data['scores']
        value=problem.rendered() if job['benchmark']=='nvos' else problem.diagonal()
        value=finish_response(job,data,problem,value).cpu();delta=(value-expected).abs()
        row=dict(family=job['benchmark'],name=job['name'],max_error=float(delta.max()),
            mean_error=float(delta.mean()),threshold_crossings=int(((value>.5)!=(expected>.5)).sum()))
        if job['benchmark']=='nvos':
            old=finish_response(job,data,problem,legacy_loop(problem)).cpu()
            row.update(fresh_old_vs_shared_max_error=float((old-value).abs().max()),
                old_vs_retained_max_error=float((old-expected).abs().max()))
        run.json(job['benchmark']+'_'+job['name']+'.json',row);rows.append(row)
        print(json.dumps(row),flush=True)
        del data,problem,value;gc.collect();torch.cuda.empty_cache()
    run.json('summary.json',dict(rows=rows,ground_truth_read=False))


if __name__=='__main__':main()
