"""Full-cohort source identity experiment on frozen evidence and renderer."""
import argparse,json,os,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
SCENES=['figurines','ramen','teatime','waldo_kitchen']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--arms',nargs='+',default=['conditioned','forced_split'],choices=['conditioned','forced_split']);p.add_argument('--gpus',nargs='+',default=['0','1','2','3','4','5']);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);jobs=[]
    for scene in SCENES:
        for arm in a.arms:
            dest=out/arm/scene;dest.mkdir(parents=True,exist_ok=True)
            commands=[
                [PYTHON,'scripts/research/condition_source_identity.py','--inputs',f'output/research_consolidation/source_run/{scene}/inputs.json','--identity-record',f'output/scope_evidence_control/identity/{scene}/evidence/identity.json','--output',str(dest/'evidence')]+(['--force-two-groups'] if arm=='forced_split' else []),
                [PYTHON,'scripts/research/decode_expanded_extent_next72.py','--inputs',str(dest/'evidence/inputs.json'),'--key',f'lerf_{scene}','--output-base',str(dest/'sets'),'--mode','admission','--scope-initialization','source','--pixel-loss','balanced','--support-prior','footprint','--raster-scale','2','--steps','240','--heldout-stride','5'],
                [PYTHON,'scripts/research/evaluate_consolidated_query.py','--scene',f'configs/consolidated/{scene}.scene.json','--readout',str(dest/'sets'/f'lerf_{scene}'/'typed_readout.pt'),'--output',str(dest/'evaluation')]]
            jobs.append(dict(scene=scene,arm=arm,directory=str(dest),commands=commands))
    (out/'plan.json').write_text(json.dumps(dict(jobs=jobs,heldout_stride=5,benchmark_labels_in_inference=False),indent=2)+'\n')
    def worker(gpu,assigned):
        for job in assigned:
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2')
            with (Path(job['directory'])/'run.log').open('w') as log:
                for command in job['commands']:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(gpu,job['scene'],job['arm'],'completed',flush=True)
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::len(a.gpus)]) for i,gpu in enumerate(a.gpus)]
        for f in futures:f.result()


if __name__=='__main__':main()
