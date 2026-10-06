"""Matched evidence comparison: fusion, subset fit, and conflict abstention.

Every fifth source view is excluded from evidence fusion and fitting. Maps and
per-frame detector caches remain fixed; held-out agreement is teacher agreement,
not independent truth or unseen-scene generalization. No tracked evidence is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
SCENES=['figurines','ramen','teatime','waldo_kitchen']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--gpus',nargs='+',default=['0','1','2','3','4','5'])
    p.add_argument('--arms',nargs='+',choices=['fusion','fit','abstention','identity','identity_veto'],default=['fusion','fit','abstention','identity'])
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for scene in SCENES:
        for arm in a.arms:
            flags={'fusion':['--source-fusion-only'],'fit':[],'abstention':['--conflict-abstention'],'identity':[],'identity_veto':[]}[arm]
            dest=out/arm/scene;dest.mkdir(parents=True,exist_ok=True)
            inputs=f'output/research_consolidation/source_run/{scene}/inputs.json'
            prepare=[]
            if arm=='identity':
                prepare=[[PYTHON,'scripts/research/filter_source_identity.py','--inputs',inputs,'--output',str(dest/'evidence')]]
                inputs=str(dest/'evidence/inputs.json')
            if arm=='identity_veto':
                prepare=[[PYTHON,'scripts/research/replay_identity_admission.py','--inputs',inputs,'--identity-record',str(out/'identity'/scene/'evidence/identity.json'),'--output',str(dest/'evidence')]]
                inputs=str(dest/'evidence/inputs.json')
            fit=[PYTHON,'scripts/research/decode_expanded_extent_next72.py',
                 '--inputs',inputs,
                 '--key',f'lerf_{scene}','--output-base',str(dest/'sets'),
                 '--mode','admission','--scope-initialization','source','--pixel-loss','balanced',
                 '--support-prior','footprint','--raster-scale','2','--steps','240',
                 '--heldout-stride','5',*flags]
            score=[PYTHON,'scripts/research/evaluate_consolidated_query.py',
                   '--scene',f'configs/consolidated/{scene}.scene.json',
                   '--readout',str(dest/'sets'/f'lerf_{scene}'/'typed_readout.pt'),
                   '--output',str(dest/'evaluation')]
            jobs.append(dict(scene=scene,arm=arm,directory=str(dest),commands=prepare+[fit,score]))
    (out/'plan.json').write_text(json.dumps(dict(jobs=jobs,heldout_stride=5,
        selection_by_test_score=False,new_detector_calls=0,unseen_scene=False),indent=2)+'\n')
    def worker(gpu,assigned):
        for job in assigned:
            dest=Path(job['directory'])
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4')
            with (dest/'run.log').open('w') as log:
                for command in job['commands']:
                    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(gpu,job['scene'],job['arm'],'completed',flush=True)
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::len(a.gpus)]) for i,gpu in enumerate(a.gpus)]
        for future in futures:future.result()


if __name__=='__main__':main()
