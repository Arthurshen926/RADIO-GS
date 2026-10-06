"""Full-query dual-output reconstructions from independent referent evidence."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY='/root/miniconda3/envs/cybersim_agent/bin/python'
p=argparse.ArgumentParser();p.add_argument('--root',default='output/source_referent_verification');p.add_argument('--gpus',type=int,nargs='+',default=list(range(6)))
p.add_argument('--policies',nargs='+',choices=['veto','affirmative','contradiction'],default=['veto','affirmative']);a=p.parse_args()


def worker(gpu,arm,cohort,scene):
    dest=Path(a.root)/arm/cohort/scene;inputs=dest/'evidence/inputs.json'
    commands=[
        [PY,'scripts/research/decode_expanded_extent_next72.py','--inputs',str(inputs),'--key','lerf_'+scene,
         '--output-base',str(dest/'sets'),'--mode','admission','--scope-initialization','source',
         '--pixel-loss','balanced','--support-prior','footprint','--raster-scale','2','--steps','240'],
        [PY,'scripts/research/fit_consolidated_response.py','--inputs',str(inputs),
         '--readout',str(dest/'sets'/('lerf_'+scene)/'typed_readout.pt'),'--output',str(dest/'response'),
         '--policies','native_protected','--solver','backprojection'],
        [PY,'scripts/research/evaluate_consolidated_query.py','--scene',f'configs/consolidated/{scene}.scene.json',
         '--readout',str(dest/'response/native_protected/typed_readout.pt'),'--output',str(dest/'evaluation')]]
    (dest/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (dest/'run.log').open('w') as log:
        for cmd in commands:subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(arm,cohort,scene,'complete',flush=True)


jobs=[(arm,cohort,scene) for arm in a.policies for cohort in ['original','tiled'] for scene in ['figurines','ramen','teatime','waldo_kitchen']]
lanes=[[(gpu,*job) for job in jobs[i::len(a.gpus)]] for i,gpu in enumerate(a.gpus)]
def lane(jobs):
    for job in jobs:worker(*job)
with ThreadPoolExecutor(max_workers=len(lanes)) as pool:list(pool.map(lane,lanes))
