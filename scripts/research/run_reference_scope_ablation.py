"""Eight-query fixed reference-only response ablation; six authorized GPUs."""
import subprocess,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
scenes=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']
def worker(gpu):
    for s in scenes[gpu::6]:
        out=Path('output/reference_scope_stages')/s/'reference_only';out.mkdir(parents=True,exist_ok=True);base=f'configs/consolidated/nvos/{s}';py='/root/miniconda3/envs/cybersim_agent/bin/python'
        commands=[[py,'scripts/research/fit_reference_instance.py','--scene',base+'.scene.json','--query',base+'.query.json','--tracks',f'output/reference_integration/fresh_tracks/{s}/tracks.pt','--reference-only','--output',str(out)],[py,'scripts/research/evaluate_reference_instance.py','--scene',base+'.scene.json','--evaluation',base+'.evaluation.json','--field',str(out/'field.pt'),'--output',str(out/'evaluation')]]
        with (out/'run.log').open('w') as log:
            for cmd in commands:subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(s,'reference-only complete',flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(worker,range(6)))
