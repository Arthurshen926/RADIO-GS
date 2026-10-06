"""Two fixed positive-evidence candidates across all eight NVOS queries."""
import os,subprocess,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
scenes=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']
jobs=[(arm,s) for arm in ['medial','raw'] for s in scenes]
def worker(gpu):
    for arm,s in jobs[gpu::6]:
        dest=Path('output/reference_evidence_coverage')/'candidates'/arm/s;dest.mkdir(parents=True,exist_ok=True);base=f'configs/consolidated/nvos/{s}';py='/root/miniconda3/envs/cybersim_agent/bin/python'
        commands=[[py,'scripts/research/fit_reference_instance.py','--scene',base+'.scene.json','--query',base+'.query.json','--tracks',f'output/reference_integration/fresh_tracks/{s}/tracks.pt','--positive-core',arm,'--output',str(dest)],[py,'scripts/research/evaluate_reference_instance.py','--scene',base+'.scene.json','--evaluation',base+'.evaluation.json','--field',str(dest/'field.pt'),'--output',str(dest/'evaluation')]]
        (dest/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        with (dest/'run.log').open('w') as log:
            for cmd in commands:subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(arm,s,'done',flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(worker,range(6)))
