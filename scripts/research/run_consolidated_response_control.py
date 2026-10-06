"""All four LERF scenes and all explicit text queries, no query-specific routing."""
import os,subprocess,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
scenes=['figurines','ramen','teatime','waldo_kitchen'];PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def worker(job):
    gpu,s=job;source=Path('output/conditioned_identity/full_source')/s;out=Path('output/consolidated_response_control')/s;out.mkdir(parents=True,exist_ok=True);commands=[[PY,'scripts/research/fit_consolidated_response.py','--inputs',str(source/'identity_evidence/inputs.json'),'--readout',str(source/'sets'/('lerf_'+s)/'typed_readout.pt'),'--output',str(out)]]
    for arm in ['native_protected','source_complement']:commands.append([PY,'scripts/research/evaluate_consolidated_query.py','--scene',f'configs/consolidated/{s}.scene.json','--readout',str(out/arm/'typed_readout.pt'),'--output',str(out/arm/'evaluation')])
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/'run.log').open('w') as log:
        for cmd in commands:subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(s,'complete',flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,enumerate(scenes)))
