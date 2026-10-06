"""Three-scene dense Gaussian / structured surface pilot, fixed recipe."""
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def run(job):
    gpu,key=job;root=Path('output/dense_surface_next43');root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    commands=[['scripts/research/build_dense_surface_next43.py','--key',key]]
    commands += [['scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/mode),'--readout','anchored','--pipeline'] for mode in ['gaussian','gaussian_gate','surface','surface_robust']]
    rows=[]
    with (root/(key+'.log')).open('w') as log:
        for cmd in commands:
            rc=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python']+cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            rows.append(dict(key=key,command=cmd,exit_code=rc));print(key,cmd[0],rc,flush=True)
            if rc:break
    return rows


with ThreadPoolExecutor(3) as pool:
    rows=sum(pool.map(run,[(0,'lerf_figurines'),(1,'lerf_ramen'),(2,'scannet_scene0000_00')]),[])
Path('output/dense_surface_next43/execution.json').write_text(json.dumps(rows,indent=2)+'\n')
