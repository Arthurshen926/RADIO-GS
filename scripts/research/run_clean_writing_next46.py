import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def run(job):
    gpu,scene=job;key='lerf_'+scene;root=Path('output/clean_writing_next46');root.mkdir(parents=True,exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    commands=[['scripts/research/build_clean_writing_next46.py','--key',key]]+[['scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/mode),'--readout','anchored','--pipeline'] for mode in ['alpha','gate','robust']]
    rows=[]
    with (root/(key+'.log')).open('w') as log:
        for cmd in commands:
            rc=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python']+cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(key=key,command=cmd,exit_code=rc));print(key,cmd[0],rc,flush=True)
            if rc:break
    return rows

with ThreadPoolExecutor(4) as pool:rows=sum(pool.map(run,enumerate(['figurines','ramen','teatime','waldo_kitchen'])),[])
Path('output/clean_writing_next46/execution.json').write_text(json.dumps(rows,indent=2)+'\n')
