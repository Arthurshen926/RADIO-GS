import os,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
def run(job):
    gpu,scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    root=Path('output/set_decoder_next48');root.mkdir(parents=True,exist_ok=True)
    commands=[[PYTHON,'scripts/research/decode_selected_sets_next48.py','--key',key,'--arm','joint'],[PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/'joint'),'--readout','anchored','--pipeline','--calibrated','--element-results',str(root/'joint'/key/'elements.pt')]];rows=[]
    with (root/(scene+'_joint.log')).open('w') as log:
        for c in commands:
            rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(scene=scene,command=c,exit_code=rc));print(scene,c[1],rc,flush=True)
            if rc:break
    return rows
with ThreadPoolExecutor(4) as pool:rows=sum(pool.map(run,enumerate(['figurines','ramen','teatime','waldo_kitchen'])),[])
Path('output/set_decoder_next48/execution_joint.json').write_text(json.dumps(rows,indent=2)+'\n')
