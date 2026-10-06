import os,subprocess,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path('output/native_proposals_next69');root.mkdir(parents=True,exist_ok=True)
def run(job):
 gpu,scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4')
 cmd=['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/build_native_proposals_next69.py','--key',key]
 with (root/(scene+'_evidence.log')).open('w') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 print(scene,rc,flush=True);return dict(scene=scene,exit_code=rc,command=cmd)
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,enumerate(['figurines','ramen','teatime','waldo_kitchen'])))
(root/'evidence_execution.json').write_text(json.dumps(rows,indent=2)+'\n')
