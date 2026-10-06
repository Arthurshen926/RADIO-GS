import os,subprocess,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def run(job):
 gpu,scene=job;key='lerf_'+scene;root=Path('output/latent_transport_next53');root.mkdir(exist_ok=True,parents=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4');rows=[]
 commands=[[PY,'scripts/research/transport_latent_next53.py','--key',key]]+[[PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/mode),'--readout','anchored','--pipeline','--calibrated'] for mode in ['direct','transport']]
 with (root/(scene+'.log')).open('w') as log:
  for cmd in commands:
   rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));print(scene,cmd[1],rc,flush=True)
   if rc:break
 return rows
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,[(4,'figurines'),(5,'ramen')]))
Path('output/latent_transport_next53/execution.json').write_text(json.dumps(rows,indent=2)+'\n')
