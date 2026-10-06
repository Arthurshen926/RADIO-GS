import os,subprocess,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path('output/zoom_grounding_next70');root.mkdir(parents=True,exist_ok=True)
def run(job):
 gpu,scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.:/root/external/sam3',OMP_NUM_THREADS='4')
 commands=[['/root/miniconda3/envs/sam3/bin/python','scripts/research/build_zoom_grounding_next70.py','--key',key],['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/qualify_zoom_grounding_next70.py','--key',key]];rows=[]
 with (root/(scene+'_evidence.log')).open('w') as f:
  for cmd in commands:
   rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode;rows.append(dict(scene=scene,exit_code=rc,command=cmd));print(scene,cmd[1],rc,flush=True)
   if rc:break
 return rows

with ThreadPoolExecutor(4) as pool:rows=sum(pool.map(run,enumerate(['figurines','ramen','teatime','waldo_kitchen'])),[])
(root/'evidence_execution.json').write_text(json.dumps(rows,indent=2)+'\n')
