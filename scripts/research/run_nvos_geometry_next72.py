"""Scene-parallel matched geometry/video mechanism controls, full eight queries."""
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PY='/root/miniconda3/envs/cybersim_agent/bin/python';OUT=Path('output/nvos_transfer_next72')
def run(item):
 gpu,scenes=item;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
 for scene in scenes:
  OUT.mkdir(parents=True,exist_ok=True);cmds=[]
  if scene!='horns_left':cmds.append([PY,'scripts/research/reconstruct_nvos_carrier_next72.py','--scene',scene,'--arm','adaptive_allrgb'])
  for carrier in ['old','new']:cmds.append([PY,'scripts/research/fit_nvos_geometry_transfer_next72.py','--scene',scene,'--carrier',carrier])
  records=[]
  with (OUT/(scene+'.log')).open('w') as log:
   for c in cmds:
    rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;records.append(dict(command=c,exit_code=rc));(OUT/(scene+'_execution.json')).write_text(json.dumps(records,indent=2))
    if rc:raise RuntimeError((scene,c[1],rc))
if __name__=='__main__':
 with ThreadPoolExecutor(2) as p:list(p.map(run,[(2,['flower','fern','trex']),(3,['fortress','horns_center','horns_left'])]))
