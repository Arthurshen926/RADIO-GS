import argparse,json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
p=argparse.ArgumentParser();p.add_argument('--scenes',default='figurines,ramen,teatime,waldo_kitchen');p.add_argument('--arm',choices=['mass','ownership'],default='mass');a=p.parse_args()
PY='/root/miniconda3/envs/cybersim_agent/bin/python';root=Path('output/evidence_mass_next56');root.mkdir(parents=True,exist_ok=True)
def run(job):
 gpu,scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4');rows=[]
 commands=[[PY,'scripts/research/refine_evidence_mass_next56.py','--key',key]+(['--ownership'] if a.arm=='ownership' else []),[PY,'scripts/research/align_evidence_mass_next56.py','--key',key,'--arm',a.arm]]
 for mode in [a.arm+'_scope',a.arm+'_masscal']:
  commands.append([PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/mode),'--readout','anchored','--pipeline','--calibrated','--element-results',str(root/mode/key/'typed_readout.pt')])
 with (root/(scene+'_'+a.arm+'.log')).open('w') as log:
  for cmd in commands:
   rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));print(scene,cmd[1],rc,flush=True)
   if rc:break
 return rows
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,enumerate(a.scenes.split(','))))
(root/('execution_'+a.arm+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
