import os,subprocess,json,argparse
p=argparse.ArgumentParser();p.add_argument('--scenes',default='figurines,ramen,teatime,waldo_kitchen');args=p.parse_args()
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def run(job):
 gpu,scene=job;key='lerf_'+scene;root=Path('output/native_rebuild_next54');root.mkdir(exist_ok=True,parents=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4');rows=[]
 commands=[[PY,'scripts/research/refine_native_scope_next54.py','--key',key]]+[[PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/mode),'--readout','anchored','--pipeline','--calibrated','--element-results',str(root/mode/key/'typed_readout.pt')] for mode in ['mixture_scope']]
 with (root/(scene+'_scope.log')).open('w') as log:
  for cmd in commands:
   rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));print(scene,cmd[1],rc,flush=True)
   if rc:break
 return rows
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,[(i+2,s) for i,s in enumerate(args.scenes.split(','))]))
Path('output/native_rebuild_next54/execution_scope_'+args.scenes.replace(',','_')+'.json').write_text(json.dumps(rows,indent=2)+'\n')
