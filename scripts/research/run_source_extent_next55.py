import os,subprocess,json,argparse
p=argparse.ArgumentParser();p.add_argument('--scenes',default='figurines,ramen,teatime,waldo_kitchen');p.add_argument('--domain',choices=['pixel','element'],default='pixel');args=p.parse_args();mode='isotonic' if args.domain=='pixel' else 'element_isotonic'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
def run(job):
 gpu,scene=job;key='lerf_'+scene;root=Path('output/source_extent_next55');root.mkdir(exist_ok=True,parents=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4');rows=[]
 commands=[[PY,'scripts/research/calibrate_source_extent_next55.py','--key',key,'--domain',args.domain]]+[[PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/mode),'--readout','anchored','--pipeline','--calibrated','--element-results',str(root/mode/key/'typed_readout.pt')] for mode in [mode]]
 with (root/(scene+'_'+args.domain+'.log')).open('w') as log:
  for cmd in commands:
   rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=cmd,exit_code=rc));print(scene,cmd[1],rc,flush=True)
   if rc:break
 return rows
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,[(i+2,s) for i,s in enumerate(args.scenes.split(','))]))
Path('output/source_extent_next55/execution_'+args.domain+'_'+args.scenes.replace(',','_')+'.json').write_text(json.dumps(rows,indent=2)+'\n')
