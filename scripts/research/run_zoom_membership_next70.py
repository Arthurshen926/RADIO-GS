import os,json,subprocess,argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
p=argparse.ArgumentParser();p.add_argument('--modes',default='cold,augment,replace');p.add_argument('--gpus',default='0,1,2,3,4,5');args=p.parse_args()
root=Path('output/zoom_grounding_next70');root.mkdir(parents=True,exist_ok=True)
gpus=args.gpus.split(',');jobs=[(mode,scene) for mode in args.modes.split(',') for scene in ['figurines','ramen','teatime','waldo_kitchen']]
def run(job):
 gpu,(mode,scene)=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',PYTHONPATH='.')
 commands=[[PYTHON,'scripts/research/decode_zoom_grounding_next70.py','--key',key,'--mode',mode],[PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/mode),'--readout','anchored','--pipeline','--calibrated','--element-results',str(root/mode/key/'typed_readout.pt')]];rows=[]
 with (root/(mode+'_'+scene+'.log')).open('w') as log:
  for c in commands:
   rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(mode=mode,scene=scene,command=c,exit_code=rc));print(mode,scene,c[1],rc,flush=True)
   if rc:break
 return rows
rows=[]
for start in range(0,len(jobs),len(gpus)):
 with ThreadPoolExecutor(len(gpus)) as pool:rows+=sum(pool.map(run,list(zip(gpus,jobs[start:start+len(gpus)]))),[])
(root/('execution_'+args.modes.replace(',','_')+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
