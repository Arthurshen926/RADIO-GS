import argparse,os,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--scenes',default='figurines,ramen,teatime,waldo_kitchen');p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--steps',default='3000');a=p.parse_args();root=Path('output/joint_carrier_next66');root.mkdir(parents=True,exist_ok=True)
def run(job):
 gpu,scene=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='.',OMP_NUM_THREADS='4')
 with (root/(scene+'_'+a.arm+'.log')).open('w') as log:
  rc=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/reconstruct_adaptive_carrier_next66.py','--scene',scene,'--arm',a.arm,'--steps',a.steps],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 print(scene,a.arm,rc,flush=True);return dict(scene=scene,arm=a.arm,exit_code=rc)
scenes=a.scenes.split(',');gpus=a.gpus.split(',');rows=[]
for start in range(0,len(scenes),len(gpus)):
 with ThreadPoolExecutor(len(gpus)) as pool:rows+=list(pool.map(run,zip(gpus,scenes[start:start+len(gpus)])))
(root/('execution_'+a.arm+'_'+'_'.join(scenes)+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
