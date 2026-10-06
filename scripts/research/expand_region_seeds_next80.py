import os,subprocess,sys
from pathlib import Path
scene,gpu=sys.argv[1:];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python';root=Path('output/next80');root.mkdir(exist_ok=True)
with (root/(scene+'_build.log')).open('w') as log:
 for script in ['build_region_bank_next78','build_region_relations_next79','propose_region_seeds_next80']:
  subprocess.run([py,'scripts/research/'+script+'.py','--scene',scene],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,'scripts/research/run_region_seeds_next80.py','--scene',scene,'--gpu',gpu],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
