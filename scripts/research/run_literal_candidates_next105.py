import argparse,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python';out=Path('output/literal_candidates_next105');out.mkdir(exist_ok=True,parents=True)
with (out/(a.scene+'.log')).open('w') as log:
 subprocess.run([py,'scripts/research/propose_literal_candidates_next105.py','--scene',a.scene],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,'scripts/research/verify_region_identity_next85.py','--scene',a.scene,'--candidate-base',str(out),'--output-base','output/literal_seeds_next105'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(a.scene,'candidates and verification complete',flush=True)
