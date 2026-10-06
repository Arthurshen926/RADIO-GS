"""Matched next72 selected-set decoding after native field reconstruction."""
import argparse, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--pooling',choices=['logmeanexp','scope_max'],default='scope_max');a=p.parse_args()
key='lerf_'+a.scene
root=Path('output/late_region_next78')/a.pooling/'all'
out=root/'joint_extent'/key;out.mkdir(parents=True,exist_ok=True)
py='/root/miniconda3/envs/cybersim_agent/bin/python'
env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu)
commands=[
 [py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base',str(root/'calibrated'),'--cohort','old','--evidence-base','output/typed_evidence_next44','--output-base',str(root/'joint_extent'),'--mode','seeded'],
 [py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/'joint_extent'),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')]
]
with (out/'execution.log').open('w') as log:
 for command in commands:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
