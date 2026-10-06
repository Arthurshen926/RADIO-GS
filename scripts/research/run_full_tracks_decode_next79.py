"""Matched old-raster extent decoding with full-sequence temporal observations."""
import argparse,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--cohort',choices=['old','all'],required=True);p.add_argument('--query-indices',type=int,nargs='*');p.add_argument('--tag',required=True);p.add_argument('--steps',type=int,default=240);a=p.parse_args()
key='lerf_'+a.scene;root=Path('output/full_text_decode_next79')/a.tag/a.cohort;out=root/key;out.mkdir(parents=True,exist_ok=True);py='/root/miniconda3/envs/cybersim_agent/bin/python';env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu)
cmd=[py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort',a.cohort,'--evidence-base','output/typed_evidence_next44','--output-base',str(root),'--mode','seeded','--steps',str(a.steps),'--tracks-base','output/full_text_tracks_next79','--retained-selection',str(Path('output/native_write_next75/all/joint_extent')/key/'typed_readout.pt')]
if a.query_indices is not None:cmd+=['--query-indices',*map(str,a.query_indices)]
if a.cohort=='all':cmd+=['--extra-track-frames']
with (out/'execution.log').open('w') as log:
 subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
