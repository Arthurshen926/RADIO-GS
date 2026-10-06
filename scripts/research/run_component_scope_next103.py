"""Matched uncertain-component recovery with frozen geometry and teachers."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--arms',nargs='+',default=['rebuilt']);a=p.parse_args();key='lerf_'+a.scene;ids=[str(r['query_index']) for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python';parent=Path('output/category_identity_scope_next98/high_footprint')/key/'typed_readout.pt'
state=Path('output/fixed_measure_next100/state')/(key+'.pt')

subprocess.run([py,'scripts/research/merge_component_tracks_next103.py','--scene',a.scene],env=env,check=True)
for arm in a.arms:
 trackbase='output/component_merged_next103/union'
 stateargs=[]
 out=Path('output/component_scope_next103')/arm/key;out.mkdir(parents=True,exist_ok=True)
 with (out/'execution.log').open('w') as log:
  subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base',trackbase,'--retained-selection',str(parent),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--negative-admission','corroborated','--raster-scale','2','--pixel-loss','balanced','--source-coverage-reference',str(parent),'--support-prior','footprint','--support-prior-weight','0.1',*stateargs,'--query-indices',*ids],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(key,arm,'complete',flush=True)
