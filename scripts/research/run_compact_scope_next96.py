"""Compact selected-set priors on matched source evidence; fixed coefficient."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--arms',nargs='+',default=['low_count','low_footprint','high_footprint']);a=p.parse_args();key='lerf_'+a.scene
ids=[str(r['query_index']) for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python'
for arm in a.arms:
 raster,prior=arm.split('_');out=Path('output/compact_scope_next96')/arm/key;out.mkdir(parents=True,exist_ok=True)
 with (out/'execution.log').open('w') as log:
  subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base',f'output/boundary_tracks_next95/{raster}_merged/union','--retained-selection',str(Path('output/scope_exclusion_next93/corroborated')/key/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--negative-admission','corroborated','--raster-scale','2' if raster=='high' else '1','--support-prior',prior,'--support-prior-weight','0.1','--query-indices',*ids],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(key,arm,'complete',flush=True)
