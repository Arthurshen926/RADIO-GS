"""Matched low/high teacher rasters from the same full-source tracking run."""
import argparse,json,os,subprocess
from pathlib import Path
from scripts.research.multi_scope_next86 import merge
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--arms',nargs='+',default=['low','high']);a=p.parse_args();key='lerf_'+a.scene
rows=json.loads((Path('output/multi_seeds_next89')/key/'proposals.json').read_text());admitted=set(json.loads((Path('output/description_scope_next91')/key/'admission.json').read_text())['admitted'])
for row in rows:
 if str(row['query_index']) not in admitted:row['secondary_rank']=None
ids=[str(r['query_index']) for r in rows];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python'
for arm in a.arms:
 tracks=f'output/boundary_tracks_next95/{arm}_merged'
 merge(key,rows,f'output/boundary_tracks_next95/{arm}_primary',f'output/boundary_tracks_next95/{arm}_secondary',tracks)
 out=Path('output/boundary_scope_next95')/arm/key;out.mkdir(parents=True,exist_ok=True)
 with (out/'execution.log').open('w') as log:
  subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base',str(Path(tracks)/'union'),'--retained-selection',str(Path('output/scope_exclusion_next93/corroborated')/key/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--negative-admission','corroborated','--raster-scale','2' if arm=='high' else '1','--query-indices',*ids],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(key,arm,'complete',flush=True)
