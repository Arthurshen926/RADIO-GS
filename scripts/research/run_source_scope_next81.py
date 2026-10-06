"""Same next80 observations and objective, source-conditioned membership start."""
import argparse,json,os,subprocess
import torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--all-queries',action='store_true');p.add_argument('--gpu',required=True);a=p.parse_args();key='lerf_'+a.scene;out=Path('output/all_source_scope_next81' if a.all_queries else 'output/source_scope_next81')/key;out.mkdir(parents=True,exist_ok=True)
indices=[str(r['query_index']) for r in json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text())];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python'
commands=[[py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base','output/region_seed_tracks_next80','--retained-selection',str(Path('output/region_seed_positive_next80')/key/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--query-indices',*indices],[py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')]]
if a.all_queries:
 tracks=Path('output/combined_tracks_next81')/key;tracks.mkdir(parents=True,exist_ok=True)
 texts=torch.load(Path('output/region_seed_positive_next80')/key/'typed_readout.pt',map_location='cpu',weights_only=False)['texts']
 for qi in range(len(texts)):
  source=Path('output/region_seed_tracks_next80' if str(qi) in indices else 'output/full_text_tracks_next79')/key/f'{qi}.pt'
  link=tracks/f'{qi}.pt'
  if not link.exists():link.symlink_to(source.resolve())
 commands[0]=commands[0][:commands[0].index('--query-indices')]
 commands[0][commands[0].index('--tracks-base')+1]=str(tracks.parent)
with (out/'execution.log').open('w') as log:
 for cmd in commands:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
