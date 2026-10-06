"""Verbatim-query content reranking, full source tracking, matched scope fits."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();key='lerf_'+a.scene;root=Path('output/content_scope_next83');root.mkdir(parents=True,exist_ok=True)
ids=[str(r['query_index']) for r in json.loads((Path('output/content_seeds_next83')/key/'proposals.json').read_text())];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python'
with (root/(key+'_tracking.log')).open('w') as log:
 subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_full_text_sources_next79.py','--key',key,'--query-indices',*ids,'--seed-base','output/content_seeds_next83','--output-base','output/content_tracks_next83'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for initialization in ['semantic','source']:
 out=root/initialization/key;out.mkdir(parents=True,exist_ok=True)
 cmd=[py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base','output/content_tracks_next83','--retained-selection',str(Path('output/region_seed_positive_next80')/key/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization',initialization,'--query-indices',*ids]
 with (out/'execution.log').open('w') as log:
  subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
