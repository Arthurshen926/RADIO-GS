"""Fixed seed/track comparison with independently generated region agreement."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--initialization',choices=['semantic','source'],required=True);a=p.parse_args();key='lerf_'+a.scene;out=Path('output/verified_scope_next82')/a.initialization/key;out.mkdir(parents=True,exist_ok=True)
ids=[str(r['query_index']) for r in json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text())];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python'
cmd=[py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base','output/verified_track_regions_next82','--retained-selection',str(Path('output/region_seed_positive_next80')/key/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization',a.initialization,'--query-indices',*ids]
with (out/'execution.log').open('w') as log:
 subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
