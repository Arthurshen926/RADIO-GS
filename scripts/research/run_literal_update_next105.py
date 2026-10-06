import argparse,json,os,subprocess,shutil
from pathlib import Path
from scripts.research.multi_scope_next86 import merge
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();key='lerf_'+a.scene;env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python';root=Path('output/literal_update_next105');root.mkdir(exist_ok=True,parents=True)
subprocess.run([py,'scripts/research/prepare_literal_updates_next105.py','--scene',a.scene],env=env,check=True);decisions=json.loads((root/(key+'_decisions.json')).read_text());ids=[str(r['query_index']) for r in decisions if r['update']];parent=Path('output/core_scope_next104/core_update')/key
with (root/(key+'_tracking.log')).open('w') as log:
 if ids:subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_full_text_sources_next79.py','--key',key,'--query-indices',*ids,'--seed-base','output/literal_seeds_next105','--output-base','output/literal_update_tracks_next105','--raster-scale','2'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
rows=json.loads((Path('output/multi_seeds_next89')/key/'proposals.json').read_text());admitted=set(json.loads((Path('output/description_scope_next91')/key/'admission.json').read_text())['admitted'])
for r in rows:
 if str(r['query_index']) not in admitted:r['secondary_rank']=None
merge(key,rows,'output/literal_update_tracks_next105','output/boundary_tracks_next95/high_secondary','output/literal_merged_next105');out=root/key;out.mkdir(exist_ok=True,parents=True)
with (out/'execution.log').open('w') as log:
 if ids:subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(root),'--mode','seeded','--tracks-base','output/literal_merged_next105/union','--retained-selection',str(parent/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--negative-admission','corroborated','--raster-scale','2','--pixel-loss','balanced','--support-prior','footprint','--support-prior-weight','0.1','--query-indices',*ids],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 else:
  for name in ['typed_readout.pt','field.pt','calibration.pt']:shutil.copyfile(parent/name,out/name)
 subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(key,'literal update complete',flush=True)
