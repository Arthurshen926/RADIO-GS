import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--gpu');p.add_argument('--secondary',action='store_true');a=p.parse_args();key='lerf_'+a.scene
ids=json.loads((Path('output/description_scope_next91')/key/'admission.json').read_text())['admitted'] if a.secondary else [str(r['query_index']) for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())]
role='secondary' if a.secondary else 'primary';seeds='output/multi_seeds_next89' if a.secondary else 'output/grounded_seeds_next88'
env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu)
with Path(f'output/next95/{key}_{role}_tracking.log').open('w') as log:
 subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_full_text_sources_next79.py','--key',key,'--query-indices',*ids,'--seed-base',seeds,'--output-base',f'output/boundary_tracks_next95/high_{role}','--raster-scale','2','--low-output-base',f'output/boundary_tracks_next95/low_{role}'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(key,role,'complete',flush=True)
