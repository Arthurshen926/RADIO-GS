import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();key='lerf_'+a.scene;env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);root=Path('output/partner_tracks_next102');root.mkdir(parents=True,exist_ok=True)
subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/build_partner_seeds_next102.py','--scene',a.scene],env=env,check=True)
ids=[str(r['query_index']) for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())]
with (root/(key+'.log')).open('w') as log:
 subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_full_text_sources_next79.py','--key',key,'--query-indices',*ids,'--seed-base','output/partner_seeds_next102','--output-base',str(root),'--raster-scale','2'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(key,'partner tracks complete',flush=True)
