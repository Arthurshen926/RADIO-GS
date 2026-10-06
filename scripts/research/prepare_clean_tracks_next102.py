"""Keep old source admission; replace only propagation from cleaned primary seeds."""
import argparse,json
from pathlib import Path
import torch
from scripts.research.multi_scope_next86 import merge
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();key='lerf_'+a.scene
rows=json.loads((Path('output/multi_seeds_next89')/key/'proposals.json').read_text());admitted=set(json.loads((Path('output/description_scope_next91')/key/'admission.json').read_text())['admitted'])
for r in rows:
 if str(r['query_index']) not in admitted:r['secondary_rank']=None
merge(key,rows,'output/clean_tracks_next102','output/boundary_tracks_next95/high_secondary','output/clean_merged_next102')
out=Path('output/clean_admitted_next102')/key;out.mkdir(parents=True,exist_ok=True)
for row in rows:
 qi=row['query_index'];new=load(Path('output/clean_merged_next102/union')/key/f'{qi}.pt');old=load(Path('output/uncertain_tracks_next99/all')/key/f'{qi}.pt')
 for f,r in new['frames'].items():
  for k in ['positive_allowed','negative_allowed','scope_reliability']:
   if k in old['frames'][f]:r[k]=old['frames'][f][k]
 torch.save(new,out/f'{qi}.pt')
