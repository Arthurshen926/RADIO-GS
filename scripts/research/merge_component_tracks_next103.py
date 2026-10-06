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
merge(key,rows,'output/component_tracks_next103','output/boundary_tracks_next95/high_secondary','output/component_merged_next103')
