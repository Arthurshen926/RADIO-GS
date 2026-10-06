"""Use the other automatic region of the SAME frozen candidate pair as anchor."""
import argparse,json
from pathlib import Path
from functools import lru_cache
import torch
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();key='lerf_'+a.scene
relation=load(Path('output/region_relations_next79')/a.scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
for e in manifest['entries']:nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
@lru_cache(maxsize=4)
def bank(path):
 b=load(path);r=load(b['source']);sam=load(r['sam_path']);return r,unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
out=Path('output/partner_seeds_next102')/key;out.mkdir(parents=True,exist_ok=True);report=[]
for row in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()):
 qi=row['query_index'];seed=load(Path('output/grounded_seeds_next88')/key/f'{qi}.pt');node=next(n for n,f in zip(seed['pair_nodes'],seed['pair_frames']) if f!=seed['seed_frame']);frame,ri,path=nodes[node];r,masks=bank(path);mask=masks[int(r['native_proposal_indices'][ri])].copy();partner=dict(seed,seed_frame=frame,masks=torch.from_numpy(mask)[None].bool(),source='other automatic region in unchanged source candidate pair; not propagated from first anchor')
 torch.save(partner,out/f'{qi}.pt');report.append(dict(query=seed['query'],query_index=qi,old_anchor=seed['seed_frame'],new_anchor=frame,pair_nodes=seed['pair_nodes']))
(out/'anchors.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
