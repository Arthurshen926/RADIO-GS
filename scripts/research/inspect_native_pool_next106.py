"""Source-only native ranking before two-view proposal admission; no evaluation inputs."""
import argparse,json
from pathlib import Path
from functools import lru_cache
import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image,ImageDraw
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from scripts.research.verify_track_identity_next98 import crop
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query-index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4)
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
scene=a.scene;key='lerf_'+scene;qi=a.query_index
relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[];features=[]
for e in manifest['entries']:
 b=load(e['path']);features.append(F.normalize(b['descriptors'].float(),dim=1));nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
f=load(Path('output/native_rebuild_next54/mixture_masscal')/key/'field.pt');root=Path(f['root']);q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float(),dim=-1);scores=torch.cat(features)@q[qi];ranked=scores.argsort(descending=True).tolist();rank={n:i+1 for i,n in enumerate(ranked)}
old=next(r for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()) if r['query_index']==qi);used={n for c in old['candidates'] for n in c['nodes']}
paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']}
@lru_cache(maxsize=8)
def masks(path):
 b=load(path);native=load(b['source']);sam=load(native['sam_path']);return native,unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
out=Path('output/next106')/key;out.mkdir(parents=True,exist_ok=True);rows=[]
for page in range(4):
 canvas=Image.new('RGB',(6*210,6*240),'white');draw=ImageDraw.Draw(canvas)
 for k,node in enumerate(ranked[page*36:(page+1)*36]):
  frame,ri,path=nodes[node];native,ms=masks(path);mask=ms[int(native['native_proposal_indices'][ri])].copy();im=crop(paths[frame],mask);im.thumbnail((205,205));xy=(k%6*210,k//6*240);canvas.paste(im,(xy[0],xy[1]+30));draw.text(xy,f"#{rank[node]} f{frame} n{node}\n{float(scores[node]):.3f} paired={node in used}",fill='black');rows.append(dict(rank=rank[node],node=node,frame=frame,score=float(scores[node]),in_paired_shortlist=node in used,pixels=int(mask.sum())))
 canvas.save(out/f'{qi}_native_{page}.jpg')
(out/f'{qi}_pool.json').write_text(json.dumps(dict(query=old['query'],query_index=qi,region_count=len(nodes),source_frames=len(manifest['entries']),pair_node_ranks=sorted(rank[n] for n in used),gallery=rows,target_rgb_or_gt_used=False),indent=2)+'\n');print(key,old['query'],len(nodes),sorted(rank[n] for n in used),flush=True)
