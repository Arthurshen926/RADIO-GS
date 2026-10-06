"""Source-only candidate gallery for zero-query diagnosis, not prediction selection."""
import json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from scripts.research.verify_track_identity_next98 import crop
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);out=Path('output/next105');out.mkdir(exist_ok=True)
for scene,qi in [('ramen',9),('waldo_kitchen',7)]:
 key='lerf_'+scene;relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
 for e in manifest['entries']:nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
 paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']};r=next(r for r in json.loads((Path('output/literal_seeds_next105')/key/'proposals.json').read_text()) if r['query_index']==qi);canvas=Image.new('RGB',(4*260,4*290),'white');draw=ImageDraw.Draw(canvas)
 for i,c in enumerate(r['candidates']):
  node=c['nodes'][0];frame,ri,path=nodes[node];b=load(path);native=load(b['source']);sam=load(native['sam_path']);ms=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);mask=ms[int(native['native_proposal_indices'][ri])].copy();im=crop(paths[frame],mask);im.thumbnail((255,255));xy=(i%4*260,i//4*290);canvas.paste(im,(xy[0],xy[1]+30));draw.text(xy,f"{i}: source {frame}, m={min(r['candidate_margin'][i]):.2f}",fill='black')
 canvas.save(out/(key+'_candidates.jpg'))
