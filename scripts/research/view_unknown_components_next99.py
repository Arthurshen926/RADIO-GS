"""Uniformly sampled source-only ambiguous components; no target labels."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
from PIL import Image,ImageDraw
from scripts.research.verify_track_identity_next98 import crop
key='lerf_figurines';qi=17;scene='figurines';d=json.loads((Path('output/category_identity_tracks_next98')/key/f'{qi}.json').read_text());t=torch.load(Path('output/boundary_tracks_next95/high_merged/union')/key/f'{qi}.pt',map_location='cpu',weights_only=False);paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']};js=[j for j in d['judgments'] if j['answer']=='C'];chosen=[js[i] for i in np.linspace(0,len(js)-1,min(12,len(js)),dtype=int)];canvas=Image.new('RGB',(1200,330*((len(chosen)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
for i,j in enumerate(chosen):
 f=j['frame'];ls,_=label((t['frames'][f]['probability']>.5).numpy(),structure=np.ones((3,3)));im=crop(paths[f],ls==j['component']);im.thumbnail((300,300));x=i%4*300;y=i//4*330;canvas.paste(im,(x,y));draw.text((x+4,y+304),f'source {f} component {j["component"]}, C',fill='black')
out=Path('output/next99');out.mkdir(exist_ok=True);canvas.save(out/'handle_unknown.jpg');(out/'handle_unknown.json').write_text(json.dumps(chosen,indent=2)+'\n')
