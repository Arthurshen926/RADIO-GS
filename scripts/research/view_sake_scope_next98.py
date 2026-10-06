"""Source-only inspection of recovered secondary scope across all present views."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
key='lerf_ramen';t=torch.load('output/boundary_tracks_next95/high_merged/union/'+key+'/11.pt',map_location='cpu',weights_only=False);paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads(Path('output/source_coverage_next72/ramen/all.json').read_text())['images']};rows=[]
for f,r in sorted(t['frames'].items()):
 if not r['present'] or f==t['seed_frame']:continue
 m=r['probability']>.5
 if not m.any():continue
 im=Image.open(paths[f]).convert('RGB');mask=np.array(Image.fromarray(m.numpy()).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(mask);dx=max(30,int((xx.max()-xx.min())*.5));dy=max(30,int((yy.max()-yy.min())*.2));box=(max(0,xx.min()-dx),max(0,yy.min()-dy),min(im.width,xx.max()+dx+1),min(im.height,yy.max()+dy+1));im=im.crop(box);x0,y0,x1,y1=box;im.thumbnail((300,300));cm=torch.from_numpy(np.array(Image.fromarray(mask[y0:y1,x0:x1]).resize(im.size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(cm,3,1,1)+F.max_pool2d(-cm,3,1,1))[0,0]>.5;pixels=np.array(im);pixels[edge.numpy()]=[255,0,0];rows.append((f,int(m.sum()),Image.fromarray(pixels)))
rows=[rows[i] for i in np.linspace(0,len(rows)-1,min(20,len(rows)),dtype=int)];out=Path('output/next98');out.mkdir(exist_ok=True);canvas=Image.new('RGB',(1200,330*((len(rows)+3)//4)),'white');d=ImageDraw.Draw(canvas)
for i,(f,count,im) in enumerate(rows):
 x=i%4*300;y=i//4*330;canvas.paste(im,(x,y));d.text((x+4,y+302),f'source {f}, positive pixels {count}',fill='black')
canvas.save(out/'sake_track.jpg');(out/'sake_track.json').write_text(json.dumps(dict(seed=t['seed_frame'],frames=[dict(frame=f,pixels=n) for f,n,_ in rows]),indent=2)+'\n')
