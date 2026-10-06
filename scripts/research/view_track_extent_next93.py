"""Source-only inspection of recovered secondary scope across all present views."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
key='lerf_waldo_kitchen';t=torch.load('output/secondary_tracks_next89/'+key+'/14.pt',map_location='cpu');paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads(Path('output/source_coverage_next72/waldo_kitchen/all.json').read_text())['images']};rows=[]
for f,r in sorted(t['frames'].items()):
 if not r['present'] or f==t['seed_frame']:continue
 m=r['probability']>.5
 if not m.any():continue
 im=Image.open(paths[f]).convert('RGB');mask=np.array(Image.fromarray(m.numpy()).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(mask);dx=max(30,int((xx.max()-xx.min())*.5));dy=max(30,int((yy.max()-yy.min())*.2));box=(max(0,xx.min()-dx),max(0,yy.min()-dy),min(im.width,xx.max()+dx+1),min(im.height,yy.max()+dy+1));im=im.crop(box);x0,y0,x1,y1=box;im.thumbnail((300,300));cm=torch.from_numpy(np.array(Image.fromarray(mask[y0:y1,x0:x1]).resize(im.size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(cm,3,1,1)+F.max_pool2d(-cm,3,1,1))[0,0]>.5;pixels=np.array(im);pixels[edge.numpy()]=[255,0,0];rows.append((f,int(m.sum()),Image.fromarray(pixels)))
out=Path('output/next93');out.mkdir(exist_ok=True);canvas=Image.new('RGB',(1200,330*((len(rows)+3)//4)),'white');d=ImageDraw.Draw(canvas)
for i,(f,count,im) in enumerate(rows):
 x=i%4*300;y=i//4*330;canvas.paste(im,(x,y));d.text((x+4,y+302),f'source {f}, positive pixels {count}',fill='black')
canvas.save(out/'secondary_track.jpg');(out/'source_track.json').write_text(json.dumps(dict(seed=t['seed_frame'],frames=[dict(frame=f,pixels=n) for f,n,_ in rows]),indent=2)+'\n')
