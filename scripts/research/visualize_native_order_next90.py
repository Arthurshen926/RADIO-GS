"""Source-only matched scope choices, not an input to prediction."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']}
 for row in json.loads((Path('output/native_order_seeds_next90/qualified_native')/key/'proposals.json').read_text()):
  crops=[]
  for base in ['output/grounded_seeds_next88','output/native_order_seeds_next90/qualified_native']:
   seed=torch.load(Path(base)/key/f"{row['query_index']}.pt",map_location='cpu');im=Image.open(paths[seed['seed_frame']]).convert('RGB');m=np.array(Image.fromarray(seed['masks'][0].numpy()).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(m);dx=max(10,int((xx.max()-xx.min())*.25));dy=max(10,int((yy.max()-yy.min())*.25));box=(max(0,xx.min()-dx),max(0,yy.min()-dy),min(im.width,xx.max()+dx+1),min(im.height,yy.max()+dy+1));im=im.crop(box);x0,y0,x1,y1=box;im.thumbnail((235,190));cm=torch.from_numpy(np.array(Image.fromarray(m[y0:y1,x0:x1]).resize(im.size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(cm,3,1,1)+F.max_pool2d(-cm,3,1,1))[0,0]>.5;arr=np.array(im);arr[edge.numpy()]=[255,0,0];crops.append(Image.fromarray(arr))
  rows.append((row['query'],crops))
canvas=Image.new('RGB',(960,220*((len(rows)+1)//2)),'white');d=ImageDraw.Draw(canvas)
for i,(query,crops) in enumerate(rows):
 x=i%2*480;y=i//2*220
 for j,im in enumerate(crops):canvas.paste(im,(x+j*240,y));d.text((x+j*240+3,y+195),query,fill='black');d.text((x+j*240+3,y+208),'VLM rank' if j==0 else 'native rank after admission',fill='black')
out=Path('output/next90');out.mkdir(exist_ok=True);canvas.save(out/'source_choices.jpg')
