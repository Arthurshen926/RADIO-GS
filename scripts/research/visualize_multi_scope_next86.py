"""Compare source-only primary and secondary choices; no annotation access."""
import json,argparse
p=argparse.ArgumentParser();p.add_argument("--primary-base",default="output/grounded_seeds_next85");p.add_argument("--secondary-base",default="output/multi_seeds_next86");p.add_argument("--output-base",default="output/next86");a=p.parse_args()
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene
 paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']}
 for row in json.loads((Path(a.secondary_base)/key/'proposals.json').read_text()):
  if row['secondary_rank'] is None:continue
  crops=[]
  for base in [a.primary_base,a.secondary_base]:
   seed=torch.load(Path(base)/key/f"{row['query_index']}.pt",map_location='cpu');im=Image.open(paths[seed['seed_frame']]).convert('RGB');m=np.array(Image.fromarray(seed['masks'][0].numpy()).resize(im.size,Image.Resampling.NEAREST));y,x=np.where(m);dx=max(10,int((x.max()-x.min())*.25));dy=max(10,int((y.max()-y.min())*.25));box=(max(0,x.min()-dx),max(0,y.min()-dy),min(im.width,x.max()+dx+1),min(im.height,y.max()+dy+1));im=im.crop(box);x0,y0,x1,y1=box;im.thumbnail((370,230));mask=torch.from_numpy(np.array(Image.fromarray(m[y0:y1,x0:x1]).resize(im.size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(mask,3,1,1)+F.max_pool2d(-mask,3,1,1))[0,0]>.5;arr=np.array(im);arr[edge.numpy()]=[255,0,0];crops.append(Image.fromarray(arr))
  rows.append((row,crops))
out=Path(a.output_base);out.mkdir(exist_ok=True);canvas=Image.new('RGB',(780,270*len(rows)),'white');d=ImageDraw.Draw(canvas)
for i,(row,crops) in enumerate(rows):
 for j,im in enumerate(crops):canvas.paste(im,(j*390,i*270));d.text((j*390+4,i*270+235),row['query']+(' primary' if j==0 else ' secondary'),fill='black')
canvas.save(out/'source_hypotheses.jpg')
