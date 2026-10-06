"""Source-only qualitative views: largest newly proposed pixel count per query.
These illustrations are diagnostics, not seed choices or evidence admission.
"""
import json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
out=Path('output/next107');out.mkdir(exist_ok=True)
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']};rows=[]
 for p in sorted((Path('output/set_discovery_next107/box')/key).glob('*.json')):
  if p.name=='contract.json':continue
  d=json.loads(p.read_text());j=max(d['rows'],key=lambda x:x['added_pixels']);frame=j['frame'];qi=d['query_index'];base=load(Path('output/core_merged_next104/union')/key/f'{qi}.pt');box=load(Path('output/set_discovery_next107/box')/key/f'{qi}.pt');comp=load(Path('output/set_discovery_next107/components')/key/f'{qi}.pt');ims=[]
  for name,t in [('core',base),('box',box),('components',comp)]:
   image=Image.open(paths[frame]).convert('RGB').resize((512,384));rgb=np.array(image);m=t['frames'][frame]['probability']>.5;m=np.array(Image.fromarray(m.numpy()).resize(image.size,Image.Resampling.NEAREST));rgb[m]=(.55*rgb[m]+.45*np.array([0,255,0])).astype('uint8');canvas=Image.new('RGB',(512,420),'white');canvas.paste(Image.fromarray(rgb),(0,36));ImageDraw.Draw(canvas).text((5,3),f"{d['query']} | source {frame} | {name}",fill='black');ims.append(canvas)
  rows.append(ims)
 canvas=Image.new('RGB',(1536,len(rows)*420),'white')
 for y,ims in enumerate(rows):
  for x,im in enumerate(ims):canvas.paste(im,(512*x,420*y))
 canvas.save(out/(scene+'_source_discovery.jpg'))
