"""Source-only before/after evidence; no evaluated frame or annotation."""
import json,torch,numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
cases=[('ramen','sake cup'),('waldo_kitchen','pour-over vessel'),('ramen','kamaboko')];canvas=Image.new('RGB',(700,810),'white');draw=ImageDraw.Draw(canvas)
for row,(scene,text) in enumerate(cases):
 key='lerf_'+scene;record=next(r for r in json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text()) if r['query']==text);qi=record['query_index'];d=torch.load(Path('output/region_seeds_next80')/key/f'{qi}.pt',weights_only=False);paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']};im=Image.open(paths[d['seed_frame']]).convert('RGB');mask=d['masks'][0].numpy();yy,xx=np.where(mask);h,w=mask.shape;crop=im.crop((int(xx.min()*im.width/w),int(yy.min()*im.height/h),int((xx.max()+1)*im.width/w),int((yy.max()+1)*im.height/h)));new=Image.open(Path('output/content_seeds_next83')/key/f'{qi}_chosen.jpg')
 for col,(title,crop) in enumerate([('Native cosine',crop),('Content likelihood',new)]):
  crop.thumbnail((340,230));canvas.paste(crop,(col*350,row*270));draw.text((col*350+3,row*270+237),text+' / '+title,fill='black')
out=Path('output/next83');out.mkdir(exist_ok=True);canvas.save(out/'source_choices.jpg')
