import json,torch,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 root=Path('output/region_seeds_next80')/('lerf_'+scene);paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']}
 for r in json.loads((root/'proposals.json').read_text()):
  d=torch.load(root/f"{r['query_index']}.pt",weights_only=False);mask=d['masks'][0].numpy();im=Image.open(paths[r['seed_frame']]).convert('RGB');mh,mw=mask.shape;yy,xx=np.where(mask);x0,x1=max(0,int(xx.min()*im.width/mw)-15),min(im.width,int((xx.max()+1)*im.width/mw)+15);y0,y1=max(0,int(yy.min()*im.height/mh)-15),min(im.height,int((yy.max()+1)*im.height/mh)+15);crop=im.crop((x0,y0,x1,y1));crop.thumbnail((250,210));rows.append((r['query'],crop))
canvas=Image.new('RGB',(1000,250*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
for j,(text,crop) in enumerate(rows):x=(j%4)*250;y=(j//4)*250;canvas.paste(crop,(x,y));draw.text((x+3,y+215),text,fill='black')
canvas.save('output/region_seeds_next80/source_crops.jpg')
