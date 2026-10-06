"""Post-hoc camera-conditioned SOURCE crops, not prediction inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
d=json.loads(Path('output/next86/pool_coverage.json').read_text());paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads(Path('output/source_coverage_next72/waldo_kitchen/all.json').read_text())['images']};canvas=Image.new('RGB',(900,600),'white');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(d['top_camera_covering_nodes'][:6]):
 b=load(r['path']);native=load(b['source']);sam=load(native['sam_path']);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);mask=masks[int(native['native_proposal_indices'][r['local_region']])];im=Image.open(paths[r['frame']]).convert('RGB');m=np.array(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(m);dx=max(10,int((xx.max()-xx.min())*.25));dy=max(10,int((yy.max()-yy.min())*.25));im=im.crop((max(0,xx.min()-dx),max(0,yy.min()-dy),min(im.width,xx.max()+dx+1),min(im.height,yy.max()+dy+1)));im.thumbnail((290,255));x=i%3*300;y=i//3*300;canvas.paste(im,(x,y));draw.text((x+4,y+260),f"node {r['node']}, source frame {r['frame']}",fill='black');draw.text((x+4,y+280),f"native {r['native_score']:.3f}",fill='black')
canvas.save('output/next86/camera_covering_source_crops.jpg')
