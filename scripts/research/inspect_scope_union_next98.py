"""Source-only inspection of detector/track union; no target labels."""
import json
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image,ImageDraw

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);out=Path('output/next98');out.mkdir(exist_ok=True);rows=[];tiles=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 key='lerf_'+scene;props=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text());data=load(Path('output/native_write_next75/all/calibrated')/key/'field.pt');info=json.loads((Path(data['root'])/'teachers.json').read_text());paths={int(x['image_id'].split('_')[-1]):x['path'] for x in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']}
 for q in props:
  qi=q['query_index'];track=load(Path('output/boundary_tracks_next95/high_merged/union')/key/f'{qi}.pt');frames=[]
  for f in info['train_frames']:
   det=load(Path('output/typed_evidence_next44')/key/f'identity_{f}.pt')['instances'][qi];n=len(det['scores'])
   if not n:continue
   masks=F.interpolate(det['masks'].float()[:,None],size=(192,256),mode='bilinear',align_corners=False)[:,0]>.5;admitted=det['foreground_margin']>0;union=masks.any(0);rejected=masks[~admitted].any(0) if (~admitted).any() else torch.zeros_like(union);record=track['frames'].get(f,{});present=bool(record.get('present',False)) and f!=track['seed_frame'];temporal=(record['probability']>.5) if present else torch.zeros_like(union);extra=union&~temporal
   frames.append(dict(frame=f,instances=n,positive_margin=int(admitted.sum()),margins=det['foreground_margin'].tolist(),detector_pixels=int(union.sum()),temporal_pixels=int(temporal.sum()),extra_detector_pixels=int(extra.sum()),rejected_extra_pixels=int((rejected&~temporal).sum())))
   if scene=='ramen' and q['query']=='sake cup' and extra.any():
    im=Image.open(paths[f]).convert('RGB').resize((512,384));pixels=np.array(im);dm=np.array(Image.fromarray(union.numpy()).resize(im.size,Image.Resampling.NEAREST));tm=np.array(Image.fromarray(temporal.numpy()).resize(im.size,Image.Resampling.NEAREST));pixels[dm&~tm]=(pixels[dm&~tm]*.4+np.array([0,100,255])*.6).astype('uint8');pixels[tm]=(pixels[tm]*.5+np.array([255,0,0])*.5).astype('uint8');tiles.append((f,Image.fromarray(pixels),int(extra.sum())))
  rows.append(dict(scene=scene,query=q['query'],query_index=qi,frames=frames))
Path('output/next98/source_union.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row['scene'],row['query'],sum(x['instances'] for x in row['frames']),sum(x['positive_margin'] for x in row['frames']),sum(x['extra_detector_pixels'] for x in row['frames']))
if tiles:
 tiles=sorted(tiles,key=lambda x:x[0]);canvas=Image.new('RGB',(1536,412*((len(tiles)+2)//3)),'white');draw=ImageDraw.Draw(canvas)
 for i,(f,im,n) in enumerate(tiles):x=i%3*512;y=i//3*412;canvas.paste(im,(x,y));draw.text((x+5,y+387),f'source {f}: red track; blue detector-only {n}px',fill='black')
 canvas.save(out/'sake_source_union.jpg')
