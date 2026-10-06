"""Source-only text+visual exemplar discovery of additional category instances.
Seed from native expanded-bank top1. Tracker supplies a positive visual example,
not the complete text-query extent; SAM3 searches the full source image.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,sam3_autocast_context
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query-index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);load=lambda p:torch.load(p,map_location='cpu',weights_only=False);key='lerf_'+a.scene;qi=a.query_index;track=load(Path('output/small_native_tracks_next106/expanded')/key/f'{qi}.pt');paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/a.scene/'all.json').read_text())['images']};out=Path('output/text_set_tracks_next106')/key;out.mkdir(parents=True,exist_ok=True);rows=[];start=time.monotonic();processor=_load_sam3_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',device='cuda',confidence_threshold=.5,dtype='float32',resolution=1008)
with torch.inference_mode(),sam3_autocast_context('cuda',torch.bfloat16):
 for frame,record in sorted(track['frames'].items()):
  core=record['probability'].float();m=core>.5
  if not record['present'] or not m.any():continue
  yy,xx=torch.where(m);h,w=m.shape;box=[float((xx.min()+xx.max()+1)/(2*w)),float((yy.min()+yy.max()+1)/(2*h)),float((xx.max()-xx.min()+1)/w),float((yy.max()-yy.min()+1)/h)];image=Image.open(paths[frame]).convert('RGB');state=processor.set_image(image);det=processor.set_text_prompt(track['query'],state);scores=det['scores'].float().cpu();n=len(scores)
  if n:
   probs=F.interpolate(det['masks_logits'].float().reshape(n,1,*det['masks_logits'].shape[-2:]),size=(h,w),mode='bilinear',align_corners=False)[:,0].cpu();additional=probs.amax(0);new=torch.maximum(core,additional)
  else:probs=torch.empty(0,h,w);new=core
  record.update(probability=new.half(),present=bool((new>.5).any()),exemplar_box=box,detector_instances=probs.half(),detector_scores=scores)
  rows.append(dict(frame=frame,instances=n,core_pixels=int(m.sum()),union_pixels=int((new>.5).sum()),added_pixels=int(((new>.5)&~m).sum())))
  if frame in [20,40,115]:
   rgb=np.array(image.resize((768,576)));cm=np.array(Image.fromarray(m.numpy()).resize((768,576),Image.Resampling.NEAREST));nm=np.array(Image.fromarray((new>.5).numpy()).resize((768,576),Image.Resampling.NEAREST));rgb[nm]=(.5*rgb[nm]+.5*np.array([0,255,0])).astype('uint8');rgb[cm]=(.5*rgb[cm]+.5*np.array([255,0,0])).astype('uint8');Image.fromarray(rgb).save(out/f'{qi}_source_{frame}.jpg')
  print(key,frame,n,rows[-1]['added_pixels'],flush=True)
track.update(source='native single-core track plus exact-query SAM3 full-source category discovery; matched text-only control',discovery_seconds=time.monotonic()-start);torch.save(track,out/f'{qi}.pt');(out/f'{qi}.json').write_text(json.dumps(dict(query=track['query'],source_frames=list(track['frames']),source_only=True,protocol='exact runtime text only; same eligible source views; full source image; confidence0.5; preserve core and union discovered instances; no GT/aliases/VLM',rows=rows),indent=2)+'\n')
