"""Use native regions as search windows, not object supervision masks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,sam3_autocast_context

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def source_box(mask):
 import math
 y,x=torch.where(mask);H,W=mask.shape
 cx,cy=(int(x.min())+int(x.max())+1)/2,(int(y.min())+int(y.max())+1)/2
 bw,bh=(int(x.max())-int(x.min())+1)*1.5,(int(y.max())-int(y.min())+1)*1.5
 return [max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),min(W,math.ceil(cx+bw/2)),min(H,math.ceil(cy+bh/2))]

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 data=load(Path('output/taskwise_scope_composition_next38')/a.key/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());paths={r['frame_idx']:Path(src['image_dir'])/r['source_file'] for r in src['frames']}
 masks=load(root/'episodes.pt')['masks'];h,w=masks.shape[-2:]
 processor=_load_sam3_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',device='cuda',confidence_threshold=.5,dtype='float32',resolution=1008)
 out=Path('output/zoom_grounding_next70/evidence')/a.key;out.mkdir(parents=True,exist_ok=True);summary=[]
 for frame in info['train_frames']:
  native=load(Path('output/native_proposals_next69/evidence')/a.key/f'identity_{frame}.pt');image=Image.open(paths[frame]).convert('RGB');rgb=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255;H,W=rgb.shape[1:]
  records=[dict(masks=torch.empty(0,h,w,dtype=torch.float16),scores=torch.empty(0),foreground_margin=torch.empty(0)) for _ in native['queries']];groups={}
  for qi,r in enumerate(native['instances']):
   if len(r['scores']):groups.setdefault(r['proposal_id'],[]).append(qi)
  windows={}
  for rid,qis in groups.items():
   m=F.interpolate((masks[rid]>.5)[None,None].float(),size=(H,W),mode='nearest-exact')[0,0].bool();box=source_box(m);x0,y0,x1,y1=box;crop=image.crop(box)
   with sam3_autocast_context('cuda',torch.bfloat16):state=processor.set_image(crop)
   for qi in qis:
    with sam3_autocast_context('cuda',torch.bfloat16):det=processor.set_text_prompt(native['queries'][qi],dict(state))
    scores=det['scores'];n=len(scores);windows[native['queries'][qi]]=box
    if not n:continue
    local=det['masks_logits'].float().reshape(n,1,*det['masks_logits'].shape[-2:]);local=F.interpolate(local,size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[:,0]
    full=torch.zeros(n,H,W,device='cuda');full[:,y0:y1,x0:x1]=local
    low=F.interpolate(full[:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0]
    records[qi]=dict(masks=low.half().cpu(),scores=scores.float().cpu(),foreground_margin=torch.full((n,),-1.))
   del state
  torch.save(dict(frame=frame,queries=native['queries'],instances=records,windows=windows),out/f'detection_{frame}.pt');summary.append(dict(frame=frame,accepted=[t for t,r in zip(native['queries'],records) if len(r['scores'])],window_count=len(groups)))
  print(a.key,frame,len(summary[-1]['accepted']),round(time.monotonic()-start,1),flush=True)
 (out/'fit.json').write_text(json.dumps(dict(source_frames=info['train_frames'],seconds=time.monotonic()-start,records=summary,target_rgb_used=False,native_region_is_search_window_only=True,detector='SAM3 at1008 on1.5x source region bbox; same exact query',foreground_context_qualification='separate native encoder stage',new_source_views=False),indent=2)+'\n')
if __name__=='__main__':main()
