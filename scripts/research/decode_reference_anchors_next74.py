"""Re-identify source objects from transported reference-positive anchors."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,sam3_autocast_context
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--base',default='output/reference_anchors_next74');p.add_argument('--policy',choices=['joint','cover'],default='joint');a=p.parse_args();torch.set_num_threads(4);start=time.monotonic();base=Path(a.base)/a.scene;plan=json.loads((base/'anchors.json').read_text());width,height=plan['raster_wh'];processor=_load_sam3_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',device='cuda',confidence_threshold=.5,dtype='float32',resolution=1008,point_only=True,build_on_cpu=True)
 original=torch.load(Path('output/typed_evidence_next44/full_source_tracks')/a.scene/'tracks.pt',map_location='cpu',weights_only=False);variants={m:dict(original['frames']) for m in ['replace','union']};details=[]
 for row in plan['views']:
  frame=row['frame']
  if not row['points'] or frame==plan['reference_frame']:continue
  image=Image.open(row['rgb']).convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
  with sam3_autocast_context('cuda',torch.bfloat16):state=processor.set_image(image)
  if a.policy=='joint':
   with sam3_autocast_context('cuda',torch.bfloat16):
    masks,quality,_=processor.model.predict_inst(state,point_coords=np.array(row['points'],dtype=np.float32),point_labels=np.ones(len(row['points']),dtype=np.int32),multimask_output=False,return_logits=True)
   logits=torch.from_numpy(np.asarray(masks[0]).copy()).float();xy=np.asarray(row['points']);accepted=bool((logits[xy[:,1],xy[:,0]]>0).all()) and float(quality[0])>.5
   details.append(dict(frame=frame,quality=float(quality[0]),accepted=accepted,points=row['points']))
  else:
   xy=np.asarray(row['candidate_points'],dtype=np.int32);weights=np.asarray(row['candidate_weights']);pending=np.ones(len(xy),dtype=bool);parts=[];trials=[]
   while pending.any():
    index=int(np.argmax(np.where(pending,weights,-1)));point=xy[index]
    with sam3_autocast_context('cuda',torch.bfloat16):
     masks,quality,_=processor.model.predict_inst(state,point_coords=point[None].astype(np.float32),point_labels=np.ones(1,dtype=np.int32),multimask_output=False,return_logits=True)
    local=torch.from_numpy(np.asarray(masks[0]).copy()).float();good=bool(local[point[1],point[0]]>0) and float(quality[0])>.5
    if good:
     parts.append(local);pending&=~(local[xy[:,1],xy[:,0]].numpy()>0)
    else:pending&=np.square(xy-point).sum(1)>64
    pending[index]=False;trials.append(dict(point=point.tolist(),accepted=good,quality=float(quality[0])))
   part_dir=base/'parts';part_dir.mkdir(parents=True,exist_ok=True)
   torch.save(dict(logits=torch.stack(parts).half() if parts else torch.empty(0,height,width),frame=frame,rgb=row['rgb'],trials=trials),part_dir/f'{frame}.pt')
   accepted=bool(parts);logits=torch.stack(parts).amax(0) if parts else torch.full((height,width),-10.)
   details.append(dict(frame=frame,accepted=accepted,parts=len(parts),candidate_points=len(xy),trials=trials))
  print(a.scene,frame,'accepted',accepted,'parts',details[-1].get('parts',1),'calls',len(details[-1].get('trials',[0])),flush=True)
  if not accepted:continue
  record=dict(logits=logits.half(),object_logit=1.)
  variants['replace'][frame]=record
  if frame in original['frames'] and original['frames'][frame]['object_logit']>0:
   old=F.interpolate(original['frames'][frame]['logits'].float()[None,None],size=(height,width),mode='bilinear',align_corners=False)[0,0];union=torch.maximum(old,logits)
  else:union=logits
  variants['union'][frame]=dict(logits=union.half(),object_logit=1.)
 for name,frames in variants.items():
  out=Path(a.base)/name/a.scene;out.mkdir(parents=True,exist_ok=True);torch.save(dict(frames=frames,reference_frame=plan['reference_frame']),out/'tracks.pt')
 (base/'decoding.json').write_text(json.dumps(dict(policy=a.policy,rows=details,seconds=time.monotonic()-start,reference_mask_unchanged=True,target_rgb_used=False,negative_prompt_used=False,new_negative_evidence='SAM pseudo-background, not verified absence',identity_not_guaranteed=True),indent=2));print(a.scene,'accepted',sum(r['accepted'] for r in details),'/',len(details),flush=True)
if __name__=='__main__':main()
