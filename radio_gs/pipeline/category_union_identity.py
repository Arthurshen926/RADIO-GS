"""Native identity of declared category unions, not isolated detected parts.

All permitted source views and all nonempty category unions are encoded. This
is additional source-assisted teacher work, not a query-independent map asset.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.querying.query_region_evidence import same_window
from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--representation',choices=['foreground','context'],default='foreground');a=p.parse_args()
 torch.set_num_threads(4);started=time.monotonic();scene=json.loads(Path(a.scene).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 first=torch.load(scene['source_evidence']['views'][0]['identity'],map_location='cpu');texts=first['queries']
 encoder,tokenizer=load_siglip_text(scene['text_encoder'],'cuda')
 tokens=tokenizer(texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
 query=F.normalize(F.normalize(encoder(**tokens).pooler_output.float(),dim=-1).half().float(),dim=-1);del encoder
 runtime=NativeSiglip2Runtime.load(Path(scene['text_encoder']),device=torch.device('cuda'))
 if a.representation=='context':
  from radio_gs.models.context_region_encoder import ContextRegionEncoder
  contextual=ContextRegionEncoder(runtime)
 modes=['foreground'] if a.representation=='foreground' else ['natural_global','context_region']
 paths={mode:out if mode=='foreground' else out/mode for mode in modes}
 for path in paths.values():path.mkdir(parents=True,exist_ok=True)
 count=0;records=[]
 for view in scene['source_evidence']['views']:
  data=torch.load(view['identity'],map_location='cpu');rgb=torch.from_numpy(np.array(Image.open(view['source_rgb']).convert('RGB'),copy=True)).permute(2,0,1).float()/255
  crops=[];ids=[];boxes=[];parts=[];memberships=[]
  for qi,row in enumerate(data['instances']):
   if not len(row['scores']):continue
   union=(row['masks']>.5).any(0)
   if not union.any():continue
   full=F.interpolate(union[None,None].float(),size=rgb.shape[1:],mode='nearest-exact')[0,0].bool()
   triple,box=same_window(rgb,full);crops.append(triple[1] if a.representation=='foreground' else triple[0]);ids.append(qi);boxes.append(box);parts.append(len(row['scores']))
   x0,y0,x1,y1=box
   memberships.append(F.interpolate(full[y0:y1,x0:x1][None,None].float(),size=(384,384),mode='area')[0,0])
  features={mode:torch.zeros(len(texts),query.shape[-1],dtype=torch.float16) for mode in modes};valid=torch.zeros(len(texts),dtype=torch.bool)
  for begin in range(0,len(crops),8):
   idx=ids[begin:begin+8];batch=torch.stack(crops[begin:begin+8])
   values=dict(foreground=runtime.encode(batch)) if a.representation=='foreground' else contextual.encode(batch,torch.stack(memberships[begin:begin+8]))
   for mode,value in values.items():features[mode][idx]=F.normalize(value.float(),dim=-1).half().cpu()
   valid[idx]=True
  for mode,feature in features.items():
   probability=(100*F.normalize(feature.float().cuda(),dim=-1)@query.T).softmax(-1).cpu()
   torch.save(dict(queries=texts,union_descriptors=feature,valid=valid,class_probability=probability,
     frame=view['frame'],foreground_only=mode=='foreground',representation=mode,crop_scale=1.5,category_union_not_instance=True,
     encoder=scene['text_encoder'],ground_truth_read=False),paths[mode]/f"frame_{view['frame']:05d}.pt")
  count+=len(ids);records.append(dict(frame=view['frame'],encodings=len(ids),category_indices=ids,boxes=boxes,parts=parts));print(scene['scene'],view['frame'],len(ids),flush=True)
 (out/'encoding.json').write_text(json.dumps(dict(scene=a.scene,source_views=len(records),new_native_image_encodings=count,
    new_text_encoder_calls=1,new_sam_calls=0,records=records,seconds=time.monotonic()-started,
    target_rgb_read=False,ground_truth_read=False,query_conditioned_asset=True,representations=modes,
    image_forwards_shared_between_outputs=a.representation=='context'),indent=2)+'\n')
if __name__=='__main__':main()
