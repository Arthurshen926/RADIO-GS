"""Require reference-image identity evidence before admitting new source parts."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from radio_gs.querying.query_region_evidence import same_window
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic();base=Path('output/reference_cover_next74')/a.scene;plan=json.loads((base/'anchors.json').read_text());w,h=plan['raster_wh'];data=load(Path('output/taskwise_scope_composition_next38')/('nvos_'+('horns_center' if a.scene=='horns_left' else a.scene))/'field.pt');root=Path(data['root']);teacher=load(root/'teacher_siglip.pt');runtime=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'))
 report=json.loads((Path('output/nvos_reference_completion_next19')/a.scene/'report.json').read_text());qr=json.loads((Path(report['query_root'])/a.scene/'receipt.json').read_text());qrgb=Image.open(qr['image_query_crop']['path']).convert('RGB').resize((384,384),Image.Resampling.BICUBIC);q=runtime.encode(torch.from_numpy(np.array(qrgb,copy=True)).permute(2,0,1).float()[None]/255).float();q=F.normalize(q,dim=-1)[0]
 original=load(Path('output/typed_evidence_next44/full_source_tracks')/a.scene/'tracks.pt');frames=dict(original['frames']);records=[]
 for row in plan['views']:
  frame=row['frame'];path=base/'parts'/f'{frame}.pt'
  if not path.exists():continue
  parts=load(path)['logits'].float();rgb=torch.from_numpy(np.array(Image.open(row['rgb']).convert('RGB'),copy=True)).permute(2,0,1).float()/255;crops=[];indices=[]
  for i,part in enumerate(parts):
   mask=F.interpolate((part>0).float()[None,None],size=rgb.shape[1:],mode='nearest-exact')[0,0].bool()
   if not mask.any():continue
   triple,_=same_window(rgb,mask);crops.append(triple);indices.append(i)
  if not crops:continue
  batch=torch.cat(crops);f=torch.cat([runtime.encode(x).float() for x in batch.split(8)]).reshape(-1,3,len(q));f=F.normalize(f,dim=-1);margin=(f[:,1]@q-f[:,2]@q).cpu();keep=[i for i,m in zip(indices,margin) if m>0];records.append(dict(frame=frame,parts=len(parts),admitted=len(keep),margins=margin.tolist()))
  if keep:
   value=parts[keep].amax(0)
   if frame in original['frames'] and original['frames'][frame]['object_logit']>0:
    old=F.interpolate(original['frames'][frame]['logits'].float()[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0];value=torch.maximum(value,old)
   frames[frame]=dict(logits=value.half(),object_logit=1.)
  print(a.scene,frame,len(keep),'/',len(parts),flush=True)
 out=Path('output/reference_identity_next74')/a.scene;out.mkdir(parents=True,exist_ok=True);torch.save(dict(frames=frames,reference_frame=plan['reference_frame']),out/'tracks.pt');(out/'identity.json').write_text(json.dumps(dict(records=records,seconds=time.monotonic()-start,query_crop=qr['image_query_crop']['path'],criterion='reference-image cosine foreground greater than same-window context',target_rgb_used=False,verified_identity=False,base_video_retained=True),indent=2))
if __name__=='__main__':main()
