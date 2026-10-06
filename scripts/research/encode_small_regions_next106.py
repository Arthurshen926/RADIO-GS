"""Bounded small-region complement: preserve original masked-crop encoder, remove relative area floor for masks with >=64 source pixels."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--cohort',choices=['old','coverage64','all'],default='all');a=p.parse_args();torch.set_num_threads(4)
 out=Path('output/source_coverage_next72')/a.scene;authority=json.loads((out/'all.json').read_text());paths={int(r['image_id'].split('_')[-1]):r['path'] for r in authority['images']};frames=json.loads((out/'cohorts.json').read_text())['frames'][a.cohort]
 data=load(Path('output/taskwise_scope_composition_next38')/('lerf_'+a.scene)/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());teacher=load(root/info['teacher_files']['siglip']['path']);runtime=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'));dest=Path('output/small_regions_next106')/a.scene;dest.mkdir(exist_ok=True,parents=True);records=[]
 for frame in frames:
  path=dest/f'frame_{frame:05d}.pt'
  if path.exists():continue
  source=out/'source_sam'/f'frame_{frame:05d}.pt';sam=load(source);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);rgb=np.array(Image.open(paths[frame]).convert('RGB'));h,w=rgb.shape[:2];crops=[];ids=[]
  for j,m in enumerate(masks):
   # Same eligibility and crop recipe as original teacher, evaluated on native support.
   area=float(m.mean())
   if not (area<.002 and int(m.sum())>=64):continue
   ys,xs=np.where(m);x0,x1=max(0,int(xs.min())-3),min(w,int(xs.max())+4);y0,y1=max(0,int(ys.min())-3),min(h,int(ys.max())+4);crop=rgb[y0:y1,x0:x1].copy();crop[~m[y0:y1,x0:x1]]=127;crop=Image.fromarray(crop).resize((384,384),Image.Resampling.BICUBIC);crops.append(torch.from_numpy(np.array(crop,copy=True)).permute(2,0,1).float()/255);ids.append(j)
  features=F.normalize(torch.cat([runtime.encode(torch.stack(crops[i:i+8])).float().cpu() for i in range(0,len(crops),8)]),dim=-1) if crops else torch.empty(0,1536)
  torch.save(dict(frame=frame,descriptors=features.half(),native_proposal_indices=torch.tensor(ids),sam_path=str(source),native_mask_hw=[h,w],encoder=teacher['encoder'],crop='masked RGB bbox+3pixels, gray127 background,384',query_independent=True),path);records.append(dict(frame=frame,regions=len(ids)));print(a.scene,frame,len(ids),flush=True)
 (dest/(a.cohort+'.json')).write_text(json.dumps(dict(requested_frames=frames,encoded_frames=[f for f in frames if (dest/f'frame_{f:05d}.pt').exists()],records=records),indent=2)+'\n')
if __name__=='__main__':main()
