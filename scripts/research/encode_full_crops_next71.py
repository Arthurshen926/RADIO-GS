"""Native SigLIP full-image-window descriptors on the exact source region cohort."""
import json,argparse,time
from pathlib import Path
import torch,numpy as np
from PIL import Image
from torch.nn import functional as F
from radio_gs.querying.query_region_evidence import same_window
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 data=load(Path('output/taskwise_scope_composition_next38')/a.key/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());paths={r['frame_idx']:Path(src['image_dir'])/r['source_file'] for r in src['frames']};ep=load(root/'episodes.pt');teacher=load(root/info['teacher_files']['siglip']['path']);out=Path('output/full_crop_field_next71/evidence')/a.key;out.mkdir(parents=True,exist_ok=True)
 runtime=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'));features=torch.zeros_like(teacher['descriptors']);processed=[]
 for frame in info['train_frames']:
  ids=torch.where(ep['proposal_views']==info['frames'].index(frame))[0];rgb=torch.from_numpy(np.array(Image.open(paths[frame]).convert('RGB'),copy=True)).permute(2,0,1).float()/255;crops=[]
  for rid in ids:
   mask=F.interpolate((ep['masks'][rid]>.5)[None,None].float(),size=rgb.shape[1:],mode='nearest-exact')[0,0].bool();triple,_=same_window(rgb,mask);crops.append(triple[0])
  batch=torch.stack(crops);f=F.normalize(torch.cat([runtime.encode(x).float() for x in batch.split(8)]),dim=-1);features[ids]=f.cpu().to(features.dtype);processed.extend(ids.tolist());print(a.key,frame,len(ids),flush=True)
 torch.save(dict(descriptors=features,source_region_ids=processed,source_frames=info['train_frames'],encoder=teacher['encoder'],crop='unmasked original RGB within1.5x region bbox,384x384; same_window full branch',target_rgb_used=False),out/'teacher.pt')
 (out/'fit.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,source_frames=info['train_frames'],source_regions=len(processed),total_region_rows=len(features),unencoded_rows='zero; excluded from reconstruction',query_independent=True,target_rgb_used=False),indent=2)+'\n')
if __name__=='__main__':main()
