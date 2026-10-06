"""Recognize query-independent source regions without requiring text detection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from radio_gs.querying.query_region_evidence import same_window
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 base=Path('output/taskwise_scope_composition_next38')/a.key;data=load(base/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());paths={r['frame_idx']:Path(src['image_dir'])/r['source_file'] for r in src['frames']}
 teacher=load(root/info['teacher_files']['siglip']['path']);features=F.normalize(teacher['descriptors'].float().cuda(),dim=-1)
 q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float().cuda(),dim=-1)
 ep=load(root/'episodes.pt');masks=ep['masks'];views=ep['proposal_views'];del ep
 runtime=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'))
 out=Path('output/native_proposals_next69/evidence')/a.key;out.mkdir(parents=True,exist_ok=True);summary=[]
 for frame in info['train_frames']:
  original=load(Path('output/typed_evidence_next44')/a.key/f'identity_{frame}.pt');ids=torch.where(views==info['frames'].index(frame))[0];cos=features[ids.cuda()]@q.T
  best=cos.argmax(0);chosen=ids[best.cpu()];unique=chosen.unique();encoded={}
  rgb=torch.from_numpy(np.array(Image.open(paths[frame]).convert('RGB'),copy=True)).permute(2,0,1).float()/255
  crops=[]
  for rid in unique:
   mask=F.interpolate((masks[rid]>.5)[None,None].float(),size=rgb.shape[1:],mode='nearest-exact')[0,0].bool();triple,_=same_window(rgb,mask);crops.append(triple)
  batch=torch.cat(crops);emb=F.normalize(torch.cat([runtime.encode(x).float() for x in batch.split(8)]),dim=-1).reshape(-1,3,q.shape[-1])
  for rid,vector in zip(unique.tolist(),emb):encoded[rid]=vector
  records=[]
  for qi,rid in enumerate(chosen.tolist()):
   f=encoded[rid];margin=(f[1]-f[2])@q[qi];score=cos[best[qi],qi].clamp(0,1)
   if margin>0:
    records.append(dict(masks=masks[rid:rid+1].half(),scores=score[None].cpu(),foreground_margin=margin[None].cpu(),proposal_id=rid))
   else:records.append(dict(masks=masks[:0].half(),scores=torch.empty(0),foreground_margin=torch.empty(0),proposal_id=rid))
  torch.save(dict(frame=frame,queries=original['queries'],instances=records),out/f'identity_{frame}.pt');summary.append(dict(frame=frame,accepted=[t for t,r in zip(original['queries'],records) if len(r['scores'])],source_rgb=str(paths[frame])))
  print(a.key,frame,len(summary[-1]['accepted']),flush=True)
 (out/'fit.json').write_text(json.dumps(dict(source_frames=info['train_frames'],seconds=time.monotonic()-start,source_region_choice='highest native SigLIP cosine per query, then positive same-window foreground minus complement margin',target_rgb_used=False,verified_identity=False,extra_encoder_compute=True,records=summary),indent=2)+'\n')
if __name__=='__main__':main()
