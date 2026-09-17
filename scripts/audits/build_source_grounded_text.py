"""Native SAM3 text grounding on training RGB and the existing fixed vocabulary."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,set_requested_cuda_device,sam3_autocast_context
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/sam3_modelscope/sam3.pt'));a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
 info=json.loads((a.root/'teachers.json').read_text());source=json.loads((a.root/'source_frames.json').read_text());auth=json.loads((a.root/'responsibility.json').read_text())
 if sha256_file(a.root/'responsibility.json')!=info['authority_sha256']:raise ValueError('geometry authority mismatch')
 rec=info['teacher_files']['siglip'];path=a.root/rec['path']
 if sha256_file(path)!=rec['sha256']:raise ValueError('vocabulary source mismatch')
 queries=torch.load(path,weights_only=False)['queries'];words=[q.removeprefix('a photo of a ') for q in queries]
 h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
 frames=info['train_frames'];src={x['frame_idx']:x for x in source['frames']}
 metadata={'schema':'source_grounded_text.v1','source_only':True,'frames':frames,'queries':queries,'grounding_phrases':words,'teacher_manifest_sha256':sha256_file(a.root/'teachers.json'),'geometry_binding':auth['metadata']['frozen_geometry_binding'],'checkpoint_sha256':sha256_file(a.checkpoint),'confidence_threshold':.7,'mask_probability_threshold':.5,'resolution':1008,'dtype':'float32_amp_bfloat16','benchmark_labels_used':False,'implementation_sha256':sha256_file(__file__)}
 protocol=a.output.with_suffix('.protocol.json')
 if protocol.exists():raise FileExistsError(protocol)
 protocol.write_text(json.dumps(metadata,indent=2)+'\n')
 set_requested_cuda_device(a.device)
 model=_load_sam3_model(checkpoint_path=str(a.checkpoint),device=a.device,confidence_threshold=.7,dtype='float32',resolution=1008,build_on_cpu=True)
 masks=[];scores=[];source_records=[]
 with torch.inference_mode():
  for frame in frames:
   r=src[frame];image=Path(source['image_dir'])/r['source_file']
   if sha256_file(image)!=r['source_sha256']:raise ValueError('source RGB identity')
   with sam3_autocast_context(a.device,torch.bfloat16):state=model.set_image(Image.open(image).convert('RGB'))
   frame_masks=[];frame_scores=[]
   for i,word in enumerate(words):
    model.reset_all_prompts(state)
    with sam3_autocast_context(a.device,torch.bfloat16):out=model.set_text_prompt(word,state)
    probability=out['masks_logits'] # Official API already returns sigmoid probabilities.
    if len(probability):
     union=probability.float().amax(0,keepdim=True)
     small=F.interpolate(union,size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu()
     confidence=float(out['scores'].max())
    else:small=torch.zeros(h,w);confidence=0.
    frame_masks.append(small.half());frame_scores.append(confidence)
   masks.append(torch.stack(frame_masks));scores.append(frame_scores);source_records.append({'frame':frame,'sha256':r['source_sha256']})
   print('frame',frame,'nonempty',sum(bool((m>.5).any()) for m in frame_masks),flush=True)
 metadata.update(probability=torch.stack(masks),scores=torch.tensor(scores),source_records=source_records,protocol_sha256=sha256_file(protocol))
 torch.save(metadata,a.output)
 a.output.with_suffix('.json').write_text(json.dumps({'schema':metadata['schema'],'sha256':sha256_file(a.output),'frames':frames,'teacher_manifest_sha256':metadata['teacher_manifest_sha256'],'checkpoint_sha256':metadata['checkpoint_sha256'],'source_records':source_records,'protocol_sha256':metadata['protocol_sha256'],'nonempty_query_views':int((metadata['probability']>.5).flatten(2).any(2).sum())},indent=2)+'\n')

if __name__=='__main__':main()
