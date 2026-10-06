"""Matched source-set discovery: text, one extent box, separate component examples.
All arms share frozen source anchors. Discovery never changes the native field.
The caller must use these unions only for category/set queries, not instance prompts.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scipy.ndimage import label
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,sam3_autocast_context
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--track-base',default='output/core_merged_next104/union');p.add_argument('--query-indices',type=int,nargs='*');p.add_argument('--output-base',default='output/set_discovery_next107');a=p.parse_args();torch.set_num_threads(4);load=lambda p:torch.load(p,map_location='cpu',weights_only=False);key='lerf_'+a.scene
ids=a.query_indices if a.query_indices is not None else [r['query_index'] for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())];tracks={qi:load(Path(a.track_base)/key/f'{qi}.pt') for qi in ids};arms=['text','box','components'];outputs={arm:copy.deepcopy(tracks) for arm in arms};reports={arm:{qi:[] for qi in ids} for arm in arms};authority=json.loads((Path('output/source_coverage_next72')/a.scene/'all.json').read_text());paths={int(r['image_id'].split('_')[-1]):r['path'] for r in authority['images']};frames=sorted({f for t in tracks.values() for f in t['frames']});processor=_load_sam3_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',device='cuda',confidence_threshold=.5,dtype='float32',resolution=1008);start=time.monotonic()
def box(m):
 y,x=np.where(m);h,w=m.shape;return [(x.min()+x.max()+1)/(2*w),(y.min()+y.max()+1)/(2*h),(x.max()-x.min()+1)/w,(y.max()-y.min()+1)/h]
with torch.inference_mode(),sam3_autocast_context('cuda',torch.bfloat16):
 for frame in frames:
  active=[qi for qi,t in tracks.items() if frame in t['frames'] and t['frames'][frame]['present'] and (t['frames'][frame]['probability']>.5).any()]
  if not active:continue
  state=processor.set_image(Image.open(paths[frame]).convert('RGB'))
  for qi in active:
   core=tracks[qi]['frames'][frame]['probability'].float();m=core>.5;h,w=m.shape;ls,nc=label(m.numpy(),structure=np.ones((3,3)));interior=(-F.max_pool2d(-m.float()[None,None],3,1,1)[0,0]>.5).numpy();boxes=[box(ls==i) for i in range(1,nc+1) if ((ls==i)&interior).any()]
   for arm in arms:
    processor.reset_all_prompts(state);det=processor.set_text_prompt(tracks[qi]['query'],state);examples=[] if arm=='text' else [box(m.numpy())] if arm=='box' else boxes
    for b in examples:det=processor.add_geometric_prompt(b,True,state)
    scores=det['scores'].float().cpu();n=len(scores)
    if n:
     probs=F.interpolate(det['masks_logits'].float().reshape(n,1,*det['masks_logits'].shape[-2:]),size=(h,w),mode='bilinear',align_corners=False)[:,0].cpu();new=torch.maximum(core,probs.amax(0))
    else:probs=torch.empty(0,h,w);new=core
    outputs[arm][qi]['frames'][frame].update(probability=new.half(),present=True,detector_instances=probs.half(),detector_scores=scores,example_boxes=examples)
    reports[arm][qi].append(dict(frame=frame,examples=len(examples),instances=n,core_pixels=int(m.sum()),union_pixels=int((new>.5).sum()),added_pixels=int(((new>.5)&~m).sum())))
  print(key,frame,len(active),round(time.monotonic()-start,1),flush=True)
for arm in arms:
 out=Path(a.output_base)/arm/key;out.mkdir(parents=True,exist_ok=True)
 for qi,t in outputs[arm].items():
  t.update(source=f'exact-text category-set discovery, {arm} examples, preserve tracked core',discovery_source=a.track_base);torch.save(t,out/f'{qi}.pt');(out/f'{qi}.json').write_text(json.dumps(dict(query=t['query'],query_index=qi,arm=arm,rows=reports[arm][qi]),indent=2)+'\n')
 (out/'contract.json').write_text(json.dumps(dict(source_frames=frames,query_indices=ids,anchor_base=a.track_base,arm=arm,threshold=.5,component_eligibility='8-connected original core components containing at least one 3x3 eroded positive pixel; all eligible components used',no_recursive_expansion=True,full_source_rgb=True,target_rgb_or_gt_used=False,exact_runtime_text=True,native_scores_unchanged=True,seconds=time.monotonic()-start),indent=2)+'\n')
