"""Matched crop control and object-erasure conditional evidence for query identity.
Freeze next83 candidate pairs. Erase only the candidate mask in the same1.5x
window. Query likelihood gain must be positive in both views; otherwise retain
the native seed. No target labels, query aliases or evaluation thresholds.
"""
import argparse,json,math
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration,BlipProcessor
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene
 relation=load(Path('output/region_relations_next79')/a.scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
 for e in manifest['entries']:
  nodes.extend((e['frame'],i,str(Path('output/source_coverage_next72')/a.scene/'native_regions'/f"frame_{e['frame']:05d}.pt")) for i in range(e['regions']))
 source=Path('output/source_coverage_next72')/a.scene;paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((source/'all.json').read_text())['images']}
 @lru_cache(maxsize=2)
 def masks(path):
  r=load(path);s=load(r['sam_path']);return r,unpack_masks(s['packed_masks'],s['mask_shape'][1])
 def crop(node):
  frame,ri,path=nodes[node];r,ms=masks(path);mask=ms[int(r['native_proposal_indices'][ri])].copy();im=Image.open(paths[frame]).convert('RGB');m=np.array(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(m);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2;bw=(xx.max()-xx.min()+1)*1.5;bh=(yy.max()-yy.min()+1)*1.5;x0=max(0,math.floor(cx-bw/2));x1=min(im.width,math.ceil(cx+bw/2));y0=max(0,math.floor(cy-bh/2));y1=min(im.height,math.ceil(cy+bh/2));rgb=np.array(im)[y0:y1,x0:x1];erased=rgb.copy();erased[m[y0:y1,x0:x1]]=127
  return Image.fromarray(rgb),Image.fromarray(erased),mask
 modelid='Salesforce/blip-image-captioning-base';processor=BlipProcessor.from_pretrained(modelid,local_files_only=True);model=BlipForConditionalGeneration.from_pretrained(modelid,local_files_only=True).eval().half().cuda();rows=json.loads((Path('output/content_seeds_next83')/key/'proposals.json').read_text());reports={arm:[] for arm in ['raw','contrast']}
 for row in rows:
  qi=row['query_index'];query=row['query'];candidates=row['candidates'];images=[]
  for c in candidates:
   for node in c['nodes']:images.extend(crop(node)[:2])
  likelihood=[]
  for start in range(0,len(images),8):
   batch=images[start:start+8];inp=processor(images=batch,text=[query]*len(batch),return_tensors='pt',padding=True).to('cuda');inp['pixel_values']=inp['pixel_values'].half();inp['input_ids'][:,0]=model.config.text_config.bos_token_id;logits=model(**inp).logits[:,:-1].float();ids=inp['input_ids'][:,1:];valid=inp['attention_mask'][:,1:].float();ll=logits.log_softmax(-1).gather(-1,ids[...,None])[...,0];likelihood.extend(((ll*valid).sum(1)/valid.sum(1)).cpu().tolist())
  values=torch.tensor(likelihood).reshape(-1,2,2);gain=values[:,:,0]-values[:,:,1]
  for arm in reports:
   view_scores=values[:,:,0] if arm=='raw' else gain;rank=view_scores.amin(1);winner=int(rank.argmax());fallback=arm=='contrast' and float(rank[winner])<=0
   if fallback:result=load(Path('output/region_seeds_next80')/key/f'{qi}.pt');node=None
   else:
    c=candidates[winner];side=int(view_scores[winner].argmax());node=c['nodes'][side];_,_,mask=crop(node);result=dict(query=query,query_index=qi,seed_frame=nodes[node][0],seed_confidence=max(0.,c['native_score']),masks=torch.from_numpy(mask)[None].bool(),pair_nodes=c['nodes'],pair_frames=c['frames'],single_scope_hypothesis=True)
   result=dict(result,source='same-window BLIP likelihood' if arm=='raw' else 'object-erasure BLIP evidence gain with native fallback',verifier=modelid,native_fallback=fallback,selected_score=float(rank[winner]))
   out=Path('output/counterfactual_seeds_next84')/arm/key;out.mkdir(parents=True,exist_ok=True);torch.save(result,out/f'{qi}.pt')
   if node is not None:
    full,erased,_=crop(node);full.save(out/f'{qi}_chosen.jpg');erased.save(out/f'{qi}_erased.jpg')
   reports[arm].append(dict(query=query,query_index=qi,seed_frame=result['seed_frame'],selected_rank=None if fallback else winner,native_fallback=fallback,selected_score=float(rank[winner]),candidate_full_likelihood=values[:,:,0].tolist(),candidate_erased_likelihood=values[:,:,1].tolist(),candidate_gain=gain.tolist()))
   print(arm,query,'native fallback' if fallback else winner,float(rank[winner]),flush=True)
 for arm,report in reports.items():
  out=Path('output/counterfactual_seeds_next84')/arm/key;(out/'proposals.json').write_text(json.dumps(report,indent=2)+'\n');(out/'contract.json').write_text(json.dumps(dict(candidate_source='output/content_seeds_next83/'+key,context_scale=1.5,erasure_fill=127,score=arm,positive_gain_in_both_views=arm=='contrast',native_fallback=arm=='contrast',target_gt_used=False,verbatim_query=True),indent=2)+'\n')
if __name__=='__main__':main()
