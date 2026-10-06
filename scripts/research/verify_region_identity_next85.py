"""Instruction-grounded referent/scope classification on fixed source candidates.
Optional Qwen provider; native fallback if no pair has A over both B and C.
No GT, query aliases, generated supervision or model-training vocabulary.
"""
import argparse,json,math,time
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--candidate-base',default='output/content_seeds_next83');p.add_argument('--output-base',default='output/grounded_seeds_next85');a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;start=time.monotonic()
 relation=load(Path('output/region_relations_next79')/a.scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
 for e in manifest['entries']:nodes.extend((e['frame'],i,str(Path('output/source_coverage_next72')/a.scene/'native_regions'/f"frame_{e['frame']:05d}.pt")) for i in range(e['regions']))
 paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/a.scene/'all.json').read_text())['images']}
 @lru_cache(maxsize=2)
 def masks(path):
  r=load(path);s=load(r['sam_path']);return r,unpack_masks(s['packed_masks'],s['mask_shape'][1])
 def region(node):
  frame,ri,path=nodes[node];r,ms=masks(path);mask=ms[int(r['native_proposal_indices'][ri])].copy();im=Image.open(paths[frame]).convert('RGB');m=np.array(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(m);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2;bw=(xx.max()-xx.min()+1)*1.5;bh=(yy.max()-yy.min()+1)*1.5;box=(max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),min(im.width,math.ceil(cx+bw/2)),min(im.height,math.ceil(cy+bh/2)));crop=im.crop(box);x0,y0,x1,y1=box
  scale=448/max(crop.size);size=(max(1,round(crop.width*scale)),max(1,round(crop.height*scale)));crop=crop.resize(size,Image.Resampling.BICUBIC);cm=torch.from_numpy(np.array(Image.fromarray(m[y0:y1,x0:x1]).resize(size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(cm,3,1,1)+F.max_pool2d(-cm,3,1,1))[0,0]>.5;marked=np.array(crop);marked[edge.numpy()]=[255,0,0]
  return Image.fromarray(marked),crop,mask
 modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True,min_pixels=56*56,max_pixels=448*448);processor.tokenizer.padding_side='left';model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().cuda();label_ids=[processor.tokenizer.encode(x,add_special_tokens=False)[0] for x in ['A','B','C']]
 instruction='The red outline marks a proposed image region. Judge only the outlined region, using surrounding pixels only as context. Requested target: {query}\nChoose A if the region depicts the exact requested target with the appropriate whole-object or part extent. Choose B for a different or merely associated object, or a wrong whole/part extent. Choose C if resolution, visibility, or your knowledge is insufficient. Do not assume a candidate must match. Answer exactly one letter: A, B, or C.'
 rows=json.loads((Path(a.candidate_base)/key/'proposals.json').read_text());out=Path(a.output_base)/key;out.mkdir(parents=True,exist_ok=True);report=[]
 for row in rows:
  qi=row['query_index'];query=row['query'];candidates=row['candidates'];images=[region(node)[0] for c in candidates for node in c['nodes']];prompt=instruction.format(query=json.dumps(query,ensure_ascii=False));logits=[];tops=[]
  for offset in range(0,len(images),4):
   batch=images[offset:offset+4];messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])];text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=processor(text=[text]*len(batch),images=batch,padding=True,return_tensors='pt').to('cuda');last=model(**inputs).logits[:,-1].float();logits.extend(last[:,label_ids].cpu().tolist());tops.extend(processor.tokenizer.convert_ids_to_tokens(last.argmax(-1).cpu().tolist()))
  values=torch.tensor(logits).reshape(-1,2,3);margin=values[:,:,0]-values[:,:,1:].amax(2);rank=margin.amin(1);winner=int(rank.argmax());fallback=float(rank[winner])<=0
  if fallback:
   seed=load(Path('output/region_seeds_next80')/key/f'{qi}.pt');node=None
  else:
   c=candidates[winner];side=int(margin[winner].argmax());node=c['nodes'][side];marked,crop,mask=region(node);seed=dict(query=query,query_index=qi,seed_frame=nodes[node][0],seed_confidence=max(0.,c['native_score']),masks=torch.from_numpy(mask)[None].bool(),pair_nodes=c['nodes'],pair_frames=c['frames'],single_scope_hypothesis=True);marked.save(out/f'{qi}_marked.jpg');crop.save(out/f'{qi}_chosen.jpg')
  seed=dict(seed,source='instruction-grounded source referent and scope classification',verifier=modelid,native_fallback=fallback,grounded_margin=float(rank[winner]));torch.save(seed,out/f'{qi}.pt');report.append(dict(query=query,query_index=qi,seed_frame=seed['seed_frame'],selected_rank=None if fallback else winner,native_fallback=fallback,selected_margin=float(rank[winner]),candidate_class_logits=values.tolist(),candidate_margin=margin.tolist(),raw_top_tokens=tops,candidates=candidates));print(query,'fallback' if fallback else winner,float(rank[winner]),flush=True)
 (out/'proposals.json').write_text(json.dumps(report,indent=2)+'\n');(out/'contract.json').write_text(json.dumps(dict(model=modelid,model_commit=model.config._commit_hash,prompt=instruction,classes=['exact referent and scope','wrong referent or scope','insufficient evidence'],label_token_ids=label_ids,score='minimum A-max(B,C) over two views',native_fallback='no pair with positive margin',candidate_source=a.candidate_base+'/'+key,context_scale=1.5,max_image_pixels=448*448,target_rgb_or_gt_used=False,extra_inference_teacher=True,seconds=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
