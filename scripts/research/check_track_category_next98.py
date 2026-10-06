"""Reuse blind captions to reject gross category drift, not guessed materials."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;out=Path('output/category_identity_tracks_next98')/key;out.mkdir(parents=True,exist_ok=True)
 modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True);model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().cuda();letters=['A','B','C'];ids=[processor.tokenizer.encode(x,add_special_tokens=False)[0] for x in letters]
 rows=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())
 instruction='Check only broad object-category compatibility of independent visual descriptions. These descriptions may guess materials incorrectly, and different viewpoints may show partial surfaces. Do NOT reject a cup because one description says metal and another ceramic, or because color/shape details differ. A compatible visible part of the reference object is allowed. A: compatible broad object category or part. B: clearly incompatible object category (for example paper versus cup). C: insufficient category information. This does not establish physical instance identity. Answer exactly one letter: A, B, or C.'
 for row in rows:
  qi=row['query_index'];d=json.loads((Path('output/batched_description_identity_tracks_next98')/key/f'{qi}.json').read_text());t=load(Path('output/boundary_tracks_next95/high_merged/union')/key/f'{qi}.pt');kept={};judgments=[]
  for j in d['judgments']:
   prompt=instruction+' References: '+json.dumps(d['reference_descriptions'])+'. Candidate: '+json.dumps(j['description']);template=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True);inputs=processor(text=[template],return_tensors='pt').to('cuda');logits=model(**inputs).logits[0,-1].float()[ids];answer=letters[int(logits.argmax())];judgments.append(dict(j,strict_answer=j['answer'],answer=answer,logits=logits.cpu().tolist()))
   if answer=='A':kept.setdefault(j['frame'],set()).add(j['component'])
  for frame,r in t['frames'].items():
   if not r['present'] or frame==t['seed_frame']:continue
   components,_=label((r['probability']>.5).numpy(),structure=np.ones((3,3)));allow=np.isin(components,list(kept.get(frame,set())));r.update(positive_allowed=torch.from_numpy(allow),negative_allowed=torch.ones(allow.shape,dtype=torch.bool),scope_reliability=float(allow.any()))
  torch.save(t,out/f'{qi}.pt');d.update(instruction=instruction,identity_mode='category',judgments=judgments);(out/f'{qi}.json').write_text(json.dumps(d,indent=2)+'\n');print(key,qi,{x:sum(j['answer']==x for j in judgments) for x in letters},flush=True)
if __name__=='__main__':main()
