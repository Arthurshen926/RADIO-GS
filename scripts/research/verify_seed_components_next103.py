"""Describe seed components without query, then reject explicit unrelated content.
Retain uncertain/disconnected parts; no largest-area selection or target GT.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from scripts.research.verify_track_identity_next98 import crop
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;out=Path('output/component_seeds_next103')/key;out.mkdir(parents=True,exist_ok=True);tracks=Path('output/component_tracks_next103')/key;tracks.mkdir(parents=True,exist_ok=True)
 paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/a.scene/'all.json').read_text())['images']};rows=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text());modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True,min_pixels=56*56,max_pixels=448*448);model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().cuda();letters=['A','B','C'];ids=[processor.tokenizer.encode(x,add_special_tokens=False)[0] for x in letters];report=[]
 for row in rows:
  qi=row['query_index'];s=load(Path('output/grounded_seeds_next88')/key/f'{qi}.pt');new=[];judgments=[]
  for mi,mask in enumerate(s['masks']):
   components,n=label(mask.numpy(),structure=np.ones((3,3)));keep=np.zeros(mask.shape,dtype=bool)
   for ci in range(1,n+1):
    m=components==ci
    # Connected seeds require no extra admission decision.
    if n==1:keep|=m;continue
    im=crop(paths[s['seed_frame']],m);prompt='Describe only the visible content inside the red outline. State the object category or visible part. If too small or unclear, say unclear. Do not guess the intended target. Ignore nearby content outside the outline.';template=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True);inputs=processor(text=[template],images=[im],return_tensors='pt').to('cuda');tokens=model.generate(**inputs,max_new_tokens=48,do_sample=False);description=processor.tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    prompt='A retrieval request is '+json.dumps(s['query'])+'. An independently described source-mask component is '+json.dumps(description)+'. Classify its relationship to the request. A: the requested object, a compatible visible part, or another instance of the requested category. B: explicitly a different, unrelated object category. C: unclear or insufficient information. Do not reject because of guessed color/material, small size, disconnection, or uncertainty about a named character. A whole object that could contain the requested part is uncertain (C), not B. Answer exactly A, B, or C.';template=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True);inputs=processor(text=[template],return_tensors='pt').to('cuda');logits=model(**inputs).logits[0,-1].float()[ids];answer=letters[int(logits.argmax())]
    if answer!='B':keep|=m
    judgments.append(dict(mask=mi,component=ci,pixels=int(m.sum()),description=description,answer=answer,logits=logits.cpu().tolist()))
   new.append(torch.from_numpy(keep))
  masks=torch.stack(new);fallback=not bool(masks.any())
  if fallback:masks=s['masks']
  changed=not torch.equal(masks,s['masks']);s.update(masks=masks,source='query-blind component caption then category/part rejection; uncertain retained; no area ranking');torch.save(s,out/f'{qi}.pt')
  if not changed:shutil.copyfile(Path('output/boundary_tracks_next95/high_primary')/key/f'{qi}.pt',tracks/f'{qi}.pt')
  d=dict(query=s['query'],query_index=qi,changed=changed,all_rejected_fallback=fallback,judgments=judgments);report.append(d);(out/f'{qi}.json').write_text(json.dumps(d,indent=2)+'\n');print(s['query'],changed,[(j['pixels'],j['answer']) for j in judgments],flush=True)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
