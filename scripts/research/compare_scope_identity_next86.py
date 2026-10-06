"""Joint competing-hypothesis identity decisions on source RGB only."""
import argparse,json,math
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
 p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',required=True);p.add_argument('--secondary-base',default='output/multi_seeds_next86');p.add_argument('--primary-base',default='output/grounded_seeds_next85');p.add_argument('--output-base',default='output/next86');a=p.parse_args();torch.set_num_threads(4)
 modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True,min_pixels=56*56,max_pixels=448*448);model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().cuda();labels=['A','B','C','D','E'];ids=[processor.tokenizer.encode(x,add_special_tokens=False)[0] for x in labels]
 instruction='These are TWO different proposed regions, each marked with a red outline. Requested target: {query}. Compare their identity and requested whole-object or part extent. Sharing a broad category or being related to the target is not sufficient. Which outlined regions actually match the exact requested target? A: only the FIRST. B: only the SECOND. C: BOTH independently match. D: NEITHER matches. E: insufficient visual evidence or knowledge. Do not assume either proposal is correct. Answer exactly one letter: A, B, C, D, or E.'
 for scene in a.scenes:
  key='lerf_'+scene;rows=json.loads((Path(a.secondary_base)/key/'proposals.json').read_text());original={r['query_index']:r for r in json.loads((Path(a.primary_base)/key/'proposals.json').read_text())};relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
  for e in manifest['entries']:nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
  paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/scene/'all.json').read_text())['images']}
  @lru_cache(maxsize=4)
  def masks(path):
   b=load(path);r=load(b['source']);s=load(r['sam_path']);return r,unpack_masks(s['packed_masks'],s['mask_shape'][1])
  def region(node):
   frame,ri,path=nodes[node];r,ms=masks(path);mask=ms[int(r['native_proposal_indices'][ri])];im=Image.open(paths[frame]).convert('RGB');m=np.array(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(m);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2;bw=(xx.max()-xx.min()+1)*1.5;bh=(yy.max()-yy.min()+1)*1.5;box=(max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),min(im.width,math.ceil(cx+bw/2)),min(im.height,math.ceil(cy+bh/2)));crop=im.crop(box);x0,y0,x1,y1=box;scale=448/max(crop.size);size=(max(1,round(crop.width*scale)),max(1,round(crop.height*scale)));crop=crop.resize(size,Image.Resampling.BICUBIC);cm=torch.from_numpy(np.array(Image.fromarray(m[y0:y1,x0:x1]).resize(size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(cm,3,1,1)+F.max_pool2d(-cm,3,1,1))[0,0]>.5;rgb=np.array(crop);rgb[edge.numpy()]=[255,0,0];return Image.fromarray(rgb)
  results=[]
  for row in rows:
   if row['secondary_rank'] is None:continue
   candidates=original[row['query_index']]['candidates'];first=candidates[row['primary_rank']]['nodes'];second=candidates[row['secondary_rank']]['nodes'];judgments=[]
   # Swap order as well as view; admission must survive both presentations.
   for view in [0,1]:
    pair=[region(first[view]),region(second[view])]
    if view==1:pair.reverse()
    prompt=instruction.format(query=json.dumps(row['query']));messages=[dict(role='user',content=[dict(type='image'),dict(type='image'),dict(type='text',text=prompt)])];text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=processor(text=[text],images=pair,return_tensors='pt').to('cuda');logits=model(**inputs).logits[0,-1].float();values=logits[ids];judgments.append(dict(view=view,order='primary_secondary' if view==0 else 'secondary_primary',answer=labels[int(values.argmax())],logits=values.cpu().tolist(),raw_top=processor.tokenizer.convert_ids_to_tokens(int(logits.argmax()))))
   accepted=all(j['answer']=='C' for j in judgments);results.append(dict(**row,comparison=judgments,secondary_admitted=accepted));print(scene,row['query'],[j['answer'] for j in judgments],accepted,flush=True)
  out=Path(a.output_base)/key;out.mkdir(parents=True,exist_ok=True);(out/'joint_identity.json').write_text(json.dumps(dict(prompt=instruction,model=modelid,revision=model.config._commit_hash,admission='BOTH in two distinct view comparisons with reversed candidate order',target_rgb_used=False,results=results),indent=2)+'\n')
if __name__=='__main__':main()
