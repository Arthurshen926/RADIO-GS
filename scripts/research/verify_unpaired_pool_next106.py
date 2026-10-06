"""Source-only probe of area eligibility and pair admission, without scope fitting.
Same encoder, same exact-query prompt; inspect union of old/new top64 regions.
A single-view A label is a diagnostic, not verified physical correspondence.
"""
import argparse,json,math
from pathlib import Path
from functools import lru_cache
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query-index',required=True,type=int);a=p.parse_args();torch.set_num_threads(4)
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
scene=a.scene;key='lerf_'+scene;qi=a.query_index;nodes=[];features=[];source=Path('output/source_coverage_next72')/scene;frames=json.loads((source/'cohorts.json').read_text())['frames']['all'];paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((source/'all.json').read_text())['images']}
for small in [False,True]:
 for f in frames:
  path=(Path('output/small_regions_next106')/scene if small else source/'native_regions')/f'frame_{f:05d}.pt';r=load(path);features.append(F.normalize(r['descriptors'].float(),dim=-1));nodes.extend(dict(frame=f,local_index=i,path=str(path),small=small) for i in range(len(r['descriptors'])))
 if not small:old_count=len(nodes)
f=load(Path('output/native_rebuild_next54/mixture_masscal')/key/'field.pt');root=Path(f['root']);q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float(),dim=-1);scores=torch.cat(features)@q[qi];old_rank=scores[:old_count].argsort(descending=True).tolist();new_rank=scores.argsort(descending=True).tolist();ranks={n:i+1 for i,n in enumerate(new_rank)};old_ranks={n:i+1 for i,n in enumerate(old_rank)};chosen=list(dict.fromkeys(old_rank[:64]+new_rank[:64]));old=next(r for r in json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text()) if r['query_index']==qi);query=old['query']
@lru_cache(maxsize=4)
def masks(path):
 r=load(path);s=load(r['sam_path']);return r,unpack_masks(s['packed_masks'],s['mask_shape'][1])
def region(node):
 n=nodes[node];r,ms=masks(n['path']);mask=ms[int(r['native_proposal_indices'][n['local_index']])].copy();im=Image.open(paths[n['frame']]).convert('RGB');m=np.array(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));yy,xx=np.where(m);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2;bw=(xx.max()-xx.min()+1)*1.5;bh=(yy.max()-yy.min()+1)*1.5;box=(max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),min(im.width,math.ceil(cx+bw/2)),min(im.height,math.ceil(cy+bh/2)));im=im.crop(box);x0,y0,x1,y1=box;scale=448/max(im.size);size=(max(1,round(im.width*scale)),max(1,round(im.height*scale)));im=im.resize(size,Image.Resampling.BICUBIC);cm=torch.from_numpy(np.array(Image.fromarray(m[y0:y1,x0:x1]).resize(size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(cm,3,1,1)+F.max_pool2d(-cm,3,1,1))[0,0]>.5;rgb=np.array(im);rgb[edge.numpy()]=[255,0,0];return Image.fromarray(rgb),mask
out=Path('output/next106')/key;out.mkdir(parents=True,exist_ok=True)
modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True,min_pixels=56*56,max_pixels=448*448);processor.tokenizer.padding_side='left';model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().cuda();letters=['A','B','C'];label_ids=[processor.tokenizer.encode(x,add_special_tokens=False)[0] for x in letters]
instruction='The red outline marks a proposed image region. Judge only the outlined region, using surrounding pixels only as context. Requested target: {query}\nChoose A if the region depicts the exact requested target with the appropriate whole-object or part extent. Choose B for a different or merely associated object, or a wrong whole/part extent. Choose C if resolution, visibility, or your knowledge is insufficient. Do not assume a candidate must match. Answer exactly one letter: A, B, or C.'
template=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=instruction.format(query=json.dumps(query)))])],tokenize=False,add_generation_prompt=True);rows=[]
with torch.inference_mode():
 for offset in range(0,len(chosen),4):
  batch=chosen[offset:offset+4];images=[region(n)[0] for n in batch];inputs=processor(text=[template]*len(batch),images=images,padding=True,return_tensors='pt').to('cuda');logits=model(**inputs).logits[:,-1].float()[:,label_ids].cpu()
  for n,l,im in zip(batch,logits,images):
   row=dict(nodes[n],node=n,rank=ranks[n],old_rank=old_ranks.get(n),native_score=float(scores[n]),answer=letters[int(l.argmax())],margin=float(l[0]-l[1:].max()),logits=l.tolist());rows.append(row);im.save(out/f'{qi}_node_{n}.jpg')
  print(query,offset+len(batch),'/',len(chosen),flush=True)
for page in range((len(rows)+23)//24):
 canvas=Image.new('RGB',(6*210,4*240),'white');draw=ImageDraw.Draw(canvas)
 for k,row in enumerate(sorted(rows,key=lambda r:r['rank'])[page*24:(page+1)*24]):
  im=Image.open(out/f"{qi}_node_{row['node']}.jpg");im.thumbnail((205,205));xy=(k%6*210,k//6*240);canvas.paste(im,(xy[0],xy[1]+32));draw.text(xy,f"#{row['rank']} f{row['frame']} small={row['small']}\n{row['answer']} m={row['margin']:.2f}",fill='black')
 canvas.save(out/f'{qi}_expanded_{page}.jpg')
report=dict(query=query,query_index=qi,old_count=old_count,expanded_count=len(nodes),new_small_in_top64=sum(nodes[n]['small'] for n in new_rank[:64]),verified=len(rows),rows=rows,protocol='source-only union of old and expanded top64; no pair admission; same native encoder and exact-query A/B/C prompt; no GT or scope fitting',small_eligibility='original proposals <0.2% of source frame and >=64 pixels; query independent; bounded experiment')
(out/f'{qi}_unpaired.json').write_text(json.dumps(report,indent=2)+'\n');print(query,report['new_small_in_top64'],{c:sum(r['answer']==c for r in rows) for c in letters},flush=True)
