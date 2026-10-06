"""Reference-conditioned identity checks on every eligible source track component."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import label
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def crop(path,mask):
 image=Image.open(path).convert('RGB');m=np.array(Image.fromarray(mask.astype(bool)).resize(image.size,Image.Resampling.NEAREST));yy,xx=np.where(m);padx=max(8,int((xx.max()-xx.min())*.15));pady=max(8,int((yy.max()-yy.min())*.15));box=(max(0,xx.min()-padx),max(0,yy.min()-pady),min(image.width,xx.max()+padx+1),min(image.height,yy.max()+pady+1));im=image.crop(box);x0,y0,x1,y1=box;im.thumbnail((448,448));cm=torch.from_numpy(np.array(Image.fromarray(m[y0:y1,x0:x1]).resize(im.size,Image.Resampling.NEAREST),copy=True)).float()[None,None];edge=(F.max_pool2d(cm,3,1,1)+F.max_pool2d(-cm,3,1,1))[0,0]>.5;rgb=np.array(im);rgb[edge.numpy()]=[255,0,0];result=Image.fromarray(rgb);canvas=Image.new('RGB',(max(28,result.width),max(28,result.height)),(128,128,128));canvas.paste(result,((canvas.width-result.width)//2,(canvas.height-result.height)//2));return canvas
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--identity-mode',choices=['direct','description'],default='direct');a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;out=Path('output/batched_description_identity_tracks_next98' if a.identity_mode=='description' else 'output/identity_tracks_next98')/key;out.mkdir(parents=True,exist_ok=True);paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((Path('output/source_coverage_next72')/a.scene/'all.json').read_text())['images']};rows=json.loads((Path('output/grounded_seeds_next88')/key/'proposals.json').read_text())
 modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True,min_pixels=56*56,max_pixels=448*448);model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().cuda();letters=['A','B','C'];ids=[processor.tokenizer.encode(x,add_special_tokens=False)[0] for x in letters]
 instruction='The first image(s) show reference object region(s) marked by red outlines. The LAST image is a candidate region from another source view, also outlined in red. Does the LAST outlined region depict the same physical object as one of the references? A partial visible surface of that object is allowed. A different nearby object, background, or a region mixing the reference object with unrelated objects is NOT a match. Judge the outlined content, not shared surroundings. A: matching object. B: different object/background or mixed region. C: insufficient visual evidence. Answer exactly one letter: A, B, or C.'
 for row in rows:
  qi=row['query_index'];destination=out/f'{qi}.pt'
  if destination.exists():continue
  t=load(Path('output/boundary_tracks_next95/high_merged/union')/key/f'{qi}.pt');seed=t['seed_evidence'];seeds=seed.get('hypotheses',[seed]);refs=[crop(paths[x['seed_frame']],x['masks'].any(0).numpy()) for x in seeds];judgments=[]
  def describe(im):
   prompt='Describe only the object or material inside the red outline: its category, material, shape, and visible distinctive features. Ignore unrelated surroundings. Do not infer an intended target. Be concise.'
   template=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True);inputs=processor(text=[template],images=[im],return_tensors='pt').to('cuda');generated=model.generate(**inputs,max_new_tokens=64,do_sample=False);return processor.tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
  descriptions=[describe(im) for im in refs] if a.identity_mode=='description' else []
  description_cache={}
  if a.identity_mode=='description':
   processor.tokenizer.padding_side='left';jobs=[]
   for f,r in sorted(t['frames'].items()):
    if not r['present'] or f==t['seed_frame']:continue
    bm=r['probability'].float()>.5;ls,nc=label(bm.numpy(),structure=np.ones((3,3)));cr=(-F.max_pool2d(-bm.float()[None,None],3,1,1)[0,0]>.5).numpy()
    for j in range(1,nc+1):
     if ((ls==j)&cr).any():jobs.append((f,j,crop(paths[f],ls==j)))
   prompt='Describe only the object or material inside the red outline: its category, material, shape, and visible distinctive features. Ignore unrelated surroundings. Do not infer an intended target. Be concise.'
   template=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True)
   for start in range(0,len(jobs),8):
    batch=jobs[start:start+8];inputs=processor(text=[template]*len(batch),images=[j[2] for j in batch],padding=True,return_tensors='pt').to('cuda');generated=model.generate(**inputs,max_new_tokens=64,do_sample=False);captions=processor.tokenizer.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    for job,caption in zip(batch,captions):description_cache[job[:2]]=caption
   print(key,qi,'described',len(jobs),'components',flush=True)
  for frame,record in sorted(t['frames'].items()):
   if not record['present'] or frame==t['seed_frame']:continue
   binary=record['probability'].float()>.5;components,n=label(binary.numpy(),structure=np.ones((3,3)));core=(-F.max_pool2d(-binary.float()[None,None],3,1,1)[0,0]>.5).numpy();keep=np.zeros(binary.shape,dtype=bool)
   for ci in range(1,n+1):
    mask=components==ci
    if not (mask&core).any():continue
    im=crop(paths[frame],mask);content=[dict(type='image') for _ in range(len(refs)+1)]+[dict(type='text',text=instruction)];template=processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True);inputs=processor(text=[template],images=refs+[im],return_tensors='pt').to('cuda');description=None
    if a.identity_mode=='description':
     description=description_cache[(frame,ci)]
     prompt='Reference object descriptions: '+json.dumps(descriptions)+'. Candidate region description: '+json.dumps(description)+'. Does the candidate describe the same kind of object and compatible material/appearance as at least one reference? Allow partial visible surfaces and viewpoint differences. A different object or material, or a mixture including unrelated objects is not a match. A: compatible identity. B: different identity or mixed unrelated content. C: insufficient information. Answer exactly one letter: A, B, or C.'
     template=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True);inputs=processor(text=[template],return_tensors='pt').to('cuda')
    logits=model(**inputs).logits[0,-1].float()[ids];answer=letters[int(logits.argmax())];accepted=answer=='A';keep|=mask&accepted
    judgments.append(dict(frame=frame,component=ci,pixels=int(mask.sum()),description=description,answer=answer,logits=logits.cpu().tolist()))
   record.update(positive_allowed=torch.from_numpy(keep),negative_allowed=torch.ones(binary.shape,dtype=torch.bool),scope_reliability=float(keep.any()))
  t.update(identity_gate='source reference-component A/B/C; rejected positives unknown, not negative',source_track=str(Path('output/boundary_tracks_next95/high_merged/union')/key/f'{qi}.pt'));torch.save(t,destination)
  (out/f'{qi}.json').write_text(json.dumps(dict(query=t['query'],query_index=qi,identity_mode=a.identity_mode,description_batch_size=8 if a.identity_mode=='description' else None,reference_descriptions=descriptions,instruction=instruction,model=modelid,revision=model.config._commit_hash,target_rgb_or_labels_used=False,query_text_in_prompt=False,reference_frames=[s['seed_frame'] for s in seeds],judgments=judgments),indent=2)+'\n');print(key,qi,t['query'],len(judgments),{x:sum(j['answer']==x for j in judgments) for x in letters},flush=True)
if __name__=='__main__':main()
