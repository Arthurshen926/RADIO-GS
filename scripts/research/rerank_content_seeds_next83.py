"""Content-conditioned native-candidate reranking using a local caption model.

All authorized source regions compete. Bound verifier work to 16 geometrically
nonduplicate candidates/query, with two views/candidate. No query aliases, GT,
benchmark vocabulary, or generated captions are used to rank the candidates.
"""
import argparse,json
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from transformers import BlipForConditionalGeneration,BlipProcessor
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene
 relation=load(Path('output/region_relations_next79')/a.scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());rows=[];nodes=[]
 for e in manifest['entries']:
  r=load(e['path']);rows.append(F.normalize(r['descriptors'].float(),dim=1));nodes.extend((e['frame'],i,e['path'],r['source']) for i in range(len(r['descriptors'])))
 data=load(Path('output/native_rebuild_next54/mixture_masscal')/key/'field.pt');root=Path(data['root']);q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float(),dim=-1);scores=torch.cat(rows)@q.T;edges=torch.tensor(relation['equivalence_edges']);ij=edges[:,:2].long();source=Path('output/source_coverage_next72')/a.scene;paths={int(r['image_id'].split('_')[-1]):r['path'] for r in json.loads((source/'all.json').read_text())['images']}
 @lru_cache(maxsize=8)
 def bank(path):return load(path)
 def support(node):
  _,ri,path,_=nodes[node];r=bank(path);take=(r['columns']==ri)&(r['values']>.5);return set(r['ids'][r['rows'][take].long()].tolist())
 @lru_cache(maxsize=2)
 def mask_file(path):
  r=load(path);sam=load(r['sam_path']);return r,unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
 def region(node):
  f,ri,_,path=nodes[node];r,masks=mask_file(path);mask=masks[int(r['native_proposal_indices'][ri])].copy();rgb=Image.open(paths[f]).convert('RGB');ys,xs=np.where(mask);h,w=mask.shape;box=(int(xs.min()*rgb.width/w),int(ys.min()*rgb.height/h),int((xs.max()+1)*rgb.width/w),int((ys.max()+1)*rgb.height/h));return rgb.crop(box),mask
 modelid='Salesforce/blip-image-captioning-base';processor=BlipProcessor.from_pretrained(modelid,local_files_only=True);model=BlipForConditionalGeneration.from_pretrained(modelid,local_files_only=True).eval().half().cuda()
 out=Path('output/content_seeds_next83')/key;out.mkdir(parents=True,exist_ok=True);report=[];old=json.loads((Path('output/region_seeds_next80')/key/'proposals.json').read_text())
 for record in old:
  qi=record['query_index'];text=record['query'];pair=scores[ij,qi].amin(1);chosen=[];supports=[];used=set()
  for ei in torch.argsort(pair,descending=True).tolist():
   i,j=ij[ei].tolist()
   if i in used or j in used:continue
   sig=support(i)|support(j)
   if any(len(sig&prev)/max(len(sig|prev),1)>.8 for prev in supports):continue
   chosen.append((ei,i,j));supports.append(sig);used.update([i,j])
   if len(chosen)==16:break
  images=[]
  for _,i,j in chosen:images.extend([region(i)[0],region(j)[0]])
  likelihood=[]
  for start in range(0,len(images),8):
   batch=images[start:start+8];inputs=processor(images=batch,text=[text]*len(batch),return_tensors='pt',padding=True).to('cuda');inputs['pixel_values']=inputs['pixel_values'].half();inputs['input_ids'][:,0]=model.config.text_config.bos_token_id
   logits=model(**inputs).logits[:,:-1].float();labels=inputs['input_ids'][:,1:];valid=inputs['attention_mask'][:,1:].float();ll=logits.log_softmax(-1).gather(-1,labels[...,None])[...,0];likelihood.extend(((ll*valid).sum(1)/valid.sum(1)).cpu().tolist())
  values=torch.tensor(likelihood).reshape(-1,2);rank=values.amin(1);winner=int(rank.argmax());ei,i,j=chosen[winner];node=i if values[winner,0]>=values[winner,1] else j;crop,mask=region(node);f=nodes[node][0]
  # Keep the same native confidence semantics; BLIP changes identity choice only.
  result=dict(query=text,query_index=qi,seed_frame=f,seed_confidence=float(pair[ei].clamp(0,1)),masks=torch.from_numpy(mask)[None].bool(),source='BLIP query conditional likelihood over native geometric pair candidates',pair_nodes=[i,j],pair_frames=[nodes[i][0],nodes[j][0]],native_cosines=scores[[i,j],qi].tolist(),single_scope_hypothesis=True,verifier=modelid,verifier_log_likelihood=float(rank[winner]))
  torch.save(result,out/f'{qi}.pt');crop.save(out/f'{qi}_chosen.jpg')
  captions=processor.batch_decode(model.generate(**processor(images=[crop],return_tensors='pt').to('cuda',torch.float16),max_new_tokens=24),skip_special_tokens=True)
  candidate_rows=[dict(nodes=[i,j],frames=[nodes[i][0],nodes[j][0]],native_score=float(pair[ei]),query_log_likelihood=values[k].tolist()) for k,(ei,i,j) in enumerate(chosen)]
  report.append(dict(query=text,query_index=qi,seed_frame=f,selected_rank=winner,selected_caption_for_diagnosis_only=captions[0],candidate_count=len(chosen),candidates=candidate_rows));print(text,'rank',winner,'caption',captions[0],flush=True)
 (out/'proposals.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'contract.json').write_text(json.dumps(dict(model=modelid,source_frames=manifest['source_frames'],candidate_budget=16,views_per_candidate=2,duplicate_support_jaccard=.8,query_verbatim=True,target_rgb_used=False,generated_captions_used_for_selection=False,extra_inference_teacher=True),indent=2)+'\n')
if __name__=='__main__':main()
