"""Read-only privileged crop diagnostic of the deployed decoder subspace.

Least-squares projection is a descriptor reconstruction diagnostic, NOT an
upper bound on classification and NOT actual scene element retrieval.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.native_query_basis import native_query_basis


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--encodings',type=Path,required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 oracle=torch.load(a.encodings,map_location='cpu',weights_only=False)
 if not oracle.get('not_training_or_deployment'):raise ValueError('expected explicitly privileged diagnostic encodings')
 image=oracle['images'].float();q=F.normalize(oracle['queries'].float(),dim=-1)
 stages={'native':F.normalize(image,dim=-1)@q.T};receipts={};reconstruction={}
 for name in a.runs:
  folder=a.root/name;report=json.loads((folder/'report.json').read_text());path=folder/'field.pt'
  if sha256_file(path)!=report['field_sha256']:raise ValueError('field digest differs')
  payload=torch.load(path,map_location='cpu',weights_only=False);state=payload['state_dict']
  if not payload['architecture'].get('native_siglip_readout'):raise ValueError('native decoder required')
  w=state['native_decoder.weight'].float();b=state['native_decoder.bias'].float();u,s,_=torch.linalg.svd(w,full_matrices=False);rank=int((s>s[0]*max(w.shape)*torch.finfo(w.dtype).eps).sum());u=u[:,:rank]
  decoded=(image-b)@u@u.T+b
  stages[name+'_fixed_norm_l2']=F.normalize(decoded,dim=-1)@q.T
  # Cosine is scale invariant: a fixed-unit L2 target can misleadingly penalize
  # an affine bias. Project directions onto closure(span(W) + positive*b_perp).
  perpendicular=b-u@(u.T@b);bn=perpendicular.norm()
  direction=image@u@u.T
  if bn>1e-6:
   axis=perpendicular/bn;direction=direction+(image@axis).clamp_min(0)[:,None]*axis[None,:]
  stages[name+'_direction']=F.normalize(direction,dim=-1)@q.T
  reconstruction[name]={'rank':rank,'direction_cosine_to_native_crop':float(F.cosine_similarity(direction,image).mean()),'bias_orthogonal_norm':float(bn),'mean_cosine_to_native_crop':float(F.cosine_similarity(decoded,image).mean()),'relative_l2':float((decoded-image).norm()/image.norm()),'least_squares_projection_not_classification_upper_bound':True}
  receipts[name]=report['field_sha256']
 info=json.loads((a.root/'teachers.json').read_text());receipt=json.loads((a.root/'episodes.json').read_text());path=a.root/'episodes.pt'
 if sha256_file(path)!=receipt['sha256']:raise ValueError('source episode digest differs')
 data=torch.load(path,map_location='cpu',weights_only=False);trainviews=[info['frames'].index(f) for f in info['train_frames']];regions=torch.tensor([i for i,v in enumerate(data['proposal_views']) if int(v) in trainviews]);bank=torch.cat([data['queries']['text'].float(),data['queries']['image'][regions].float()]);basis,_=native_query_basis(bank,64)
 # Keep q native, exactly as in deployed decoder scoring, for this control.
 stages['source_svd64_native_query']=F.normalize(image@basis.T@basis,dim=-1)@q.T
 results={}
 for name,scores in stages.items():
  rows=[]
  for meta,score in zip(oracle['metadata'],scores):
   rank=int(torch.where(score.argsort(descending=True)==meta['category_index'])[0][0])+1
   rows.append({'frame':meta['frame'],'category':meta['category'],'crop':meta['crop'],'rank':rank})
  macro={}
  for kind in ['masked','unmasked']:
   chosen=[r for r in rows if r['crop']==kind];macro[kind]={'tasks':len(chosen),'top1':sum(r['rank']==1 for r in chosen)/len(chosen),'mrr':sum(1/r['rank'] for r in chosen)/len(chosen)}
  results[name]={'macro':macro,'observations':rows}
 out={'schema':'decoder_subspace_retention.v1','oracle_encodings_sha256':sha256_file(a.encodings),'source_episode_sha256':receipt['sha256'],'field_sha256':receipts,'implementation_sha256':sha256_file(__file__),'diagnostic_only':True,'uses_target_gt_crops':True,'field_updated':False,'geometry_or_localization_evaluated':False,'scope':'affine least-squares reconstruction of oracle image descriptors through fixed learned decoder; not actual map latents and not a classification ceiling','native_image_norm_mean':float(image.norm(dim=-1).mean()),'reconstruction':reconstruction,'results':results}
 a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v['macro'] for k,v in results.items()});print(reconstruction)
if __name__=='__main__':main()
