"""Evaluate source-only SVD compression using privileged oracle diagnostic crops."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.native_query_basis import native_query_basis
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--encodings',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
if a.output.exists():raise FileExistsError(a.output)
info=json.loads((a.root/'teachers.json').read_text());receipt=json.loads((a.root/'episodes.json').read_text());assert sha256_file(a.root/'episodes.pt')==receipt['sha256'];data=torch.load(a.root/'episodes.pt',weights_only=False);oracle=torch.load(a.encodings,weights_only=False);trainviews=[info['frames'].index(f) for f in info['train_frames']];regions=torch.tensor([i for i,v in enumerate(data['proposal_views']) if int(v) in trainviews]);bank=torch.cat([data['queries']['text'].float(),data['queries']['image'][regions].float()]);image=oracle['images'].float();q=oracle['queries'].float();results={}
for dim in (64,128):
 if dim>min(bank.shape):continue
 basis,energy=native_query_basis(bank,dim);scores=F.normalize(image@basis.T,dim=-1)@F.normalize(q@basis.T,dim=-1).T;rows=[]
 for row,score in zip(oracle['metadata'],scores):
  rank=int(torch.where(score.argsort(descending=True)==row['category_index'])[0][0])+1;rows.append({'frame':row['frame'],'category':row['category'],'crop':row['crop'],'rank':rank})
 macro={}
 for kind in ('masked','unmasked'):
  selected=[r for r in rows if r['crop']==kind];macro[kind]={'top1':sum(r['rank']==1 for r in selected)/len(selected),'mrr':sum(1/r['rank'] for r in selected)/len(selected)}
 results[str(dim)]={'source_energy':energy,'macro':macro,'observations':rows}
a.output.write_text(json.dumps({'schema':'source_basis_oracle.v1','oracle_encodings_sha256':sha256_file(a.encodings),'episode_sha256':receipt['sha256'],'source_only_basis':True,'gt_or_target_features_used_to_fit_basis':False,'diagnostic_only':True,'results':results},indent=2)+'\n');print({k:v['macro'] for k,v in results.items()})
