"""Compare teacher raw-latent aggregation with query unit-latent aggregation."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(4);device='cuda:2';root=a.root;report=json.loads((root/a.run/'report.json').read_text());assert sha256_file(root/a.run/'field.pt')==report['field_sha256'];payload=torch.load(root/a.run/'field.pt',weights_only=False)
 info=json.loads((root/'teachers.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];auth=json.loads((root/'responsibility.json').read_text());assert payload['geometry_binding']==auth['metadata']['frozen_geometry_binding'];z=payload['state_dict']['latent'].to(device);observed=payload['state_dict']['observed'].to(device);h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];rows=[]
 def summary(x):return dict(zip(['min','p01','median','p99','max'],torch.quantile(x,torch.tensor([0.,.01,.5,.99,1.],device=device)).tolist()))
 with torch.no_grad():
  for frame,r in zip(info['frames'],auth['views']):
   if frame not in info['train_frames']:continue
   path=root/r['relative_path'];assert sha256_file(path)==r['sha256'];hit=torch.load(path,map_location=device,weights_only=False);pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weight);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(z))).coalesce();support=torch.sparse.mm(mat,observed.float()[:,None])[:,0];valid=support>.5
   raw=torch.sparse.mm(mat,z*observed[:,None]);unit=torch.sparse.mm(mat,F.normalize(z,dim=-1)*observed[:,None]);cos=F.cosine_similarity(raw[valid],unit[valid],dim=-1)
   rows.append({'frame':frame,'direction_cosine':summary(cos),'fraction_below_095':float((cos<.95).float().mean())})
 result={'field_sha256':report['field_sha256'],'train_only':True,'element_norms':summary(z[observed].norm(dim=1)),'views':rows,'interpretation':'direction differences from norm-dependent semantic mixture; geometry weights themselves unchanged'}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
