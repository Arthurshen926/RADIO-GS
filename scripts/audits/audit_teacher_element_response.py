"""Separate region reconstruction from actual element readout on fixed assets."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 root=a.root;info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());receipt=json.loads((root/'episodes.json').read_text());report=json.loads((root/a.run/'report.json').read_text())
 def checked(path,digest):
  if sha256_file(path)!=digest:raise ValueError('asset identity differs: '+str(path))
 checked(root/'episodes.pt',receipt['sha256']);checked(root/'responsibility.json',info['authority_sha256'])
 data=torch.load(root/'episodes.pt',weights_only=False,map_location='cpu');checked(root/'teachers.json',data['teacher_manifest_sha256']);tr=info['teacher_files']['siglip'];checked(root/tr['path'],tr['sha256']);teacher=torch.load(root/tr['path'],weights_only=False,map_location='cpu')['descriptors'].float().to(a.device)
 fixture=torch.load(root/'cold_query_fixture.pt',weights_only=False,map_location='cpu');geo=FrozenGaussianGeometry.load(fixture['geometry_path'],expected_sha256=fixture['geometry_sha256'],device=a.device);field,_=load_element_retrieval_field(root/a.run/'field.pt',expected_sha256=report['field_sha256'],geometry=geo,expected_query_encoder=json.loads((root/'native_query_encoder.json').read_text()),device=a.device)
 if field.native_decoder is None:raise ValueError('coupled native decoder required')
 q=F.normalize(data['queries']['text'].float().to(a.device),dim=-1);masks=data['masks'].flatten(1).to(a.device).bool();rows=[]
 with torch.no_grad():
  element_cos=[]
  for z in field.latent.split(4096):element_cos.append(F.normalize(field.native_decoder(z),dim=-1)@q.T)
  element_cos=torch.cat(element_cos);element_prob=field.cosine_logits(element_cos,'text').sigmoid()
  for vi,record in enumerate(auth['views']):
   path=root/record['relative_path'];checked(path,record['sha256']);hit=torch.load(path,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float();n=len(field.latent);pixels=masks.shape[1]
   mass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();support=torch.sparse.mm(mat,field.observed.float()[:,None])[:,0]
   zmap=torch.sparse.mm(mat,field.latent*field.observed[:,None])/support[:,None].clamp_min(1e-8);cosmap=torch.sparse.mm(mat,element_cos*field.observed[:,None])/support[:,None].clamp_min(1e-8);probmap=torch.sparse.mm(mat,element_prob*field.observed[:,None])/support[:,None].clamp_min(1e-8)
   for ri in torch.where(data['proposal_views']==vi)[0].tolist():
    region=masks[ri]&(support>.5)
    if not region.any():rows.append({'region':ri,'view':vi,'available':False});continue
    decoded=field.native_decoder(zmap[region].mean(0));t=F.normalize(teacher[ri],dim=0);tc=t@q.T;pooled=F.normalize(decoded,dim=0)@q.T;actual=cosmap[region].mean(0);prob=probmap[region].mean(0);top=int(tc.argmax());order=tc.argsort(descending=True)[:5]
    row={'region':ri,'view':vi,'split':'train' if info['frames'][vi] in info['train_frames'] else 'validation','available':True,'known_pixels':int(region.sum()),'region_reconstruction_cosine':float(F.cosine_similarity(decoded,t,dim=0)), 'teacher_top_query':data['queries']['queries'][top], 'pooled_cosine_mae':float((pooled-tc).abs().mean()),'element_cosine_mae':float((actual-tc).abs().mean()),'aggregation_cosine_gap':float((pooled-actual).abs().mean()),'pooled_teacher_top1_agreement':int(pooled.argmax())==top,'element_teacher_top1_agreement':int(actual.argmax())==top,'posterior_teacher_top1_agreement':int(prob.argmax())==top,'teacher_top5':order.tolist(),'teacher_top5_scores':tc[order].tolist(),'pooled_top5_scores':pooled[order].tolist(),'element_top5_scores':actual[order].tolist(),'posterior_top5_scores':prob[order].tolist()};rows.append(row)
 keys=['region_reconstruction_cosine','pooled_cosine_mae','element_cosine_mae','aggregation_cosine_gap','pooled_teacher_top1_agreement','element_teacher_top1_agreement','posterior_teacher_top1_agreement'];macro={}
 for split in ['train','validation']:
  selected=[r for r in rows if r.get('split')==split and r['available']];macro[split]={'regions':len(selected),**{k:sum(r[k] for r in selected)/max(1,len(selected)) for k in keys}}
 out={'schema':'teacher_element_response_audit.v1','field_sha256':report['field_sha256'],'episode_sha256':receipt['sha256'],'teacher_sha256':tr['sha256'],'implementation_sha256':sha256_file(__file__),'teacher_agreement_is_not_ground_truth_accuracy':True,'training_or_geometry_updates':False,'macro':macro,'rows':rows};a.output.write_text(json.dumps(out,indent=2)+'\n');print(macro)
if __name__=='__main__':main()
