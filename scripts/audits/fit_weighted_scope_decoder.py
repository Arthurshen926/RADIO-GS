"""Fixed-budget final concept-layer refit to pixel-union, evidence-weighted scopes."""
import argparse,json,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--aggregation',choices=['weighted','equal'],default='weighted');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root/('weighted_scope_decoder' if a.aggregation=='weighted' else 'equal_scope_decoder');out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();impl={}
for f in [Path(__file__),Path('radio_gs/models/grounded_semantic_transfer.py')]:shutil.copy2(f,snap/f.name);impl[str(f)]=sha256_file(f)
inputs={}
def load(name,digest):
 path=root/name;assert sha256_file(path)==digest;inputs[str(path)]=digest;return torch.load(path,weights_only=False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];e=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
pr=json.loads((root/'joint_supervised_image_residual/report.json').read_text());payload=load('joint_supervised_image_residual/field.pt',pr['field_sha256']);assert e['geometry_binding']==payload['geometry_binding'];model=ElementRetrievalField(**payload['architecture']).to(a.device);model.load_state_dict(payload['state_dict']);model.requires_grad_(False);head=model.grounded_transfer;groups=json.loads((root/'grounded_semantic_transfer96/report.json').read_text())['groups'];n=len(model.latent);G=len(groups);pixels=e['masks'][0].numel();offset=head.identity_selection['trusted_concept_start'];hidden_start=head.residual[0].out_features//2;assert hidden_start==128
num=torch.zeros(n,G,device=a.device);den=torch.zeros_like(num);train_num=torch.zeros_like(num);train_den=torch.zeros_like(num);views=[];heldout=set(info['train_frames'][::5])
with torch.no_grad():
 z=head.features(model.latent);hidden=head.residual[1](head.residual[0](z))[:,hidden_start:];x=torch.cat([z,hidden,torch.ones(n,1,device=a.device)],1);old=(head.decoder(z)+head.residual(z))[:,offset:].clamp(1e-6,1-1e-6);assert old.shape[1]==G
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);gid=hit['gaussian_ids'].long().to(a.device);pid=hit['pixel_ids'].long().to(a.device);w=hit['base_weights'].float().to(a.device);pm=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/pm[pid].clamp_min(1e-8),(pixels,n)).coalesce();mass=torch.sparse.sum(mat,dim=0).to_dense();seen=mass>1e-6
  ids=[j for j,g in enumerate(groups) if vi in g['views']];masks=torch.stack([e['masks'][[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]].bool().any(0).flatten() for j in ids],1).to(a.device);contribution=torch.sparse.mm(mat.T,masks.float())*seen[:,None];effective=mass*seen
  if a.aggregation=='equal':contribution=contribution/mass[:,None].clamp_min(1e-8);effective=seen.float()
  num[:,ids]+=contribution;den[:,ids]+=effective[:,None]
  if frame not in heldout:train_num[:,ids]+=contribution;train_den[:,ids]+=effective[:,None]
  else:views.append((frame,ids,masks,mat))
  print(a.scene,'prepare',frame,flush=True)
 def fit(numerator,denominator):
  coeff=[];enabled=[];torch.manual_seed(42)
  for j in range(G):
   target=numerator[:,j]/denominator[:,j].clamp_min(1e-8);known=(denominator[:,j]>1e-6)&model.observed;pos=torch.where(known&(target>.5))[0];neg=torch.where(known&(target<=.5))[0]
   if min(len(pos),len(neg))<8:
    coeff.append(torch.cat([head.decoder.weight[offset+j],head.residual[2].weight[offset+j,hidden_start:],(head.decoder.bias[offset+j]+head.residual[2].bias[offset+j]).reshape(1)]));enabled.append(False);continue
   pos=pos[torch.randperm(len(pos),device=a.device)[:4096]];neg=neg[torch.randperm(len(neg),device=a.device)[:4096]];indices=torch.cat([pos,neg]);weight=torch.cat([torch.full((len(pos),),.5/len(pos),device=a.device),torch.full((len(neg),),.5/len(neg),device=a.device)]);xx=x[indices];regularizer=torch.eye(x.shape[1],device=a.device)*.001;regularizer[-1,-1]=0;coef=torch.linalg.solve(xx.T@(xx*weight[:,None])+regularizer,xx.T@(target[indices]*weight));coeff.append(coef);enabled.append(True)
  return torch.stack(coeff,1),enabled
 validation_coef,enabled=fit(train_num,train_den);prediction=(x@validation_coef).clamp(1e-6,1-1e-6);rows=[]
 for frame,ids,masks,mat in views:
  coverage=torch.sparse.mm(mat,model.observed.float()[:,None]);valid=coverage>.5
  for name,values in [('parent_in_sample',old),('refit_heldout',prediction)]:
   pred=(torch.sparse.mm(mat,values[:,ids]*model.observed[:,None])/coverage.clamp_min(1e-8)>.5)&valid
   for ci,j in enumerate(ids):
    if not enabled[j]:continue
    truth=masks[:,ci];inter=(pred[:,ci]&truth).sum();rows.append(dict(frame=frame,group=j,phrase=groups[j]['phrase'],arm=name,iou=float(inter/(pred[:,ci]|truth).sum().clamp_min(1))))
 metrics={name:sum(r['iou'] for r in rows if r['arm']==name)/sum(r['arm']==name for r in rows) for name in ['parent_in_sample','refit_heldout']};passed=metrics['refit_heldout']>=metrics['parent_in_sample'];print(a.scene,'validation',metrics,passed,flush=True)
 coefficient,all_enabled=fit(num,den)
 head.decoder.weight[offset:].copy_(coefficient[:z.shape[1]].T);head.decoder.bias[offset:].copy_(coefficient[-1]);head.residual[2].weight[offset:,:hidden_start].zero_();head.residual[2].weight[offset:,hidden_start:].copy_(coefficient[z.shape[1]:-1].T);head.residual[2].bias[offset:].zero_()
 state={k:v.cpu() for k,v in model.state_dict().items()};changed={'grounded_transfer.decoder.weight','grounded_transfer.decoder.bias','grounded_transfer.residual.2.weight','grounded_transfer.residual.2.bias'}
 for k,v in payload['state_dict'].items():
  if k in changed:assert torch.equal(v[:offset],state[k][:offset])
  else:assert torch.equal(v,state[k]),k
 torch.save(dict(payload,state_dict=state),out/'field.pt');report=dict(schema='weighted_scope_decoder.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],source_only=True,benchmark_data_used=False,source_acceptance_passed=passed,validation=metrics,validation_rows=rows,heldout_frames=sorted(heldout),validation_scope='Views held out from final concept-layer fit only; frozen field and hidden features saw these source images/masks. Parent baseline also saw them.',group_lifting='pixel union before projection; '+('actual normalized pixel contribution weights' if a.aggregation=='weighted' else 'equal detected-view conditional membership'),aggregation=a.aggregation,ridge=.001,maximum_samples_per_sign=4096,seed=42,only_concept_output_layer_changed=True,all_field_native_region_and_routing_tensors_exact=True,inputs=inputs,implementations=impl,all_enabled=sum(all_enabled));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
