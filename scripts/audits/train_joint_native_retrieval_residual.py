"""Bounded joint text/image source retrieval control in the original native space.

Caption phrase masks are source SAM pseudo-targets, not benchmark or instance GT.
The native fallback is supervised explicitly, separate from concept takeover.
"""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.train_rendered_extent import balanced_mask_loss


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--steps',type=int,default=240);p.add_argument('--parent',default='exact_render_query_threshold96');a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42);root=a.root
 parent=root/a.parent;pr=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==pr['field_sha256']
 payload=torch.load(parent/'field.pt',weights_only=False);ep=root/'caption_region_evidence.pt';er=json.loads(ep.with_suffix('.json').read_text());assert sha256_file(ep)==er['sha256'];e=torch.load(ep,weights_only=False)
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
 assert e['source_only'] and not e['benchmark_data_used'] and e['geometry_binding']==payload['geometry_binding'] and e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
 assert sha256_file(root/'responsibility.json')==info['authority_sha256']
 arch=dict(payload['architecture'],native_semantic_residual={'relative_bound':.25,'modalities':['text','image']})
 field=ElementRetrievalField(**arch).to(a.device);missing=field.load_state_dict(payload['state_dict'],strict=False)
 assert set(missing.missing_keys)=={'native_residual_decoder.weight','native_residual_decoder.bias'} and not missing.unexpected_keys
 field.requires_grad_(False);field.native_residual_decoder.requires_grad_(True)
 h,w=e['masks'].shape[-2:];views=[]
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);ids=torch.where(e['proposal_views']==vi)[0]
  if not len(ids):continue
  phrase_ids=sorted({i for r in ids.tolist() for i in e['records'][r]['phrase_indices']})
  if not phrase_ids:raise ValueError('source view lacks phrase supervision')
  targets=torch.stack([e['masks'][[r for r in ids.tolist() if i in e['records'][r]['phrase_indices']]].any(0) for i in phrase_ids])
  rec=auth['views'][vi];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False)
  gid,pid,weight=hit['gaussian_ids'].long(),hit['pixel_ids'].long(),hit['base_weights'].float()
  full=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);keep=field.observed[gid];gid,pid,weight=gid[keep],pid[keep],weight[keep]
  mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8)
  unique,inverse=gid.unique(return_inverse=True);mapping=torch.sparse_coo_tensor(torch.stack([pid,inverse]),weight/mass[pid].clamp_min(1e-8),(h*w,len(unique))).coalesce()
  tasks={'image':(e['native_descriptors'][ids],e['masks'][ids]),'text':(e['text_embeddings'][phrase_ids],targets)}
  views.append(dict(frame=frame,rows=unique,mapping=mapping,valid=valid,tasks={m:(q.float().to(a.device),t.flatten(1).T.float().to(a.device)) for m,(q,t) in tasks.items()}))
 val=views[-1];train=views[:-1];assert len(train)>0
 def prediction(v,m,cols):
  q=v['tasks'][m][0][cols];cos=field.native_cosine(q,v['rows'],modality=m)
  if m in field.native_query_thresholds:
   logits=field.readout_log_scale[m].clamp(-4.,6.).exp()*(cos-field.native_query_thresholds[m](q)[None])
  else:logits=field.cosine_logits(cos,m)
  return torch.sparse.mm(v['mapping'],logits.sigmoid())
 @torch.no_grad()
 def evaluate():
  result={}
  for m in ('text','image'):
   scores=[]
   for cols in torch.arange(len(val['tasks'][m][0]),device=a.device).split(8):
    pred=prediction(val,m,cols)>.5;target=val['tasks'][m][1][:,cols]>.5;valid=val['valid'][:,None]
    scores.extend(((pred&target&valid).sum(0)/((pred|target)&valid).sum(0).clamp_min(1)).tolist())
   result[m]=sum(scores)/len(scores)
  return result
 def snapshot():return {k:v.detach().cpu().clone() for k,v in field.native_residual_decoder.state_dict().items()}
 baseline=evaluate();best=baseline;chosen=0;state=snapshot();history=[dict(step=0,validation=baseline)]
 opt=torch.optim.Adam(field.native_residual_decoder.parameters(),lr=1e-4)
 for step in range(a.steps):
  v=train[(step//2)%len(train)];m=('text','image')[step%2];q,target=v['tasks'][m];cols=torch.randperm(len(q),device=a.device)[:4]
  loss=balanced_mask_loss(prediction(v,m,cols),target[:,cols],v['valid']);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(field.native_residual_decoder.parameters(),1.);opt.step()
  if (step+1)%30==0 or step+1==a.steps:
   metric=evaluate();admissible=all(metric[m]>=baseline[m] for m in baseline)
   history.append(dict(step=step+1,validation=metric,admissible=admissible,train_loss=float(loss.detach())))
   if admissible and sum(metric.values())>sum(best.values()):best=metric;chosen=step+1;state=snapshot()
   print(root.name,history[-1],flush=True)
 field.native_residual_decoder.load_state_dict(state);new={k:v.detach().cpu() for k,v in field.state_dict().items()}
 for k,v in payload['state_dict'].items():assert torch.equal(v,new[k]),k
 out=root/'joint_native_retrieval_residual';out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch,state_dict=new),out/'field.pt')
 report=dict(schema='joint_native_retrieval_residual.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],evidence_sha256=er['sha256'],authority_sha256=info['authority_sha256'],
 source_only=True,benchmark_data_used=False,geometry_unchanged=True,all_parent_tensors_exact=True,relative_bound=.25,steps=a.steps,selected_step=chosen,history=history,
 source_views={str(v['frame']):{m:len(q) for m,(q,t) in v['tasks'].items()} for v in views},validation_frame=val['frame'],
 selection='Neither modality source IoU below zero residual; then maximize equal-modality mean; source view held out only from residual fitting',
 supervision='Source caption phrase mask unions and image crops; actual native fallback sigmoid before exact raster; not concept-head loss or physical identity GT',
 implementation_sources={f:sha256_file(f) for f in [__file__,'radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_native_residual.py','scripts/audits/train_rendered_extent.py']})
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
