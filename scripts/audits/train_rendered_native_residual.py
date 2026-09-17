"""Fit bounded native residual with actual sigmoid-before-raster image retrieval.

Only source masks/crops are used. The final source view selects a checkpoint;
that view is not held out from the original frozen field. No geometry is updated.
"""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.train_rendered_extent import balanced_mask_loss


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--root',type=Path);p.add_argument('--device',required=True);p.add_argument('--steps',type=int,default=120);a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42)
 root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
 parent=root/'native_residual_source_selected';rp=parent/'report.json';r=json.loads(rp.read_text());assert sha256_file(parent/'field.pt')==r['field_sha256']
 payload=torch.load(parent/'field.pt',weights_only=False);field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
 field.native_residual_decoder.requires_grad_(True)
 ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==r['evidence_sha256'];e=torch.load(ep,weights_only=False)
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
 assert e['geometry_binding']==payload['geometry_binding'] and e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
 assert sha256_file(root/'responsibility.json')==info['authority_sha256']
 h,w=e['masks'].shape[-2:];views=[];queries=e['native_descriptors'].float().to(a.device)
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);ids=torch.where(e['proposal_views']==vi)[0]
  if not len(ids):continue
  rec=auth['views'][vi];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False)
  gid,pid,weight=hit['gaussian_ids'].long(),hit['pixel_ids'].long(),hit['base_weights'].float()
  full=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);keep=field.observed[gid];gid,pid,weight=gid[keep],pid[keep],weight[keep]
  mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8)
  unique,inverse=gid.unique(return_inverse=True)
  mapping=torch.sparse_coo_tensor(torch.stack([pid,inverse]),weight/mass[pid].clamp_min(1e-8),(h*w,len(unique))).coalesce()
  views.append(dict(frame=frame,ids=ids.to(a.device),rows=unique,mapping=mapping,valid=valid,target=e['masks'][ids].flatten(1).T.float().to(a.device)))
 val=next(v for v in views if v['frame']==r['validation_frame']);train=[v for v in views if v is not val]
 # Field/readout parameters stay fixed: isolate changing supervision, not capacity.
 opt=torch.optim.Adam(field.native_residual_decoder.parameters(),lr=1e-4)
 def prediction(v,cols):
  return torch.sparse.mm(v['mapping'],field.logits(queries[v['ids'][cols]],'image',v['rows']).sigmoid())
 @torch.no_grad()
 def evaluate(v):
  values=[];losses=[]
  for cols in torch.arange(len(v['ids']),device=a.device).split(8):
   pred=prediction(v,cols);target=v['target'][:,cols];valid=v['valid'][:,None]
   values.extend((((pred>.5)&(target>.5)&valid).sum(0)/(((pred>.5)|(target>.5))&valid).sum(0).clamp_min(1)).tolist())
   losses.append(float(balanced_mask_loss(pred,target,v['valid'])))
  return dict(iou=sum(values)/len(values),loss=sum(losses)/len(losses))
 def snapshot():return {k:v.detach().cpu().clone() for k,v in field.native_residual_decoder.state_dict().items()}
 best=evaluate(val);chosen=0;best_state=snapshot();history=[dict(step=0,validation=best)]
 executed_steps=a.steps if field.native_semantic_residual['relative_bound']>0 else 0
 for step in range(executed_steps):
  v=train[step%len(train)];cols=torch.randperm(len(v['ids']),device=a.device)[:4]
  loss=balanced_mask_loss(prediction(v,cols),v['target'][:,cols],v['valid'])
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(field.native_residual_decoder.parameters(),1.);opt.step()
  if (step+1)%30==0 or step+1==a.steps:
   metric=evaluate(val);history.append(dict(step=step+1,validation=metric,train_loss=float(loss.detach())))
   if metric['iou']>best['iou']:best=metric;chosen=step+1;best_state=snapshot()
   print(a.scene,history[-1],flush=True)
 field.native_residual_decoder.load_state_dict(best_state);state={k:v.detach().cpu() for k,v in field.state_dict().items()}
 for k,v in payload['state_dict'].items():
  if not k.startswith('native_residual_decoder.'):assert torch.equal(v,state[k]),k
 out=root/'native_residual_rendered_selected';out.mkdir(exist_ok=False);torch.save(dict(payload,state_dict=state),out/'field.pt')
 report=dict(schema='rendered_native_residual.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],evidence_sha256=sha256_file(ep),
  authority_sha256=info['authority_sha256'],source_only=True,benchmark_data_used=False,selection='maximum source-view image IoU at steps 0,30,60,90,120; base field saw this view',
  selected_step=chosen,validation_frame=val['frame'],steps=a.steps,executed_steps=executed_steps,zero_bound_noop=executed_steps==0,history=history,relative_bound=field.native_semantic_residual['relative_bound'],
  native_anchor_exact=True,geometry_unchanged=True,prompt_unchanged=True,all_nonresidual_parameters_exact=True,
  supervision='same-view source crop query and SAM pseudo-mask; full observed positive/negative pixel scope, not physical cross-view truth',
  implementation_sources={f:sha256_file(f) for f in [__file__,'radio_gs/models/bounded_native_residual.py','radio_gs/models/element_retrieval_field.py']})
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
