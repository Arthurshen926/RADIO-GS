"""Registered-camera source control: same selected visibility objective on ScanNet.

The selected-only arm uses a declared continuous opacity surrogate; validation
uses the exact deployed hard element selection and PNG alpha threshold. No
benchmark input, teacher encoder, geometry, or native field tensor is fitted.
"""
import argparse,json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.train_rendered_extent import balanced_mask_loss
from radio_gs.training.selected_output_supervision import selected_mask_supervision


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--mode',choices=['selected','visible_selected'],required=True);p.add_argument('--steps',type=int,default=400);p.add_argument('--epochs',type=int,default=0);p.add_argument('--parent-name',default='counterfactual_extent_benefit_admission');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
 out=a.output;out.mkdir(parents=True,exist_ok=False);parent=a.root/a.parent_name/'field.pt';report=json.loads(parent.with_name('report.json').read_text());assert sha256_file(parent)==report['field_sha256'];payload=torch.load(parent,weights_only=False);model=ElementRetrievalField(**payload['architecture']).to(a.device);model.load_state_dict(payload['state_dict']);model.requires_grad_(False)
 head=model.grounded_transfer;head.decoder.requires_grad_(True)
 if head.residual is not None:head.residual.requires_grad_(True)
 ep=a.root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['source_only'] and not e['benchmark_data_used'] and e['geometry_binding']==payload['geometry_binding'];del e['region_probability']
 info=json.loads((a.root/'teachers.json').read_text());auth=json.loads((a.root/'responsibility.json').read_text());assert sha256_file(a.root/'responsibility.json')==info['authority_sha256'];assert e['teacher_manifest_sha256']==sha256_file(a.root/'teachers.json');assert sha256_file(a.geometry_report)==auth['metadata']['frozen_geometry_report_sha256']
 gr=json.loads(a.geometry_report.read_text());g=gr['results']['independent_scale'];geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);geometry.assert_binding(payload['geometry_binding']);geometry.assert_binding(auth['metadata']['frozen_geometry_binding'])
 h,w=e['masks'].shape[-2:];assert (h,w)==(auth['metadata']['feature_height'],auth['metadata']['feature_width']);pixels=h*w;views=[]
 for frame in info['train_frames']:
  targets={}
  for i,r in enumerate(e['records']):
   if r['frame']!=frame or r['confidence']<.7 or int(e['masks'][i].sum())<20:continue
   for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros(h,w,dtype=torch.bool))|e['masks'][i].bool()
  if not targets:continue
  rec=auth['views'][info['frames'].index(frame)];path=a.root/rec['relative_path'];assert sha256_file(path)==rec['sha256'];hit=torch.load(path,map_location=a.device,weights_only=False);pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();full=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,weight);keep=model.observed[gid];pid,gid,weight=pid[keep],gid[keep],weight[keep];mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8);mapping=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,len(model.latent))).coalesce()
  camera=auth['metadata']['cameras'][str(frame)];k=torch.tensor(camera['intrinsic'],device=a.device);world_to_camera=torch.tensor(camera['world_to_camera'],device=a.device);assert k.shape==(3,3) and world_to_camera.shape==(4,4) and torch.isfinite(k).all() and torch.isfinite(world_to_camera).all()
  # Deduplicate exact same source target; retain all aliases for deterministic cycling.
  scopes={}
  for qi,target in targets.items():
   key=target.numpy().tobytes();entry=scopes.setdefault(key,dict(target=target.flatten().float().to(a.device),queries=[]));entry['queries'].append(qi)
  views.append(dict(frame=frame,mapping=mapping,valid=valid,mass=mass,full_mass=full,camera=world_to_camera,k=k,tasks=list(scopes.values())))
 assert len(views)>=8;train,val=views[:-4],views[-4:];queries=e['text_embeddings'].float().to(a.device)
 def probability(qi):return model.logits(queries[qi:qi+1],'text').sigmoid()[:,0]*model.observed
 def alpha(v,opacity):
  _,value,_=rasterization(means=geometry.get_xyz(),quats=geometry.get_rotation(),scales=geometry.get_scaling(),opacities=opacity,colors=torch.ones(len(model.latent),1,device=a.device),viewmats=v['camera'][None],Ks=v['k'][None],width=w,height=h,near_plane=auth['metadata']['near_plane'],far_plane=auth['metadata']['far_plane'],packed=False)
  return value[0,...,0].flatten()
 # Validate generic metadata cameras against the original sparse compositor.
 with torch.no_grad():
  camera_replay_max_error=max(float((alpha(v,geometry.get_opacity().reshape(-1))-v['full_mass']).abs().max()) for v in views)
 assert camera_replay_max_error<5e-4, ('Registered camera/compositor mismatch',camera_replay_max_error)
 print('camera_replay_max_error',camera_replay_max_error,flush=True)
 # A bounded fixed validation cohort, selected by source order before any outcomes.
 validation=[(v,t) for v in val for t in v['tasks'][::max(1,len(v['tasks'])//20)]]
 @torch.no_grad()
 def evaluate():
  values=[]
  for v,t in validation:
   q=probability(t['queries'][0]);full=(torch.sparse.mm(v['mapping'],q[:,None])[:,0]>.5)&v['valid'];sel=alpha(v,geometry.get_opacity().reshape(-1)*(q>.5));selected=(sel*255+.5).floor().clamp(0,255)>10;target=t['target'].bool()
   def iou(pred):return float((pred&target).sum()/(pred|target).sum().clamp_min(1))
   values.append((iou(full),iou(selected)))
  return dict(full_iou=sum(v[0] for v in values)/len(values),selected_iou=sum(v[1] for v in values)/len(values),observations=len(values))
 initial=evaluate();history=[dict(step=0,validation=initial)];best=initial['selected_iou'];chosen=0
 def snapshot():return {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
 state=snapshot();optimizer=torch.optim.Adam([v for v in head.parameters() if v.requires_grad],lr=1e-4);tasks=[(v,t) for v in train for t in v['tasks']];schedule=[]
 steps=a.epochs*len(tasks) if a.epochs else a.steps
 while len(schedule)<steps:schedule.extend(torch.randperm(len(tasks)).tolist())
 skipped=0;print('initial',initial,flush=True)
 for step,index in enumerate(schedule[:steps]):
  v,t=tasks[index];qi=t['queries'][step%len(t['queries'])];q=probability(qi);full=torch.sparse.mm(v['mapping'],q[:,None]);full_loss=balanced_mask_loss(full,t['target'][:,None],v['valid'])
  if a.mode in ('selected','visible_selected'):
   # Compact linear gate eliminates tiny everywhere-positive opacity tails.
   gate=((q-.25)/.5).clamp(0,1);silhouette=alpha(v,geometry.get_opacity().reshape(-1)*gate);soft=silhouette/(silhouette+10.5/255)
   selected_valid=torch.ones(pixels,dtype=torch.bool,device=a.device)
   if a.mode=='visible_selected':
    full_selected_mass=torch.sparse.mm(v['mapping'],gate[:,None])[:,0]*v['mass']
    selected_valid,visible_fraction=selected_mask_supervision(t['target'],full_selected_mass,silhouette,v['valid'])
   loss=.5*(full_loss+balanced_mask_loss(soft[:,None],t['target'][:,None],selected_valid))
  else:loss=full_loss
  if loss.requires_grad:
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad],1.);optimizer.step()
  else:skipped+=1
  if (step+1)%100==0 or step+1==steps:
   metric=evaluate();history.append(dict(step=step+1,validation=metric,loss=float(loss.detach())));print(history[-1],flush=True)
   if metric['full_iou']>=initial['full_iou']-1e-6 and metric['selected_iou']>best+1e-6:state=snapshot();best=metric['selected_iou'];chosen=step+1
 head.load_state_dict(state);saved={k:v.detach().cpu() for k,v in model.state_dict().items()}
 allowed=('grounded_transfer.decoder.','grounded_transfer.residual.')
 for k,v in payload['state_dict'].items():
  if not k.startswith(allowed):assert torch.equal(v,saved[k]),k
 geometry.assert_binding(payload['geometry_binding']);torch.save(dict(payload,state_dict=saved),out/'field.pt')
 result=dict(schema='registered_visibility_qualified_selected_scope.v1',mode=a.mode,parent_name=a.parent_name,source_only=True,benchmark_data_used=False,assistant_review_used_for_training=False,parent_field_sha256=sha256_file(parent),field_sha256=sha256_file(out/'field.pt'),geometry_unchanged=True,native_field_unchanged=True,query_encoder_unchanged=True,geometry_binding=payload['geometry_binding'],teacher_manifest_sha256=sha256_file(a.root/'teachers.json'),evidence_sha256=sha256_file(ep),authority_sha256=sha256_file(a.root/'responsibility.json'),geometry_report_sha256=sha256_file(a.geometry_report),camera_replay_max_error=camera_replay_max_error,selected_step=chosen,steps=steps,epochs=a.epochs,unique_training_scopes=len(tasks),skipped_no_trainable_route=skipped,history=history,training_frames=[v['frame'] for v in train],validation_frames=[v['frame'] for v in val],validation_scope='Source views held out only from new head fitting; parent saw all. Fixed source-order subsample, first alias per exact scope; not family-independent validation.',selection='Exact hard selected-render IoU improvement with no full-scene source IoU decrease; step zero eligible.',surrogate='Selected arm averages original full-visible mask loss and selected-opacity alpha/(alpha+10.5/255); opacity gate clamp((probability-.25)/.5,0,1). Explicit fixed scope-pass/step budget recorded.',negative_supervision='All source pixels for selected control; visible_selected excludes isolated negative alpha without at least half full-scene visibility. Positive source masks retained; visibility detached.', visibility_implementation_sha256=sha256_file('radio_gs/training/selected_output_supervision.py'),implementation_sha256=sha256_file(__file__))
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('selected',chosen,flush=True)
if __name__=='__main__':main()
