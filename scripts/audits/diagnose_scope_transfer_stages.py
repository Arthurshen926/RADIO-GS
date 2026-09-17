"""Matched source-query scope stages; labels are teacher masks, not instance GT."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/'scope_transfer_stages';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');inputs={}
def load(name,digest):
 path=root/name;assert sha256_file(path)==digest;inputs[str(path)]=digest;return torch.load(path,weights_only=False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];e=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
pr=json.loads((root/'joint_supervised_image_residual/report.json').read_text());payload=load('joint_supervised_image_residual/field.pt',pr['field_sha256']);assert e['geometry_binding']==payload['geometry_binding'];field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False);groups=json.loads((root/'grounded_semantic_transfer96/report.json').read_text())['groups'];n=len(field.latent);G=len(groups);pixels=e['masks'][0].numel();views=[];total_old=torch.zeros(n,G,device=a.device);total_union=torch.zeros_like(total_old);den=torch.zeros_like(total_old);union_error=[]
with torch.no_grad():
 head=field.grounded_transfer;z=head.features(field.latent);raw=head.decoder(z)
 if head.residual is not None:raw+=head.residual(z)
 offset=head.identity_selection['trusted_concept_start'];decoded=raw[:,offset:offset+G].clamp(1e-6,1-1e-6)
 route,accepted,_=head.route(e['text_embeddings'].float().to(a.device))
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);gid=hit['gaussian_ids'].long().to(a.device);pid=hit['pixel_ids'].long().to(a.device);w=hit['base_weights'].float().to(a.device);pm=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/pm[pid].clamp_min(1e-8),(pixels,n)).coalesce();mass=torch.sparse.sum(mat,dim=0).to_dense();seen=mass>1e-6
  ids=[j for j,g in enumerate(groups) if vi in g['views']];masks=torch.stack([e['masks'][[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]].bool().any(0).flatten() for j in ids],1).to(a.device)
  old=torch.stack([e['region_probability'][[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]].float().amax(0) for j in ids],1).to(a.device);union=torch.sparse.mm(mat.T,masks.float())/mass[:,None].clamp_min(1e-8);union*=seen[:,None];total_old[:,ids]+=old;total_union[:,ids]+=union;den[:,ids]+=e['seen'][vi].to(a.device)[:,None]
  union_error.append(dict(frame=frame,max=float((union-old).abs().max()),mean=float((union-old).abs().mean()),positive_threshold_changed=int(((union>.5)!=(old>.5)).sum())))
  views.append((vi,ids,masks.cpu(),mat,mass,seen));print(a.scene,'lift',frame,flush=True)
 rows=[]
 for vi,ids,mcpu,mat,mass,seen in views:
  masks=mcpu.to(a.device);old=torch.stack([e['region_probability'][[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]].float().amax(0) for j in ids],1).to(a.device);union=torch.sparse.mm(mat.T,masks.float())/mass[:,None].clamp_min(1e-8);union*=seen[:,None];remaining=den.clone();remaining[:,ids]-=e['seen'][vi].to(a.device).float()[:,None];remaining.clamp_min_(0)
  observed=field.observed;coverage=torch.sparse.mm(mat,observed.float()[:,None])[:,0];valid=coverage>.5
  def render(value,known):
   known=known&observed[:,None];support=torch.sparse.mm(mat,known.float());score=torch.sparse.mm(mat,value*known)/support.clamp_min(1e-8);return (score>.5)&(support>.5),support
  oldrest=total_old.clone();oldrest[:,ids]-=old;newrest=total_union.clone();newrest[:,ids]-=union
  rendered={};supports={}
  for name,value,known in [('old_full',total_old/den.clamp_min(1),den>0),('union_full',total_union/den.clamp_min(1),den>0),('old_lvo',oldrest/remaining.clamp_min(1),remaining>0),('union_lvo',newrest/remaining.clamp_min(1),remaining>0),('decoded_in_sample',decoded,observed[:,None].expand(-1,G))]:rendered[name],supports[name]=render(value,known)
  cycle,_=render(union,seen[:,None].expand(-1,len(ids)));targets={}
  for ri in torch.where(e['proposal_views']==vi)[0].tolist():
   r=e['records'][ri];mask=e['masks'][ri].bool().flatten().to(a.device)
   if r['confidence']<.7 or int(mask.sum())<20:continue
   for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(mask))|mask
  def metrics(pred,target):
   inter=(pred&target).sum().float();return dict(iou=float(inter/(pred|target).sum().clamp_min(1)),precision=float(inter/pred.sum().clamp_min(1)),recall=float(inter/target.sum().clamp_min(1)))
  for qi,target in targets.items():
   gi=int(route[qi]);local=ids.index(gi) if gi in ids else None
   if local is None or len(groups[gi]['views'])<2:continue
   oracle=((masks&target[:,None]).sum(0)/(masks|target[:,None]).sum(0).clamp_min(1)).max()
   row=dict(frame=info['frames'][vi],phrase=e['phrases'][qi],query_index=qi,selected_group=gi,group_phrase=groups[gi]['phrase'],concept_accepted=bool(accepted[qi]),candidate_oracle_iou=float(oracle),selected_2d=metrics(masks[:,local],target),same_view_cycle=metrics(cycle[:,local],target),lvo_positive_support=float((supports['union_lvo'][:,gi][target]>.5).float().mean()))
   for name,pred in rendered.items():row[name]=metrics(pred[:,gi],target)
   rows.append(row)
  print(a.scene,'render',info['frames'][vi],len(rows),flush=True)
 names=['selected_2d','same_view_cycle',*rendered];macro={name:{k:sum(r[name][k] for r in rows)/len(rows) for k in ['iou','precision','recall']} for name in names};macro['candidate_oracle_iou']=sum(r['candidate_oracle_iou'] for r in rows)/len(rows);macro['lvo_positive_support']=sum(r['lvo_positive_support'] for r in rows)/len(rows)
 result=dict(schema='scope_transfer_stages.v1',source_only=True,benchmark_data_used=False,teacher_masks_are_not_gt=True,physical_identity_verified=False,target_view_excluded_for_lvo=True,decoder_saw_target_source_masks=True,protocol='same routed concept in every stage; source phrase union confidence>=.7 area>=20; full target pixel denominator; only concepts detected in target and another source view; missing detections unknown',inputs=inputs,implementation_sha256=sha256_file(__file__),macro=macro,rows=rows,union_order_difference=union_error)
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,flush=True)
