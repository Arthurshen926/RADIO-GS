"""Audit scope granularity and fit admission against rendered source retrieval benefit."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.extent_benefit_gate import extent_agreement_features
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--parent',default='joint_supervised_image_residual');p.add_argument('--name',default='counterfactual_extent_benefit_admission');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
 root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root/a.name;out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();impl={}
 for f in [Path(__file__),Path('radio_gs/models/element_retrieval_field.py'),Path('radio_gs/models/extent_benefit_gate.py'),Path('radio_gs/models/grounded_semantic_transfer.py')]:shutil.copy2(f,snap/f.name);impl[str(f)]=sha256_file(f)
 parent=root/a.parent;pr=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==pr['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
 field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
 ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and e['geometry_binding']==payload['geometry_binding'] and e['source_only'] and not e['benchmark_data_used']
 groups=json.loads((root/'grounded_semantic_transfer96/report.json').read_text())['groups'];h,w=e['masks'].shape[-2:];pixels=h*w;n=len(field.latent);views=[];inputs={str(ep):sha256_file(ep)}
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);rec=auth['views'][vi];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location=a.device,weights_only=False)
  gid,pid,weight=hit['gaussian_ids'].long(),hit['pixel_ids'].long(),hit['base_weights'].float();full=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,weight);keep=field.observed[gid];gid,pid,weight=gid[keep],pid[keep],weight[keep];mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();targets={}
  for ri in torch.where(e['proposal_views']==vi)[0].tolist():
   record=e['records'][ri];mask=e['masks'][ri].bool().flatten()
   if record['confidence']<.7 or int(mask.sum())<20:continue
   for qi in record['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(mask))|mask
  group_masks={j:e['masks'][ids].bool().any(0).flatten() for j,g in enumerate(groups) if (ids:=[i for i in g['regions'] if int(e['proposal_views'][i])==vi])}
  views.append((vi,mat,valid,targets,group_masks))
 records=[];features={};head=field.grounded_transfer;P=len(e['text_embeddings']);G=len(groups)
 with torch.no_grad():
  z=head.features(field.latent);raw=head.decoder(z)
  if head.residual is not None:raw=raw+head.residual(z)
  offset=head.identity_selection['trusted_concept_start'];group_probability=raw[:,offset:offset+G].clamp(1e-6,1-1e-6);group_logits=torch.logit(group_probability)
  group_render={vi:((torch.sparse.mm(mat,group_probability*field.observed[:,None])>.5)&valid[:,None]) for vi,mat,valid,_,_ in views}

 with torch.no_grad():
  for start in range(0,len(e['text_embeddings']),32):
   q=e['text_embeddings'][start:start+32].float().to(a.device);cos=field.native_cosine(q);threshold=field.native_query_thresholds['text'](q);scale=field.readout_log_scale['text'].clamp(-4,6).exp();native=(scale*(cos-threshold)).sigmoid();candidate=field.logits(q,'text');route,concept,_=head.route(q)
   _,accepted=head(q,field.latent,native_threshold=threshold,native_support=cos>threshold,observed=field.observed)
   x=extent_agreement_features(cos,threshold,candidate,field.observed,concept)
   for j in range(len(q)):features[start+j]=x[j].cpu()
   responses=torch.cat([native,candidate.sigmoid()],1)*field.observed[:,None]
   for vi,mat,valid,targets,group_masks in views:
    rendered=(torch.sparse.mm(mat,responses)>.5)&valid[:,None]
    for qi,targetcpu in targets.items():
     j=qi-start
     if not 0<=j<len(q) or not accepted[j]:continue
     target=targetcpu.to(a.device);values=[]
     for k in [j,len(q)+j]:
      pred=rendered[:,k];values.append(float((pred&target).sum()/(pred|target).sum().clamp_min(1)))
     group=int(route[j]);scope='unknown';overlap=None
     if group in group_masks:
      gm=group_masks[group];inter=int((gm&targetcpu).sum());ta=int(targetcpu.sum());ga=int(gm.sum());overlap=inter/max(ta+ga-inter,1);coverage=inter/max(ta,1);precision=inter/max(ga,1)
      scope=('aligned' if overlap>=.5 else 'selected_parent' if coverage>=.8 and precision<.5 else 'selected_part' if precision>=.8 and coverage<.5 else 'disjoint' if overlap<=.1 else 'partial_overlap')
     records.append(dict(query_index=qi,base_query_index=qi,kind='routed',phrase=e['phrases'][qi],view=vi,native_iou=values[0],extent_iou=values[1],gain=values[1]-values[0],concept_accepted=bool(concept[j]),selected_group=group,selected_scope_relation=scope,scope_iou=overlap))

   similarity=torch.nn.functional.normalize(q,dim=-1)@torch.nn.functional.normalize(head.prototypes,dim=-1).T
   similarity.scatter_(1,route[:,None],-torch.inf)
   neighbors=similarity.topk(min(3,G-1),dim=1).indices
   for j in range(len(q)):
    if not accepted[j]:continue
    qi=start+j
    for group in neighbors[j].tolist():
     synthetic=P+qi*G+group
     fx=extent_agreement_features(cos[:,j:j+1],threshold[j],group_logits[:,group:group+1],field.observed,torch.ones(1,device=a.device,dtype=torch.bool))
     features[synthetic]=fx[0].cpu()
     for vi,mat,valid,targets,group_masks in views:
      if qi not in targets or group not in group_masks:continue
      targetcpu=targets[qi];target=targetcpu.to(a.device);gm=group_masks[group]
      pred=((torch.sparse.mm(mat,native[:,j:j+1]*field.observed[:,None])[:,0]>.5)&valid)
      candidate_mask=group_render[vi][:,group]
      ni=float((pred&target).sum()/(pred|target).sum().clamp_min(1));ci=float((candidate_mask&target).sum()/(candidate_mask|target).sum().clamp_min(1))
      inter=int((gm&targetcpu).sum());ta=int(targetcpu.sum());ga=int(gm.sum());overlap=inter/max(ta+ga-inter,1);coverage=inter/max(ta,1);precision=inter/max(ga,1)
      relation=('aligned' if overlap>=.5 else 'selected_parent' if coverage>=.8 and precision<.5 else 'selected_part' if precision>=.8 and coverage<.5 else 'disjoint' if overlap<=.1 else 'partial_overlap')
      records.append(dict(query_index=synthetic,base_query_index=qi,kind='counterfactual',phrase=e['phrases'][qi],view=vi,native_iou=ni,extent_iou=ci,gain=ci-ni,concept_accepted=True,selected_group=group,selected_scope_relation=relation,scope_iou=overlap))
   print(a.scene,start,'records',len(records),flush=True)
 unique=sorted({r['query_index'] for r in records});assert len(unique)>=10
 base_ids=sorted({r['base_query_index'] for r in records});order=torch.tensor(base_ids)[torch.randperm(len(base_ids))];heldout=set(order[::5].tolist());base_of={r['query_index']:r['base_query_index'] for r in records};validation=sorted(heldout)
 x=torch.stack([features[q] for q in unique]);native=torch.tensor([sum(r['native_iou'] for r in records if r['query_index']==q)/sum(r['query_index']==q for r in records) for q in unique]);extent=torch.tensor([sum(r['extent_iou'] for r in records if r['query_index']==q)/sum(r['query_index']==q for r in records) for q in unique]);is_valid=torch.tensor([base_of[q] in heldout for q in unique]);center=x[~is_valid].mean(0);scale=x[~is_valid].std(0).clamp_min(1e-4);z=torch.cat([(x-center)/scale,torch.ones(len(x),1)],1);target=extent-native
 sample_weight=torch.tensor([1. if q<P else 1./sum(k>=P and base_of[k]==base_of[q] for k in unique) for q in unique]);routed=torch.tensor([q<P for q in unique])
 def avg(v,mask):return float((v[mask]*sample_weight[mask]).sum()/sample_weight[mask].sum())
 candidates=[];coeffs=[]
 for penalty in [.1,1.,10.]:
  zt=z[~is_valid];yt=target[~is_valid];reg=torch.eye(z.shape[1])*penalty;reg[-1,-1]=0;wt=sample_weight[~is_valid];wt=wt/wt.sum();coef=torch.linalg.solve(zt.T@(zt*wt[:,None])+reg,zt.T@(yt*wt));allow=z@coef>0;utility=torch.where(allow,extent,native);candidates.append(dict(penalty=penalty,train_iou=avg(utility,~is_valid),validation_iou=avg(utility,is_valid),validation_extent_iou=avg(extent,is_valid),validation_native_iou=avg(native,is_valid),validation_routed_iou=avg(utility,is_valid&routed),validation_routed_parent_iou=avg(extent,is_valid&routed),validation_allowed=int(allow[is_valid].sum())));coeffs.append(coef)
 admissible=[i for i,c in enumerate(candidates) if c['validation_routed_iou']>=c['validation_routed_parent_iou']-1e-6];best=max(admissible or range(len(candidates)),key=lambda i:(candidates[i]['validation_iou'],candidates[i]['penalty']));arch=dict(payload['architecture'],extent_benefit_admission=True);new=ElementRetrievalField(**arch);missing=new.load_state_dict(payload['state_dict'],strict=False);assert set(missing.missing_keys)=={'extent_benefit_admission.center','extent_benefit_admission.scale','extent_benefit_admission.coefficient'} and not missing.unexpected_keys
 new.extent_benefit_admission.center.copy_(center);new.extent_benefit_admission.scale.copy_(scale);new.extent_benefit_admission.coefficient.copy_(coeffs[best]);state=new.state_dict()
 for k,v in payload['state_dict'].items():assert torch.equal(state[k],v)
 torch.save(dict(payload,architecture=arch,state_dict=state),out/'field.pt')
 strat={}
 for relation in sorted({r['selected_scope_relation'] for r in records}):
  rr=[r for r in records if r['selected_scope_relation']==relation and r['concept_accepted']]
  if rr:strat[relation]=dict(observations=len(rr),native_iou=sum(r['native_iou'] for r in rr)/len(rr),extent_iou=sum(r['extent_iou'] for r in rr)/len(rr),harmful=sum(r['gain']<0 for r in rr))
 report=dict(schema='counterfactual_extent_benefit_admission.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],benchmark_data_used=False,source_only=True,all_parent_tensors_exact=True,inputs=inputs,implementations=impl,selected=candidates[best],source_routed_validation_passed=bool(admissible),counterfactual_policy='Top three nonrouted semantic neighbor concepts, only coobserved masks; actual route and mean counterfactual family equal total weight per phrase; phrase family held out together',candidates=candidates,granularity=strat,observations=records,heldout_phrases=[e['phrases'][q] for q in validation],validation_scope='Phrases excluded from gate fitting only; base field and extent heads already saw source images/masks. Teacher masks are not independent object truth.',feature_names=['native_extent_iou','extent_precision','native_recall','log_mass_ratio','inside_native_margin','outside_native_margin','native_peak_margin','concept_accepted'],selection_metric='mean per-phrase rendered source IoU; full pixel denominator')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,report['selected'],strat,flush=True)
if __name__=='__main__':main()
