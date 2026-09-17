"""Trace decoded false positives to source visibility, conflicting votes and mask scope."""
import argparse,json,shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--boundary-radius',type=int,choices=[1,3],default=3);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/('scope_core_conflict_audit' if a.boundary_radius==3 else 'scope_core_conflict_radius1_audit');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');inputs={}
def load(name,digest):
 path=root/name;assert sha256_file(path)==digest;inputs[str(path)]=digest;return torch.load(path,weights_only=False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];e=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and e['geometry_binding']==auth['metadata']['frozen_geometry_binding'];pr=json.loads((root/'joint_supervised_image_residual/report.json').read_text());payload=load('joint_supervised_image_residual/field.pt',pr['field_sha256']);assert e['geometry_binding']==payload['geometry_binding'];field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False);groups=json.loads((root/'grounded_semantic_transfer96/report.json').read_text())['groups'];n=len(field.latent);G=len(groups);H,W=e['masks'].shape[-2:];P=H*W;views=[]
with torch.no_grad():
 head=field.grounded_transfer;z=head.features(field.latent);raw=head.decoder(z)+head.residual(z);offset=head.identity_selection['trusted_concept_start'];decoded=raw[:,offset:offset+G].clamp(1e-6,1-1e-6)
 maxmass=torch.zeros(n,G,device=a.device);num=torch.zeros_like(maxmass);den=torch.zeros_like(maxmass)
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);gid=hit['gaussian_ids'].long().to(a.device);pid=hit['pixel_ids'].long().to(a.device);w=hit['base_weights'].float().to(a.device);pm=torch.zeros(P,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/pm[pid].clamp_min(1e-8),(P,n)).coalesce();mass=torch.sparse.sum(mat,dim=0).to_dense();seen=mass>1e-6;mass=mass*seen
  ids=[j for j,g in enumerate(groups) if vi in g['views']];masks=torch.stack([e['masks'][[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]].bool().any(0) for j in ids]);union=torch.sparse.mm(mat.T,masks.flatten(1).T.float().to(a.device))/mass[:,None].clamp_min(1e-8);union*=seen[:,None]
  maxmass[:,ids]=torch.maximum(maxmass[:,ids],mass[:,None]);num[:,ids]+=union*mass[:,None];den[:,ids]+=mass[:,None];views.append((frame,ids,masks,mat,mass,union.cpu()))
  print(a.scene,'source',frame,flush=True)
 pos=torch.zeros(n,G,device=a.device,dtype=torch.int16);neg=torch.zeros_like(pos);anypos=torch.zeros_like(pos);anyneg=torch.zeros_like(pos);corepos=torch.zeros_like(pos);coreneg=torch.zeros_like(pos)
 for frame,ids,masks,mat,mass,ucpu in views:
  flat=masks.float().to(a.device)[:,None];dilate=F.max_pool2d(flat,2*a.boundary_radius+1,1,a.boundary_radius)[:,0];erode=1-F.max_pool2d(1-flat,2*a.boundary_radius+1,1,a.boundary_radius)[:,0];pc=torch.sparse.mm(mat.T,erode.flatten(1).T)/mass[:,None].clamp_min(1e-8);nc=torch.sparse.mm(mat.T,(1-dilate).flatten(1).T)/mass[:,None].clamp_min(1e-8)
  u=ucpu.to(a.device);strong=(mass[:,None]>=.1*maxmass[:,ids])&(mass[:,None]>1e-6);positive=u>.8;negative=u<.2;pos[:,ids]+=(strong&positive).short();neg[:,ids]+=(strong&negative).short();anypos[:,ids]+=((mass[:,None]>1e-6)&positive).short();anyneg[:,ids]+=((mass[:,None]>1e-6)&negative).short();corepos[:,ids]+=(strong&(pc>.8)).short();coreneg[:,ids]+=(strong&(nc>.8)).short()
 conflict=(pos>0)&(neg>0);weak_only_conflict=(anypos>0)&(anyneg>0)&~conflict;negative_only=(neg>0)&(anypos==0);unsupported=den<=1e-6;ambiguous=~(conflict|weak_only_conflict|negative_only|unsupported)
 coreconflict=(corepos>0)&(coreneg>0)
 strata=dict(strong_core_conflict=conflict&coreconflict,strong_boundary_conflict=conflict&~coreconflict,weak_vote_conflict=weak_only_conflict,strong_negative_no_positive=negative_only,unobserved=unsupported,other=ambiguous)
 assert all(not (v&u).any() for i,v in enumerate(strata.values()) for u in list(strata.values())[i+1:])
 # Counterfactual safety control removes decoder positives with no positive source vote.
 conservative=decoded.clone();conservative[negative_only]=torch.minimum(conservative[negative_only],torch.tensor(.5,device=a.device))
 rows=[];global_mass={k:0. for k in strata};global_fp=0.;boundary_fp=0;parent_fp=0;far_fp=0
 for frame,ids,mcpu,mat,mass,ucpu in views:
  masks=mcpu.to(a.device).flatten(1).T;coverage=torch.sparse.mm(mat,field.observed.float()[:,None]);valid=coverage>.5;score=torch.sparse.mm(mat,decoded[:,ids]*field.observed[:,None])/coverage.clamp_min(1e-8);pred=(score>.5)&valid
  cscore=torch.sparse.mm(mat,conservative[:,ids]*field.observed[:,None])/coverage.clamp_min(1e-8);cpred=(cscore>.5)&valid
  fp=pred&~masks;area=masks.sum(0).float();inter=masks.float().T@masks.float();parents=(inter/area[:,None].clamp_min(1)>=.9)&(area[None,:]>=2*area[:,None])&(area[None,:]<=4*area[:,None]);parent_region=(masks.float()@parents.T.float())>0
  dilated=F.max_pool2d(mcpu.float().to(a.device)[:,None],2*a.boundary_radius+1,1,a.boundary_radius)[:,0].flatten(1).T>0;near=fp&dilated;parent_out=fp&~dilated&parent_region;far=fp&~dilated&~parent_region
  boundary_fp+=int(near.sum());parent_fp+=int(parent_out.sum());far_fp+=int(far.sum())
  # Soft attribution sums to the decoded positive mass on false-positive pixels.
  attributed={}
  for name,mask in strata.items():
   rendered=torch.sparse.mm(mat,decoded[:,ids]*mask[:,ids]*field.observed[:,None])/coverage.clamp_min(1e-8);attributed[name]=(rendered*fp).sum(0);global_mass[name]+=float(attributed[name].sum())
  global_fp+=float((score*fp).sum())
  for ci,j in enumerate(ids):
   target=masks[:,ci]
   def metric(pp):
    tp=(pp&target).sum();return dict(iou=float(tp/(pp|target).sum().clamp_min(1)),precision=float(tp/pp.sum().clamp_min(1)),recall=float(tp/target.sum().clamp_min(1)),false_positive=int((pp&~target).sum()))
   rows.append(dict(frame=frame,group=j,phrase=groups[j]['phrase'],parent=metric(pred[:,ci]),no_positive_veto=metric(cpred[:,ci]),attributed_mass={k:float(v[ci]) for k,v in attributed.items()},boundary_fp=int(near[:,ci].sum()),parent_context_fp=int(parent_out[:,ci].sum()),far_other_fp=int(far[:,ci].sum())))
  print(a.scene,'trace',frame,flush=True)
 macro={arm:{m:sum(r[arm][m] for r in rows)/len(rows) for m in ['iou','precision','recall']} for arm in ['parent','no_positive_veto']};assert abs(sum(global_mass.values())-global_fp)<max(1,global_fp*1e-5)
 result=dict(schema='scope_conflict_audit.v1',source_only=True,benchmark_data_used=False,teacher_masks_are_not_gt=True,physical_identity_verified=False,diagnostic_only=True,source_masks_used_in_attribution=True,weak_visibility_relative_threshold=.1,positive_vote_threshold=.8,negative_vote_threshold=.2,mask_boundary_radius=a.boundary_radius,parent_containment=.9,parent_area_ratio=[2,4],observations=len(rows),macro=macro,fp_decoded_mass_fraction={k:v/global_fp for k,v in global_mass.items()},fp_pixel_partition=dict(boundary=boundary_fp,parent_context=parent_fp,far_other=far_fp),rows=rows,inputs=inputs,implementation_sha256=sha256_file(__file__),limitation='Strong conflicts do not distinguish wrong geometry from inconsistent teacher scope; parent context is a mask containment pattern, not physical part truth. Veto uses all source votes including target-view masks and is not deployable or heldout validation.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,result['fp_decoded_mass_fraction'],result['fp_pixel_partition'],flush=True)
