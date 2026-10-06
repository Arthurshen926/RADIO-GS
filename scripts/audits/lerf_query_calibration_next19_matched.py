"""Two source-query crossfit calibrators; frozen native semantics and scope."""
import argparse,json,sys,runpy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_threshold import fit_threshold_ridge
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.graph_transfer_20260919_structure_runtime import refine_structure_scores

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def predict(qtrain,ttrain,q):
 w,b=fit_threshold_ridge(qtrain,ttrain,1.);return (F.normalize(q.float(),dim=1)@w+b).clamp(ttrain.min(),ttrain.max())
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);out=Path('output/lerf_query_calibration_next19_matched')/a.scene;out.mkdir(parents=True,exist_ok=False)
 reg=dict(arms=['zmedian','zridge'],ridge_penalty=1.,near_duplicate_group_cosine=.95,stats='mean/std population over entire observed field, not minibatch; density weighted by element count',std_floor=1e-6,target_gt_fit=False,source_cutoff_targets='existing source-only exact rendered probability optimum',source_validation='leave-near-duplicate-query-group-out calibration; field and teacher masks not independent heldout',integration='only columns whose frozen final equals native within2e-5; scope unchanged; then same structure graph',forced_positive_selection=False)
 reg['diagnostic_only_matched_absolute_baselines']=True
 (out/'registration.json').write_text(json.dumps(reg,indent=2));root=Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field';payload=load(root/'bounded_fallback_admission/field.pt');sd=payload['state_dict'];model=ElementRetrievalField(**payload['architecture']).cuda();model.load_state_dict(sd);model.requires_grad_(False);obs=sd['observed'];meta=json.loads((root/'exact_render_query_threshold96/report.json').read_text());episode=load(root/'episodes.pt');raw=load(root/'source_grounded_text.pt');assert not raw['benchmark_labels_used'];assert sha256_file(root/'episodes.pt')==meta['episode_sha256'];assert sha256_file(root/'source_grounded_text.pt')==meta['raw_source_grounding_sha256'];assert raw['geometry_binding']==payload['geometry_binding'];qid=meta['query_indices'];q=F.normalize(episode['queries']['text'][qid].float(),dim=1);targets=torch.tensor(meta['history'][-1]['thresholds']);cos=model.native_cosine(q.cuda()).cpu();mu=cos[obs].mean(0);std=cos[obs].std(0,correction=0).clamp_min(1e-6);z=(targets-mu)/std;sim=q@q.T
 groups=list(range(len(q)))
 def find(i):
  while groups[i]!=i:i=groups[i]
  return i
 for i in range(len(q)):
  for j in range(i):
   if sim[i,j]>=.95:groups[find(i)]=find(j)
 groups=[find(i)for i in range(len(q))];fold={k:torch.empty(len(q))for k in ['absolute_loo','zmedian_loo','zridge_loo']}
 for i in range(len(q)):
  keep=torch.tensor([g!=groups[i]for g in groups]);assert keep.sum()>=2
  fold['absolute_loo'][i]=predict(q[keep],targets[keep],q[i:i+1])[0];fold['zmedian_loo'][i]=mu[i]+std[i]*z[keep].median();fold['zridge_loo'][i]=mu[i]+std[i]*predict(q[keep],z[keep],q[i:i+1])[0]
 scale=model.readout_log_scale['text'].clamp(-4,6).exp().cpu();source_scores={'current_in_sample':torch.sigmoid(scale*(cos-model.native_query_thresholds['text'](q.cuda()).cpu()))}
 source_scores['absolute_fit1']=torch.sigmoid(scale*(cos-predict(q,targets,q)[None]))
 source_scores['absolute_loo']=torch.sigmoid(scale*(cos-fold['absolute_loo'][None]))
 for v in source_scores.values():v*=obs[:,None]
 gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(payload['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 from gsplat import rasterization
 names=list(source_scores);source_rows=[];h,w=raw['probability'].shape[-2:]
 for vi,frame in enumerate(raw['frames']):
  target=raw['probability'][vi,qid]>.5;eligible=(raw['scores'][vi,qid]>=.7)&(target.flatten(1).sum(1)>=20);cam=prior.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;kw=dict(viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
  for qi in torch.where(eligible)[0].tolist():
   colors=torch.stack([source_scores[name][:,qi]for name in names],1).cuda();colors=torch.cat([colors,obs.cuda().float()[:,None]],1);im,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**kw);mass=im[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=torch.where(valid[...,None],im[0,...,:len(names)]/mass[...,None].clamp_min(1e-8),0);gt=target[qi].cuda();metrics={}
   for j,name in enumerate(names):
    selected=(source_scores[name][:,qi]>.5)&obs;ids=selected.cuda()
    if ids.any():
     _,aa,_=rasterization(means=geo.xyz[ids],quats=geo.quaternions[ids],scales=geo.scales[ids],opacities=geo.opacities.flatten()[ids],colors=torch.ones(int(ids.sum()),1,device='cuda'),**kw);mask=(aa[0,...,0]*255+.5).floor()>10
    else:mask=torch.zeros_like(gt)
    pp=response[...,j]>.5
    def iou(x):return float((x&gt).sum()/(x|gt).sum().clamp_min(1))
    metrics[name]=dict(probability_iou=iou(pp),selected_iou=iou(mask))
   source_rows.append(dict(frame=frame,query_id=qid[qi],query=str(raw['queries'][qid[qi]]),metrics=metrics))
  print(a.scene,'heldout source',frame,flush=True)
 source_report=dict(registration=reg,group_ids=groups,query_ids=qid,observation_count=len(source_rows),cutoff_mse={k:float((v-targets).square().mean())for k,v in fold.items()},observations=source_rows,macro={name:{metric:float(np.mean([r['metrics'][name][metric]for r in source_rows]))for metric in ['probability_iou','selected_iou']}for name in names},source_only=True,independent_test=False)
 (out/'source_report.json').write_text(json.dumps(source_report,indent=2));print(a.scene,'source',source_report['macro'],flush=True)

if __name__=='__main__':
 with torch.no_grad():main()
