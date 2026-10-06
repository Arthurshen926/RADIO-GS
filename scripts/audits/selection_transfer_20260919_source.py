"""Audit frozen probabilities and actual selected-only outputs on source scopes.

Deterministic bounded diagnosis, no refitting and no benchmark inputs. Source
teacher masks are not physical object truth or an independent validation set.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
root=Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field';fp=root/'bounded_fallback_admission/field.pt';fr=json.loads(fp.with_name('report.json').read_text());assert sha256_file(fp)==fr['field_sha256'];payload=torch.load(fp,weights_only=False);model=ElementRetrievalField(**payload['architecture']).cuda();model.load_state_dict(payload['state_dict']);model.requires_grad_(False)
ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['source_only'] and not e['benchmark_data_used'] and e['geometry_binding']==payload['geometry_binding'];info=json.loads((root/'teachers.json').read_text());assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json');gr=json.loads((Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json').read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(payload['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);views=[]
for frame in info['train_frames'][:4]:
 targets={}
 for i,r in enumerate(e['records']):
  if r['frame']==frame and r['confidence']>=.7 and int(e['masks'][i].sum())>=20:
   for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(e['masks'][i],dtype=torch.bool))|e['masks'][i].bool()
 keys=sorted(targets);chosen=[keys[i] for i in np.unique(np.linspace(0,len(keys)-1,min(12,len(keys)),dtype=int))] if keys else []
 views.append((frame,{i:targets[i] for i in chosen}))
queries=sorted({i for _,t in views for i in t});scores={};rows=[]
with torch.no_grad():
 for i in queries:
  q=e['text_embeddings'][i:i+1].float().cuda();native=(model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q))).sigmoid();scores[i]=torch.cat([native,model.retrieve(q,'text')['scores']],1)*model.observed[:,None]
 for frame,targets in views:
  for qi,tcpu in targets.items():
   target=tcpu.cuda();h,w=target.shape;cam=prior.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;kwargs=dict(viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   colors=torch.cat([scores[qi],model.observed.float()[:,None]],1);render,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**kwargs);mass=render[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);prob=torch.where(valid[...,None],render[0,...,:2]/mass[...,None].clamp_min(1e-8),0);metrics={}
   for j,name in enumerate(['native','final']):
    ids=(scores[qi][:,j]>.5)&model.observed
    if ids.any():
     _,sa,_=rasterization(means=geo.xyz[ids],quats=geo.quaternions[ids],scales=geo.scales[ids],opacities=geo.opacities.flatten()[ids],colors=torch.ones(int(ids.sum()),1,device='cuda:0'),**kwargs);selected=(sa[0,...,0]*255+.5).floor()>10
    else:selected=torch.zeros_like(target)
    def iou(pred):return float((pred&target).sum()/(pred|target).sum().clamp_min(1))
    metrics[name]=dict(probability_iou=iou(prob[...,j]>.5),selected_iou=iou(selected),selected_elements=int(ids.sum()),selected_precision=float((selected&target).sum()/selected.sum().clamp_min(1)),selected_recall=float((selected&target).sum()/target.sum().clamp_min(1)))
   rows.append(dict(frame=frame,query_index=qi,phrase=e['phrases'][qi],metrics=metrics))
  print(a.scene,frame,flush=True)
 report=dict(source_only=True,benchmark_data_used=False,parameters_fitted=False,independent_validation=False,teacher_masks_not_physical_gt=True,sampling='First four source train views, 12 evenly spaced sorted query IDs each; disjoint from risk fitting last4 frames; parent reconstruction saw these views; not picked by quality',grid='native source teacher mask96x128',observations=rows,macro={name:{k:float(np.mean([r['metrics'][name][k] for r in rows])) for k in ['probability_iou','selected_iou','selected_precision','selected_recall']} for name in ['native','final']},field_sha256=fr['field_sha256'],evidence_sha256=sha256_file(ep),implementation_sha256=sha256_file(__file__))
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,report['macro'],flush=True)
