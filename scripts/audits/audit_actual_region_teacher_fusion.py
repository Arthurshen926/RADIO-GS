"""Same actual source teacher, frozen geometry and pre-sigmoid ranking contract.

Native region fusion is a diagnostic construction, not an accuracy upper bound.
"""
import argparse,json,gc
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--run',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--support-mask',type=Path);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 def checked(p,h):
  if sha256_file(p)!=h:raise ValueError('asset hash mismatch')
 root=a.root;info=json.loads((root/'teachers.json').read_text());receipt=json.loads((root/'episodes.json').read_text());checked(root/'episodes.pt',receipt['sha256']);data=torch.load(root/'episodes.pt',map_location='cpu',weights_only=False);checked(root/'teachers.json',data['teacher_manifest_sha256']);auth=json.loads((root/'responsibility.json').read_text());checked(root/'responsibility.json',info['authority_sha256']);checked(a.geometry_report,auth['metadata']['frozen_geometry_report_sha256'])
 gr=json.loads(a.geometry_report.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);identity=json.loads((root/'native_query_encoder.json').read_text());r=json.loads((root/a.run/'report.json').read_text());field,_=load_element_retrieval_field(root/a.run/'field.pt',expected_sha256=r['field_sha256'],geometry=geo,expected_query_encoder=identity,device=a.device)
 if field.native_decoder is None:raise ValueError('native coupled field required')
 tr=info['teacher_files']['siglip'];checked(root/tr['path'],tr['sha256']);teacher=torch.load(root/tr['path'],map_location='cpu',weights_only=False)['descriptors'].float().to(a.device);trainviews={info['frames'].index(f) for f in info['train_frames']};regions=torch.tensor([i for i,v in enumerate(data['proposal_views']) if int(v) in trainviews]);weights=data['region_probability'][regions].float().to(a.device);mass=weights.sum(0);known=field.observed&(mass>1e-8)
 if a.support_mask is not None:
  support=torch.load(a.support_mask,weights_only=False);geo.assert_binding(support['geometry_binding']);fixed=support['observed'].to(a.device)
  if fixed.shape!=known.shape or fixed.dtype!=torch.bool or (fixed&~known).any():raise ValueError('fixed support exceeds source evidence')
  known=fixed
 labels=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label');annotations,categories,sh,sw=load_labels(labels,a.scene)
 if set(annotations)&set(info['train_frames']):raise ValueError('source and annotation overlap')
 encoder=NativeThreeQueryEncoder(identity,a.device);q=F.normalize(encoder.encode_text(categories).half().float(),dim=-1);del encoder;gc.collect();torch.cuda.empty_cache()
 with torch.no_grad():
  native=[];learned=[];compressed=[]
  _,_,vh=torch.linalg.svd(F.normalize(teacher[regions],dim=-1),full_matrices=False);basis=vh[:64]
  for start in range(0,len(geo.xyz),4096):
   end=min(start+4096,len(geo.xyz));t=weights[:,start:end].T@F.normalize(teacher[regions],dim=-1);native.append((F.normalize(t,dim=-1)@q.T+1)/2);compressed.append((F.normalize((t@basis.T)@basis,dim=-1)@q.T+1)/2);learned.append((F.normalize(field.native_decoder(field.latent[start:end]),dim=-1)@q.T+1)/2)
  scores={'native_source_region_fusion':torch.cat(native),'source_svd64_fusion':torch.cat(compressed),'learned_decoded_field':torch.cat(learned)};colors=torch.cat([x*known[:,None] for x in scores.values()]+[known.float()[:,None]],dim=1);height=round(sh*480/sw);rows={k:[] for k in scores}
  for frame,objects in sorted(annotations.items()):
   camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=480/camera.width;k[1]*=height/camera.height
   image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities[:,0],colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=480,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   support=image[0,...,-1];valid=(support/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);maps=image[0,...,:-1]/support[...,None].clamp_min(1e-8);truths=ground_truth_masks(objects,categories,height,480,sh,sw)
   for ci,c in enumerate(categories):
    truth=truths[c].to(a.device)
    if not truth.any():continue
    for j,key in enumerate(scores):
     score=torch.where(valid,maps[...,j*len(q)+ci],torch.zeros_like(support));rows[key].append({'frame':frame,'category':c,**binary_ranking_metrics(score,truth)})
 macro={k:{m:sum(x[m] for x in v)/len(v) for m in ['average_precision','auroc']} for k,v in rows.items()}
 out={'schema':'actual_region_teacher_native_fusion_ranking.v1','field_sha256':r['field_sha256'],'geometry_sha256':g['geometry_sha256'],'source_episode_sha256':receipt['sha256'],'teacher_sha256':tr['sha256'],'implementation_sha256':sha256_file(__file__),'source_regions':len(regions),'shared_valid_elements':int(known.sum()),'fixed_support_sha256':sha256_file(a.support_mask) if a.support_mask is not None else None,'shared_output_contract':'render mean of (native cosine+1)/2, no sigmoid or learned calibration; full geometry and identical observed support; unknown score zero','svd_basis':'64 directions fitted to normalized source region teacher descriptors only; no target data or benchmark queries used to fit basis','native_fusion':'source region_probability weighted normalized descriptors; no compression; not an accuracy ceiling','target_rgb_opened':False,'ground_truth_used_only_for_evaluation':True,'no_training_updates':True,'macro':macro,'observations':rows};a.output.write_text(json.dumps(out,indent=2)+'\n');print(macro)
if __name__=='__main__':main()
