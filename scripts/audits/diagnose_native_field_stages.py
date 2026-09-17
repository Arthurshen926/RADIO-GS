"""Controlled native-crop / adapter / geometry / field development diagnostic.

Target-RGB branches are explicitly diagnostic, never deployment candidates.
The query-free fixed crop grid is identical for source and target images.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from gsplat import rasterization
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks,binary_iou
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--run',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.protocol.json').exists():raise FileExistsError(a.output)
 root=a.root;dev=a.device;info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());source=json.loads((root/'source_frames.json').read_text());identity=json.loads((root/'native_query_encoder.json').read_text());r=json.loads((root/a.run/'report.json').read_text());gr=json.loads(a.geometry_report.read_text())
 assert sha256_file(root/'responsibility.json')==info['authority_sha256'];assert sha256_file(a.geometry_report)==auth['metadata']['frozen_geometry_report_sha256']
 geometry=FrozenGaussianGeometry.load(gr['results']['independent_scale']['geometry'],expected_sha256=gr['results']['independent_scale']['geometry_sha256'],device=dev);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);field,_=load_element_retrieval_field(root/a.run/'field.pt',expected_sha256=r['field_sha256'],geometry=geometry,expected_query_encoder=identity,device=dev)
 if getattr(field,'native_decoder',None) is not None:raise ValueError('independent query-adapter diagnostic is not applicable to native decoder readout; compare decoded elements in native space')
 labels=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label');annotations,categories,sh,sw=load_labels(labels,a.scene);assert not set(annotations)&set(info['train_frames'])
 protocol={'schema':'native_stage_diagnostic.v1','scene':a.scene,'field_sha256':r['field_sha256'],'geometry_sha256':gr['results']['independent_scale']['geometry_sha256'],'train_frames':info['train_frames'],'target_frames':sorted(annotations),'queries':categories,'crop_grid':[6,8],'crop_extent_fraction':[.25,.5],'crop_assignment':'mean normalized descriptors over two scales, nearest grid-cell assignment','source_fusion':'raw contribution-weighted native descriptor mean, normalize after fusion','target_rgb_diagnostic_only':True,'target_masks_used_for_features':False,'metric':'AP/AUROC primary; sigmoid(10*cos-5), fixed .5 IoU diagnostic, not calibrated native membership','common_support':'all methods restricted to frozen learned field support; unsupported positives remain misses','width':480,'implementation_sha256':sha256_file(__file__),'label_files':{str(x):sha256_file(x) for x in sorted((labels/a.scene).glob('*.json'))}}
 a.output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 encoder=NativeThreeQueryEncoder(identity,dev);q=encoder.encode_text(categories).half().float();qa=F.normalize(field.query_adapters['text'](q),dim=-1).detach();qh,qw=auth['metadata']['feature_height'],auth['metadata']['feature_width'];image_records={}
 @torch.no_grad()
 def descriptors(frame):
  path=Path(source['image_dir'])/f'frame_{frame:05d}.jpg';image_records[str(frame)]={'path':str(path),'sha256':sha256_file(path)};im=Image.open(path).convert('RGB');crops=[]
  for y in range(6):
   for x in range(8):
    for scale in (.25,.5):
     cx=(x+.5)*im.width/8;cy=(y+.5)*im.height/6;w=scale*im.width;h=scale*im.height;left=min(max(0,round(cx-w/2)),im.width-round(w));top=min(max(0,round(cy-h/2)),im.height-round(h));crop=im.crop((left,top,left+round(w),top+round(h))).resize((384,384),Image.Resampling.BICUBIC);crops.append(pil_to_tensor(crop).float()/255)
  if encoder.vision is None:
   from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
   encoder.vision=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(dev))
  features=torch.cat([encoder.vision.encode(torch.stack(crops[i:i+8]).to(dev)).float() for i in range(0,len(crops),8)])
  features=F.normalize(features.reshape(48,2,-1).mean(1),dim=-1).reshape(6,8,-1).permute(2,0,1)
  return F.interpolate(features[None],size=(qh,qw),mode='nearest')[0].flatten(1).T.contiguous()
 with torch.no_grad():
  n=len(geometry.xyz);sums=torch.zeros(n,q.shape[1],device=dev);mass=torch.zeros(n,device=dev)
  for frame in info['train_frames']:
   features=descriptors(frame);vi=info['frames'].index(frame);record=auth['views'][vi];path=root/record['relative_path'];assert sha256_file(path)==record['sha256'];hit=torch.load(path,map_location=dev,weights_only=False);gid=hit['gaussian_ids'].long();pid=hit['pixel_ids'].long();weight=hit['base_weights'].float();mat=torch.sparse_coo_tensor(torch.stack([gid,pid]),weight,(n,qh*qw)).coalesce();sums+=torch.sparse.mm(mat,features);mass.scatter_add_(0,gid,weight);print('source',frame,flush=True)
  native=F.normalize(sums/mass[:,None].clamp_min(1e-8),dim=-1);del sums
  known=field.observed&(mass>1e-8);adapted=F.normalize(field.query_adapters['image'](native),dim=-1)
  responses={'source_native_fused':torch.sigmoid(10*(native@q.T)-5),'source_adapted_fused':torch.sigmoid(10*(adapted@qa.T)-5),'learned_field':field.retrieve(q,'text')['scores']};del native,adapted
  colors=torch.cat([responses[k]*known[:,None] for k in responses]+[known.float()[:,None]],dim=1);rows={k:[] for k in ['target_native_2d','target_adapted_2d',*responses]};height=round(sh*480/sw)
  query_cos=q@q.T;adapted_cos=qa@qa.T;upper=torch.triu(torch.ones_like(query_cos,dtype=torch.bool),diagonal=1);alignment={'query_pair_cosine_pearson':float(torch.corrcoef(torch.stack([query_cos[upper],adapted_cos[upper]]))[0,1]),'native_offdiag_cosine_mean':float(query_cos[upper].mean()),'adapted_offdiag_cosine_mean':float(adapted_cos[upper].mean())}
  for frame,objects in sorted(annotations.items()):
   features=descriptors(frame);direct=torch.sigmoid(10*(features@q.T)-5);projected=torch.sigmoid(10*(F.normalize(field.query_adapters['image'](features),dim=-1)@qa.T)-5)
   maps={k:F.interpolate(v.T.reshape(1,len(q),qh,qw),size=(height,480),mode='nearest')[0].permute(1,2,0) for k,v in [('target_native_2d',direct),('target_adapted_2d',projected)]}
   camera=prior.cameras[frame];intr=camera.intrinsic.to(dev).clone();intr[0]*=480/camera.width;intr[1]*=height/camera.height
   image,alpha,_=rasterization(means=geometry.get_xyz(),quats=geometry.get_rotation(),scales=geometry.get_scaling(),opacities=geometry.get_opacity().reshape(-1),colors=colors,viewmats=camera.world_to_camera.to(dev)[None],Ks=intr[None],width=480,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
   support=image[0,...,-1];valid=(support/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
   for i,k in enumerate(responses):maps[k]=image[0,...,i*len(q):(i+1)*len(q)]/support[...,None].clamp_min(1e-8)
   targets=ground_truth_masks(objects,categories,height,480,sh,sw)
   for ci,c in enumerate(categories):
    truth=targets[c].to(dev)
    if not truth.any():continue
    for stage,score in maps.items():
     score=torch.where(valid,score[...,ci],torch.zeros_like(support));iou,_,_=binary_iou(score>.5,truth);rows[stage].append({'frame':frame,'category':c,'iou':iou,**binary_ranking_metrics(score,truth),'positive_support_recall':float((truth&valid).sum()/truth.sum())})
   print('target',frame,flush=True)
  macro={k:{metric:sum(r[metric] for r in values)/len(values) for metric in ['iou','average_precision','auroc','positive_support_recall']} for k,values in rows.items()}
  output={'protocol':protocol,'protocol_sha256':sha256_file(a.output.with_suffix('.protocol.json')),'macro':macro,'observations':rows,'query_alignment':alignment,'image_records':image_records,'geometry_unchanged':True};geometry.assert_binding(auth['metadata']['frozen_geometry_binding']);a.output.write_text(json.dumps(output,indent=2)+'\n');print(macro,flush=True)

if __name__=='__main__':main()
