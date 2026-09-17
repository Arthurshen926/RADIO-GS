"""Exact element-logit bound ceiling on authorized reference scribbles only."""
import argparse,json,sys,os
from pathlib import Path
import torch,numpy as np
from PIL import Image
from gsplat import rasterization
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
mp=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json');m=json.loads(mp.read_text());s=next(s for s in m['scenes'] if s['scene_id']==a.scene)
forbidden={str(Path(f[k]).resolve()) for f in s['frames'] if f['frame_id'] in s['evaluation_frame_ids'] for k in ['rgb_path','annotation_rgb_path','gt_mask_path'] if f.get(k)}
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise RuntimeError('target access forbidden in signed feasibility audit')
sys.addaudithook(guard)
qrpath=Path('/mnt/pool/sqy/results/RADIO-GS/output/nvos_reference_sam_queries_20260915')/a.scene/'receipt.json';qr=json.loads(qrpath.read_text());crop=qr['image_query_crop'];assert sha256_file(crop['path'])==crop['sha256']
identity=json.loads((a.root/'native_query_encoder.json').read_text());enc=NativeThreeQueryEncoder(identity,a.device);q=enc.encode_image(Image.open(crop['path']).convert('RGB'));enc.vision=None
import gc;gc.collect();torch.cuda.empty_cache()
gr=json.loads(a.geometry_report.read_text());r=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(r['geometry'],expected_sha256=r['geometry_sha256'],device=a.device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);ref=next(f for f in s['frames'] if f['frame_id']==s['prompt']['frame_id']);cam=prior.cameras[int(ref['canonical_index'])];w,h=qr['prompt_raster_wh'];k=cam.intrinsic.to(a.device).clone();k[0]*=w/cam.width;k[1]*=h/cam.height;pose=cam.world_to_camera.to(a.device)
signs={}
for sign in ['positive','negative']:
 r=qr['official_scribbles'][sign];assert sha256_file(r['path'])==r['sha256'];mask=np.asarray(Image.open(r['path']));mask=np.any(mask>127,axis=2) if mask.ndim==3 else mask>127;signs[sign]=torch.from_numpy(mask).to(a.device)
rows={}
with torch.no_grad():
 for name in ['nvos96_protected_imagecal_balanced','image_supervised_native_residual','nvos96_protected_imagecal_balanced_measured_support512','image_supervised_native_residual_measured_support512']:
  report=json.loads((a.root/name/'report.json').read_text());field,_=load_element_retrieval_field(a.root/name/'field.pt',expected_sha256=report['field_sha256'],geometry=geo,expected_query_encoder=identity,device=a.device);logit=field.logits(q,'image');observed=field.observed_for('image').float()[:,None];bounds=[8.,16.,32.];colors=torch.cat([v*observed for bound in bounds for v in [(logit-bound).sigmoid(),(logit+bound).sigmoid()]]+[observed],1)
  image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=pose[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
  mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);values=image[0,...,:-1]/mass[...,None].clamp_min(1e-8);pos=signs['positive']&valid;neg=signs['negative']&valid;rr=[]
  for i,b in enumerate(bounds):
   impossible_pos=pos&(values[...,2*i+1]<.5);impossible_neg=neg&(values[...,2*i]>=.5)
   rr.append(dict(bound=b,impossible_supported_positive=int(impossible_pos.sum()),impossible_supported_negative=int(impossible_neg.sum()),minimum_balanced_supported_error=.5*(float(impossible_pos.sum()/pos.sum())+float(impossible_neg.sum()/neg.sum()))))
  rows[name]=dict(field_sha256=report['field_sha256'],supported_positive=int(pos.sum()),supported_negative=int(neg.sum()),unsupported_positive=int((signs['positive']&~valid).sum()),unsupported_negative=int((signs['negative']&~valid).sum()),bounds=rr)
a.output.write_text(json.dumps(dict(schema='signed_logit_correction_feasibility.v1',scene=a.scene,query_receipt_sha256=sha256_file(qrpath),geometry_report_sha256=sha256_file(a.geometry_report),rows=rows,target_rgb_or_gt_opened=False,map_modified=False,scope='Necessary per-pixel feasibility under any element correction within the bound. Passing is not joint feasibility of a shared query coefficient.',implementation_sha256=sha256_file(__file__)),indent=2)+'\n');print(rows)
