"""Matched full-scene mask reconstruction on old/new NVOS geometry.

No feature/index transfer across carriers. All observed support is recomputed
from legal source rays. Frozen positive-reference video evidence is identical.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--carrier',choices=['old','new'],required=True);p.add_argument('--steps',type=int,default=120);p.add_argument('--fit-scale',type=int,default=1);p.add_argument('--tracks-base',default='output/source_video_next43');p.add_argument('--output-base',default='output/nvos_transfer_next72');a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 data=load(Path('output/taskwise_scope_composition_next38')/('nvos_'+('horns_center' if a.scene=='horns_left' else a.scene))/'field.pt');root=Path(data['root']);gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];h*=a.fit_scale;w*=a.fit_scale;g=gr['results']['independent_scale'];path=Path(g['geometry']) if a.carrier=='old' else Path('output/nvos_carrier_next72/adaptive_allrgb')/('horns_center' if a.scene=='horns_left' else a.scene)/'geometry.pt';geo=FrozenGaussianGeometry.load(path,expected_sha256=sha256_file(path),device='cuda');n=len(geo.xyz)
 tracks=load(Path(a.tracks_base)/a.scene/'tracks.pt');examples=[];observed=torch.zeros(n,dtype=torch.bool,device='cuda');source_frames=[]
 manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text());scene=next(r for r in manifest['scenes'] if r['scene_id']==a.scene);legal={int(r['rgb_sorted_index']) for r in scene['training_frames'] if r['frame_id'] not in scene['evaluation_frame_ids'] and r['frame_id'] not in scene['excluded_training_frame_ids']}-set(info['validation_frames']);legal.add(tracks['reference_frame'])
 for f in sorted(legal&set(gp.cameras)):
  cam=gp.cameras[f];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale);hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['weights'].float().cuda();observed[gid[weight>1e-8]]=True
  if f not in tracks['frames'] or tracks['frames'][f]['object_logit']<=0:continue
  total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight);op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/total[pid].clamp_min(1e-8),(h*w,n)).coalesce();record=tracks['frames'][f];y=F.interpolate(record['logits'].float()[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False).sigmoid();binary=(y>.5).float();positive=(-F.max_pool2d(-binary,2*a.fit_scale+1,1,a.fit_scale)>.5).flatten()[:,None];negative=(F.max_pool2d(binary,2*a.fit_scale+1,1,a.fit_scale)<.5).flatten()[:,None];known=(positive.float()+negative.float()*float(record.get('negative_reliability',1.)))*(total[:,None]>1e-8);examples.append((op,op.T.coalesce(),y.flatten()[:,None],known));source_frames.append(f)
 value=torch.full((n,1),.5,device='cuda');den=torch.zeros_like(value)
 for op,adj,y,known in examples:den+=torch.sparse.mm(adj,known.float())
 with torch.no_grad():
  for step in range(a.steps):
   grad=torch.zeros_like(value)
   for op,adj,y,known in examples:grad+=torch.sparse.mm(adj,(y-torch.sparse.mm(op,value)).clamp(-.25,.25)*known)
   value.add_(.95*grad/den.clamp_min(1e-8)).clamp_(0,1)
  value[(~observed)[:,None]|(den<=1e-8)]=0
  target=next(f for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids']);report=json.loads((Path('output/nvos_reference_completion_next19')/a.scene/'report.json').read_text());qr=json.loads((Path(report['query_root'])/a.scene/'receipt.json').read_text());width,height=qr['prompt_raster_wh'];cam=gp.cameras[int(target['canonical_index'])];k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
  image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=torch.cat([value,observed.float()[:,None]],1),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False);mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
 out=Path(a.output_base)/a.carrier/a.scene;out.mkdir(parents=True,exist_ok=True);torch.save(dict(elements=value.cpu(),image_observed=observed.cpu(),geometry=str(path),response=response),out/'prediction.pt')
 gt=np.asarray(Image.open(target['gt_mask_path']));truth=torch.from_numpy(np.any(gt>127,axis=2) if gt.ndim==3 else gt>127);response=torch.from_numpy(cv2.resize(response.numpy(),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR));binary=response>=.5;metrics=dict(iou=float((binary&truth).sum()/(binary|truth).sum().clamp_min(1)),**binary_ranking_metrics(response.flatten(),truth.flatten()));row=dict(scene=a.scene,carrier=a.carrier,metrics=metrics,source_frames=source_frames,tracks_base=a.tracks_base,source_observation_frames=sorted(legal&set(gp.cameras)),observed_elements=int(observed.sum()),elements=n,steps=a.steps,source_fit_hw=[h,w],boundary_radius=a.fit_scale,output_operator='full-scene observed-support-normalized response',target_rgb_used=False,old_element_indices_transferred=False,structural_features_used=False,no_query_evidence='abstain to zero response, not a training negative',seconds=time.monotonic()-start,control='source-only zero-prior mask fit; evidence identified by tracks_base; NOT retained full multimodal pipeline');(out/'report.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
if __name__=='__main__':main()
