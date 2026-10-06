"""Transport official positive reference scribbles through visible geometry."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.utils.immutable_artifacts import sha256_file

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-base',default='output/reference_surface_next74');p.add_argument('--ownership',choices=['broad','dominant'],default='dominant');a=p.parse_args();torch.set_num_threads(4)
 key='nvos_'+('horns_center' if a.scene=='horns_left' else a.scene);data=load(Path('output/taskwise_scope_composition_next38')/key/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());gr=json.loads(Path(data['geometry_report']).read_text());gp=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);path=Path('output/nvos_carrier_next72/adaptive_allrgb')/key.removeprefix('nvos_')/'geometry.pt';geo=FrozenGaussianGeometry.load(path,expected_sha256=sha256_file(path),device='cuda');n=len(geo.xyz)
 manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text());scene=next(s for s in manifest['scenes'] if s['scene_id']==a.scene)
 report=json.loads((Path('output/nvos_reference_completion_next19')/a.scene/'report.json').read_text());qr=json.loads((Path(report['query_root'])/a.scene/'receipt.json').read_text());ref=next(f for f in scene['frames'] if f['frame_id']==scene['prompt']['frame_id']);reference=int(ref['canonical_index']);positive=np.asarray(Image.open(qr['official_scribbles']['positive']['path']));positive=np.any(positive>127,axis=2) if positive.ndim==3 else positive>127
 width=512;height=round(positive.shape[0]*width/positive.shape[1]);pos=F.interpolate(torch.from_numpy(positive).float()[None,None].cuda(),size=(height,width),mode='area')[0,0].flatten()
 def intrinsic(cam):
  k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height;return k
 cam=gp.cameras[reference];k=intrinsic(cam);renderer=SimpleNamespace(scaled_intrinsics=lambda w,h:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
 hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=height,width=width,include_depths=False);gid=hit['gaussian_ids'].cuda().long();pid=hit['pixel_ids'].cuda().long();weight=hit['weights'].cuda().float();mass=torch.zeros(n,device='cuda').scatter_add_(0,gid,weight);support=torch.zeros_like(mass).scatter_add_(0,gid,weight*(pos[pid]>0));peak=torch.zeros(height*width,device='cuda').scatter_reduce_(0,pid,weight,reduce='amax',include_self=True);anchors=torch.zeros(n,dtype=torch.bool,device='cuda');anchors[gid[(weight>=peak[pid])&(pos[pid]>0)]]=True
 if a.ownership=='broad':anchors=(support/mass.clamp_min(1e-8)>.8)&(support>1e-6)
 paths={int(r['rgb_sorted_index']):r['rgb_path'] for r in scene['training_frames'] if r['frame_id'] not in scene['evaluation_frame_ids'] and r['frame_id'] not in scene['excluded_training_frame_ids'] and int(r['rgb_sorted_index']) not in info['validation_frames']};paths[reference]=qr['reference_rgb']['path'];rows=[]
 for frame,rgb in sorted(paths.items()):
  if frame not in gp.cameras:continue
  cam=gp.cameras[frame];image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=anchors.float()[:,None],viewmats=cam.world_to_camera.cuda()[None],Ks=intrinsic(cam)[None],width=width,height=height,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
  k=intrinsic(cam);renderer=SimpleNamespace(scaled_intrinsics=lambda w,h:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
  hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=height,width=width,include_depths=False);pid=hit['pixel_ids'].cuda().long();gid=hit['gaussian_ids'].cuda().long();weight=hit['weights'].cuda().float();peak=torch.zeros(height*width,device='cuda').scatter_reduce_(0,pid,weight,reduce='amax',include_self=True)
  owned=(weight>=peak[pid])&anchors[gid];score=torch.zeros(height*width,device='cuda').scatter_add_(0,pid[owned],weight[owned]).reshape(height,width)
  if a.ownership=='broad':
   evidence=image[0,...,0]/alpha[0,...,0].clamp_min(1e-8);score=evidence.masked_fill(~((evidence>.5)&(alpha[0,...,0]>.1)),0)
  anchor_pixels=torch.nonzero(score>0);anchor_weights=score[anchor_pixels[:,0],anchor_pixels[:,1]].cpu().tolist();anchor_pixels=anchor_pixels[:,[1,0]].cpu().tolist()
  points=[]
  for _ in range(3):
   i=int(score.argmax());y,x=divmod(i,width)
   if score[y,x]<=0:break
   points.append([x,y]);score[max(0,y-8):y+9,max(0,x-8):x+9]=0
  rows.append(dict(frame=frame,rgb=rgb,points=points,candidate_points=anchor_pixels,candidate_weights=anchor_weights))
 out=Path(a.output_base)/a.scene;out.mkdir(parents=True,exist_ok=True);(out/'anchors.json').write_text(json.dumps(dict(scene=a.scene,reference_frame=reference,raster_wh=[width,height],geometry=str(path),anchor_count=int(anchors.sum()),ownership=a.ownership,views=rows,target_rgb_used=False,negative_prompt_used=False,identity='official positive scribble, maximum-contribution owner in reference AND receiving source pixels'),indent=2));print(a.scene,int(anchors.sum()),sum(bool(r['points']) for r in rows),len(rows),flush=True)
if __name__=='__main__':main()
