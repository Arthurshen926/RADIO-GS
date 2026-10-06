"""Post-evaluation responsibility check for unchanged spatula projection."""
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from PIL import Image

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);key='lerf_waldo_kitchen';base=Path('output/region_seed_positive_next80')/key;new=Path('output/grounded_scope_next85/source')/key;b=load(base/'typed_readout.pt');c=load(new/'typed_readout.pt');i=c['texts'].index('spatula');bs=((b['selection_scores'][:,i]>.5)&b['selection_observed']).cuda();cs=((c['selection_scores'][:,i]>.5)&c['selection_observed']).cuda();bind=c['selection_geometry'];geo=FrozenGaussianGeometry.load(bind['path'],expected_sha256=bind['sha256'],device='cuda');data=load(new/'field.pt');gr=json.loads(Path(data['geometry_report']).read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);cam=prior.cameras[66];w,h=Image.open(new/'anchored/pipeline_calibrated/hierarchical_pipeline/frame_00066_spatula.png').size;k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
out=Path('output/next85/projection_diagnostic');out.mkdir(parents=True,exist_ok=True);results={};images={}
with torch.inference_mode():
 for name,selected in [('baseline',bs),('candidate',cs),('common',bs&cs),('added',cs&~bs),('removed',bs&~cs)]:
  _,alpha,_=rasterization(means=geo.xyz[selected],quats=geo.quaternions[selected],scales=geo.scales[selected],opacities=geo.opacities.flatten()[selected],colors=torch.ones(int(selected.sum()),1,device='cuda'),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
  raster=(alpha[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu();images[name]=raster;Image.fromarray(raster.numpy()).save(out/(name+'.png'));camera_xyz=geo.xyz[selected]@cam.world_to_camera.cuda()[:3,:3].T+cam.world_to_camera.cuda()[:3,3];uv=camera_xyz@k.T;uv=uv[:,:2]/camera_xyz[:,2:].clamp_min(1e-8);inside=(camera_xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);results[name]=dict(selected=int(selected.sum()),positive_pixels=int((raster>10).sum()),max_alpha=float(alpha.max()),centers_in_front=int((camera_xyz[:,2]>0).sum()),centers_in_image=int(inside.sum()))
 results['baseline_candidate_byte_difference']=int((images['baseline']!=images['candidate']).sum());results['candidate_saved_byte_difference']=int((images['candidate']!=torch.from_numpy(__import__('numpy').array(Image.open(new/'anchored/pipeline_calibrated/hierarchical_pipeline/frame_00066_spatula.png')))).sum())
(out/'report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
