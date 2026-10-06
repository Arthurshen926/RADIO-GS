"""Source-only fixed-view resolution control, not inferred semantic feature filling."""
import json,inspect
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4);out=Path('output/lerf_resolution_support_next19');out.mkdir(exist_ok=False)
registration={'source_only':True,'gt_read':False,'scenes':['figurines','ramen','teatime','waldo_kitchen'],'source_raster':[192,256],'original_raster':[96,128],'training_views_unchanged':True,'teacher_mask_resampling':'nearest fixed cached masks, no new teacher inference','geometry_unchanged':True,'semantic_features_changed':False,'purpose':'Distinguish raster undersampling from missing source-view coverage before any semantic rebuild','implementation_sha256':sha256_file(__file__),'compositor_sha256':sha256_file(inspect.getsourcefile(inspect.unwrap(rasterize_single_view_contributions)))}
(out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n');rows=[]
with torch.no_grad():
 for scene in registration['scenes']:
  root=Path('output/depth_opacity_pipeline_20260918')/scene/'reconstruction/field';info=json.loads((root/'teachers.json').read_text());ep=torch.load(root/'episodes.pt',map_location='cpu',weights_only=False);assert sha256_file(root/'episodes.pt')==json.loads((root/'episodes.json').read_text())['sha256'];fp=root/'bounded_fallback_admission/field.pt';p=torch.load(fp,map_location='cpu',weights_only=False);obs=p['state_dict']['observed'].cuda();gp=Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(p['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);h,w=192,256;n=len(obs);mass=torch.zeros(n,device='cuda');seen=torch.zeros(n,device='cuda',dtype=torch.int32)
  assert set(info['train_frames'])<=set(gr['mapping_frames']);assert not set(info['train_frames'])&set(info['validation_frames'])
  for frame in info['train_frames']:
   vi=info['frames'].index(frame);ix=torch.where(ep['proposal_views']==vi)[0];mask=F.interpolate(ep['masks'][ix].any(0)[None,None].float(),size=(h,w),mode='nearest')[0,0].flatten().cuda();c=prior.cameras[frame];k=c.intrinsic.clone();k[0]*=w/c.width;k[1]*=h/c.height;renderer=FeatureFieldRenderer(h,w,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]),near_plane=.001*prior.scale,far_plane=100*prior.scale).cuda();hit=rasterize_single_view_contributions(geo,renderer,c.world_to_camera.cuda(),height=h,width=w);pid,gid,val=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['weights'].float();den=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,val);v=torch.zeros(n,device='cuda').scatter_add_(0,gid,val/den[pid].clamp_min(1e-8)*mask[pid]);mass+=v;seen+=(v>1e-6).int();print(scene,frame,'new observed',int(((mass>1e-6)&~obs).sum()),flush=True)
  dst=out/(scene+'.pt');torch.save({'geometry_binding':p['geometry_binding'],'observed':obs.cpu(),'measured_192':(mass>1e-6).cpu(),'multiview_192':(seen>=2).cpu(),'source_mass':mass.cpu()},dst);row={'scene':scene,'old_observed':int(obs.sum()),'new_measured':int(((mass>1e-6)&~obs).sum()),'new_multiview':int(((seen>=2)&~obs).sum()),'old_not_measured_at_192':int((obs&(mass<=1e-6)).sum()),'field_sha256':sha256_file(fp),'geometry_sha256':g['geometry_sha256'],'output_sha256':sha256_file(dst)};rows.append(row);(out/'report.json').write_text(json.dumps({'complete':len(rows)==4,'source_only':True,'rows':rows,'registration_sha256':sha256_file(out/'registration.json')},indent=2)+'\n');print(row,flush=True);del geo,p,hit;torch.cuda.empty_cache()
