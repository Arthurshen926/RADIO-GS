"""Post-hoc attribution of fixed predicted silhouettes; GT never changes outputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);key='lerf_waldo_kitchen';out=Path('output/next93');out.mkdir(exist_ok=True);rows=[]
for base in ['output/description_scope_next91','output/medial_description_scope_next92']:
 root=Path(base)/key;r=load(root/'typed_readout.pt');qi=r['texts'].index('spatula');selection=((r['selection_scores'][:,qi]>.5)&r['selection_observed']).cuda();binding=r['selection_geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda');data=load(root/'field.pt');gr=json.loads(Path(data['geometry_report']).read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);cam=prior.cameras[66];metrics=json.loads((root/'anchored/pipeline_calibrated/evaluation.json').read_text());obj=next(x for x in metrics['metrics']['hierarchical_pipeline']['selected']['objects'] if x['query']=='spatula');gt=torch.from_numpy(load_binary_mask(Path(obj['gt_path']),threshold=10)).cuda();h,w=gt.shape;k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*prior.scale,far_plane=100*prior.scale)
 with torch.inference_mode():
  subset=SimpleNamespace(get_xyz=lambda:geo.xyz[selection],get_rotation=lambda:geo.quaternions[selection],get_scaling=lambda:geo.scales[selection],get_opacity=lambda:geo.opacities[selection]);hit=rasterize_single_view_contributions(subset,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['weights'].float();visible=(hit['rendered_alpha'].flatten()*255+.5).floor()>10;foreground=gt.flatten()[pid];keep=visible[pid];fg=torch.zeros(len(selection),device='cuda');bg=fg.clone();fg.index_add_(0,gid,weight*foreground*keep);bg.index_add_(0,gid,weight*~foreground*keep);total=fg+bg;fraction=fg/total.clamp_min(1e-12);groups={}
  for name,mask in [('no_target_contribution',(fg<=1e-10)&(bg>0)),('mixed',(fg>1e-10)&(bg>1e-10)),('target_only',(fg>0)&(bg<=1e-10))]:groups[name]=dict(gaussians=int(mask.sum()),foreground_alpha_mass=float(fg[mask].sum()),background_alpha_mass=float(bg[mask].sum()),background_mass_fraction=float(bg[mask].sum()/bg.sum().clamp_min(1e-12)))
  reconstructed=(hit['rendered_alpha']*255+.5).floor().clamp(0,255).byte().cpu().numpy();saved=np.array(__import__('PIL').Image.open(obj['pred_path']));rows.append(dict(base=base,query='spatula',frame=66,groups=groups,visible_gaussians=int((total>0).sum()),replay_byte_difference=int((reconstructed!=saved).sum()),diagnostic_only=True,scope='visible contribution attribution in one scored image; not intrinsic 3D identity or oracle selection'))
(out/'extent_attribution.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
