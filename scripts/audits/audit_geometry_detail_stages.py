"""Geometry-only diagnostic on mapping development views, never semantic loss."""
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scripts.repair_moge_geometry import load_views
torch.set_num_threads(4);device='cuda:1'
root=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914/figurines')
report=json.loads((root/'report.json').read_text())
prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
ids=[f for f in report['mapping_frames'] if prior.calibration[f]['accepted']][:3]
views=[v for v in load_views(prior,480,device) if v['frame'] in ids]
g=report['results']['independent_scale']
model=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=device)
binding=model.fingerprint();original_scales=model.scales.clone();rows=[]
def stats(m,v):
 h,w=v['depth'].shape
 image,alpha,_=rasterization(means=m.get_xyz(),quats=m.get_rotation(),scales=m.get_scaling(),
  opacities=m.get_opacity().reshape(-1),colors=torch.zeros(len(m.xyz),1,device=device),
  viewmats=torch.linalg.inv(v['pose'])[None],Ks=v['k'][None],width=w,height=h,
  near_plane=.001*prior.scale,far_plane=100*prior.scale,render_mode='RGB+ED',packed=False)
 d=image[0,...,-1]/prior.scale;ref=v['depth']/prior.scale
 valid=v['valid'];covered=valid&(alpha[0,...,0]>.1)&torch.isfinite(d)&(d>0)
 # Horizontal neighboring rays; reference depth discontinuity >3 cm.
 edge=(ref[:,1:]-ref[:,:-1]).abs()>.03
 known=valid[:,1:]&valid[:,:-1];both=covered[:,1:]&covered[:,:-1]
 selected=edge&known&both
 diff=(d[:,1:]-d[:,:-1]).abs();refdiff=(ref[:,1:]-ref[:,:-1]).abs()
 return {'coverage':float(covered.sum()/valid.sum()),'absrel':float(((d-ref).abs()/ref.clamp_min(1e-8))[covered].mean()),
  'reference_edge_pairs':int((edge&known).sum()),'edge_pair_coverage':float(selected.sum()/(edge&known).sum().clamp_min(1)),
  'edge_magnitude_mae_metres':float((diff-refdiff).abs()[selected].mean()),
  'median_edge_magnitude_ratio':float((diff/refdiff.clamp_min(1e-8))[selected].median())}
for v in views:
 entry={'frame':v['frame'],'role':'mapping_development_diagnostic','variants':{}}
 for factor in (.5,1.,2.):
  model.scales.copy_(original_scales);model.scales[:,:2]*=factor
  entry['variants']['fused_tangent_'+str(factor)]=stats(model,v)
 # Same-view depth round trip: a reference-dependent diagnostic, not a candidate.
 h,w=v['depth'].shape
 yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
 rays=torch.stack([xx+.5,yy+.5,torch.ones_like(xx)],-1).float()@torch.linalg.inv(v['k']).T
 xyz=(rays*v['depth'][...,None])@v['pose'][:3,:3].T+v['pose'][:3,3]
 normal=v['normals']@v['pose'][:3,:3].T
 if normal.shape != xyz.shape: raise ValueError('normal shape differs')
 good=v['valid']&(normal.norm(dim=-1)>.5)
 raw=FrozenGaussianGeometry(xyz[good],normal[good],tangent_sigma=.0075*prior.scale,normal_sigma=.001*prior.scale)
 entry['variants']['same_view_points_fixed_sigma_reference_dependent']=stats(raw,v)
 footprint=.35*v['depth'][good]/torch.sqrt(v['k'][0,0]*v['k'][1,1])
 raw.scales[:,:2]=footprint[:,None]
 raw.scales[:,2]=footprint*.1
 entry['variants']['same_view_points_pixel_sigma_reference_dependent']=stats(raw,v)

 rows.append(entry);print(json.dumps(entry),flush=True)
model.scales.copy_(original_scales);model.assert_binding(binding)
out=Path('paper/artifacts/geometry_detail_stages_pixel_footprint_20260915.json')
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps({'scene':'figurines','geometry_sha256':g['geometry_sha256'],'geometry_unchanged':True,
 'normal_note':'camera normals rotated into world coordinates before Gaussian construction',
 'limitations':'same-view round trip is reference-dependent; edge metric is horizontal only; no final heldout tuning',
 'results':rows},indent=2)+'\n')
