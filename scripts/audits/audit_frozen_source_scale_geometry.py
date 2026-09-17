"""Audit full Gaussian contribution depth distributions on untouched point IDs."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file


def summary(values):
 x=torch.tensor(values,dtype=torch.float64)
 return dict(count=len(x),median=float(x.median()),p90=float(x.quantile(.9)),mean=float(x.mean())) if len(x) else dict(count=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--all-source-views',action='store_true');p.add_argument('--width',type=int,default=480);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
 gr=json.loads(a.geometry_report.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
 inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']};points,_=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
 rows=[];frames=[]
 groups_to_audit=[('mapping',gr['mapping_frames']),('fusion_heldout',gr['heldout_frames'])]
 if a.all_source_views:groups_to_audit.append(('rejected_source_views',sorted(set(prior.records)-set(gr['mapping_frames'])-set(gr['heldout_frames']))))
 with torch.no_grad():
  for split,frameids in groups_to_audit:
   for frame in frameids:
    cam=prior.cameras[frame];w=a.width;h=round(cam.height*w/cam.width);k=cam.intrinsic.clone();k[0]*=w/cam.width;k[1]*=h/cam.height
    refs=[]
    for pid,xy in zip(cam.point_ids.tolist(),cam.xy):
     if pid%10!=5 or pid not in points:continue
     x,y=(xy*torch.tensor([w/cam.width,h/cam.height])).floor().long().tolist();z=float(points[pid]@cam.world_to_camera[2,:3]+cam.world_to_camera[2,3])
     if 0<=x<w and 0<=y<h and z>0:refs.append((pid,y*w+x,z))
    if not refs:continue
    renderer=FeatureFieldRenderer(h,w,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]),near_plane=.001*prior.scale,far_plane=100*prior.scale).to(a.device)
    pose=cam.world_to_camera.to(a.device);hit=rasterize_single_view_contributions(geo,renderer,pose,height=h,width=w)
    wanted=torch.zeros(h*w,dtype=torch.bool,device=a.device);wanted[torch.tensor([r[1] for r in refs],device=a.device)]=True
    keep=wanted[hit['pixel_ids'].long()];pid=hit['pixel_ids'][keep].long();gid=hit['gaussian_ids'][keep].long();weight=hit['weights'][keep].float()
    z=(geo.xyz[gid]@pose[2,:3]+pose[2,3]);order=pid.argsort();pid,z,weight=pid[order].cpu(),z[order].cpu(),weight[order].cpu();del hit
    keys,counts=pid.unique_consecutive(return_counts=True);starts=torch.cat([torch.zeros(1,dtype=torch.long),counts.cumsum(0)[:-1]]);lookup={int(k):(int(st),int(n)) for k,st,n in zip(keys,starts,counts)}
    covered=0
    for point_id,pixel,reference in refs:
     record=dict(frame=frame,split=split,point_id=point_id,pixel=pixel,covered=False)
     if pixel in lookup:
      start,n=lookup[pixel];zz,ww=z[start:start+n],weight[start:start+n];mass=float(ww.sum());record['alpha']=mass
      if mass>=.5:
       covered+=1;ww=ww/ww.sum();mean=(zz*ww).sum();variance=((zz-mean).square()*ww).sum();idx=zz.argsort();cdf=ww[idx].cumsum(0)
       q=zz[idx][torch.searchsorted(cdf,torch.tensor([.1,.5,.9])).clamp_max(len(zz)-1)]
       record.update(covered=True,absrel=float((mean-reference).abs()/reference),centre_depth_cv=float(variance.sqrt()/mean),mass_within_5pct=float(ww[(zz-reference).abs()<.05*reference].sum()),mass_in_front_20pct=float(ww[zz<.8*reference].sum()),mass_behind_20pct=float(ww[zz>1.2*reference].sum()),interval_10_90_contains_reference=bool(q[0]<=reference<=q[2]),relative_interval_width=float((q[2]-q[0])/reference),contributing_elements=n)
     rows.append(record)
    frames.append(dict(frame=frame,split=split,candidates=len(refs),covered=covered));print(frame,covered,len(refs),flush=True)
 groups={}
 for split in [name for name,_ in groups_to_audit]:
  rr=[r for r in rows if r['split']==split];cc=[r for r in rr if r['covered']]
  groups[split]=dict(candidates=len(rr),covered=len(cc),coverage=len(cc)/max(1,len(rr)),unique_reference_points=len({r['point_id'] for r in rr}),**{key:summary([r[key] for r in cc]) for key in ['absrel','centre_depth_cv','mass_within_5pct','mass_in_front_20pct','mass_behind_20pct','relative_interval_width']})
 a.output.write_text(json.dumps(dict(schema='source_gaussian_depth_distribution_audit.v1',geometry_sha256=g['geometry_sha256'],geometry_report_sha256=sha256_file(a.geometry_report),source_prior_sha256=gr['source_report_sha256'],source_only=True,benchmark_data_used=False,point_split='point ID modulo10 ==5',point_ids_excluded_from_calibration=prior.report.get('schema') in ('radio_gs.independent_source_scale_prior.v1','radio_gs.local_source_scale_prior.v1'),width=a.width,alpha_gate=.5,groups=groups,views=frames,rows=rows,limitations='Sparse source SfM reference and existing poses; pixel-floor quantization remains; centre-depth contribution distribution does not integrate within-Gaussian thickness; not dense independent GT.',implementation_sha256=sha256_file(__file__)),indent=2)+'\n');print(groups,flush=True)
if __name__=='__main__':main()
