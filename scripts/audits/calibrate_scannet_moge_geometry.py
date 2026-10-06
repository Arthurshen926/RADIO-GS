"""Calibrate source MoGe depths against supplied *unlabelled* ScanNet vertices.

Vertices/faces are declared reconstruction inputs. Semantic labels and queries
are unused. Vertex IDs modulo5 separate fit, admission and audit evidence.
The held-out metric is still registered-input consistency, not independent GT.
"""
import argparse,json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from radio_gs.geometry.mesh_visibility import rasterize_registered_mesh_depth
from radio_gs.geometry.registered_pinhole import load_registered_camera_table
from radio_gs.v4.geometry.depth_calibration import fit_constrained_affine_depth
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--points',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
 if a.points.name!='points3d.ply':raise ValueError('Only the declared unlabelled reconstruction input is accepted')
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ['.labels.ply','language_features']):raise RuntimeError('Semantic data forbidden in geometry calibration')
 sys.addaudithook(guard)
 authority=a.root/'moge_source_authority.json';auth=json.loads(authority.read_text());camera_path=Path(auth['registered_camera_authority']);assert sha256_file(camera_path)==auth['registered_camera_authority_sha256'];cameras=load_registered_camera_table(camera_path);manifest_path=a.root/'scene_manifest.json';manifest=json.loads(manifest_path.read_text());assert manifest['source_authority_sha256']==sha256_file(authority)
 ply=PlyData.read(a.points);vertices=ply['vertex'];xyz=torch.from_numpy(np.stack([vertices[k] for k in ['x','y','z']],1).copy()).float();assert torch.isfinite(xyz).all();ids=torch.arange(len(xyz));heldout=ids%5==0;admission=ids%5==1;calibrations=[];diagnostics=[]
 faces=np.stack(ply['face']['vertex_indices']);assert faces.shape[1]==3
 import nvdiffrast.torch as dr
 context=dr.RasterizeCudaContext(device=a.device)
 for rec in manifest['records']:
  frame=rec['frame_index'];c=cameras[frame];assert sha256_file(rec['prediction'])==rec['prediction_sha256'];payload=torch.load(rec['prediction'],map_location='cpu',weights_only=False);assert payload['model_checkpoint_sha256']==manifest['model_checkpoint_sha256'];depth=payload['point_map'][...,2].float();h,w=depth.shape;valid=payload['validity'].bool();k=c.intrinsic.clone();k[0]*=w/c.width;k[1]*=h/c.height;camera=xyz@c.world_to_camera[:3,:3].T+c.world_to_camera[:3,3];uv=camera@k.T;xy=uv[:,:2]/uv[:,2:].clamp_min(1e-8);pixel=xy.floor().long();inside=(camera[:,2]>0)&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h);pid=pixel[:,1]*w+pixel[:,0];zbuffer=torch.full((h*w,),torch.inf);zbuffer.scatter_reduce_(0,pid[inside],camera[inside,2],reduce='amin',include_self=True);safe=pid.clamp(0,h*w-1);visible=inside&(camera[:,2]<=zbuffer[safe]+.01)&valid.flatten()[safe]&torch.isfinite(depth.flatten()[safe])&(depth.flatten()[safe]>0)
  # Dense mesh visibility rejects vertices behind foreground surfaces; it never compares MoGe depth.
  mesh_depth=rasterize_registered_mesh_depth(camera.to(a.device),faces,k,h,w,context=context).cpu().flatten()
  visible &= torch.isfinite(mesh_depth[safe]) & ((camera[:,2]-mesh_depth[safe]).abs() <= .03+.01*camera[:,2])
  # Boundary exclusion uses only source predicted depth, independent of held-out reference depths.
  gx=torch.zeros_like(depth);gy=torch.zeros_like(depth);gx[:,1:]=(depth[:,1:]-depth[:,:-1]).abs();gy[1:]=(depth[1:]-depth[:-1]).abs();smooth=torch.maximum(gx,gy)<.03*depth;visible&=smooth.flatten()[safe]
  fit=visible&~heldout&~admission;select=visible&admission;test=visible&heldout
  if int(fit.sum())<100 or min(int(test.sum()),int(select.sum()))<20:raise ValueError(f'{frame}: insufficient source calibration evidence')
  pred=depth.flatten()[safe];reference=camera[:,2];cal=fit_constrained_affine_depth(pred[fit],reference[fit]);row=dict(frame_index=frame,**asdict(cal))
  def metrics(scale,offset,mask):
   error=(scale*pred[mask]+offset-reference[mask]).abs()/reference[mask]
   return dict(median=float(error.median()),p90=float(torch.quantile(error,.9)),mean=float(error.mean()))
  raw_select=metrics(1.,0.,select);affine_select=metrics(cal.scale,cal.offset,select)
  def quality(m):return m['median']<=.08 and m['p90']<=.25
  affine_ok=cal.accepted and quality(affine_select) and all(affine_select[k]<=raw_select[k] for k in ['median','p90','mean'])
  if affine_ok:chosen='affine'
  elif quality(raw_select):
   chosen='identity';row.update(scale=1.,offset=0.,median_absolute_residual=float((pred[fit]-reference[fit]).abs().median()),accepted=True,rejection_reason=None)
  else:
   chosen='rejected';row.update(accepted=False,rejection_reason='registered_admission_quality_or_fit_bounds')
  row['selected_candidate']=chosen
  calibrations.append(row);diagnostics.append(dict(frame=frame,fit_vertices=int(fit.sum()),admission_vertices=int(select.sum()),audit_vertices=int(test.sum()),admission_raw=raw_select,admission_affine=affine_select,raw_audit=metrics(1.,0.,test),selected_audit=metrics(row['scale'],row['offset'],test),affine_fit=asdict(cal),accepted=row['accepted'],selected_candidate=chosen));print(frame,diagnostics[-1],flush=True)
 inputs=[dict(role=role,path=str(path.resolve()),sha256=sha256_file(path)) for role,path in [('source_authority',authority),('registered_camera_table',camera_path),('moge3_scene_manifest',manifest_path),('unlabelled_registered_mesh',a.points)]]
 report=dict(schema='registered_source_moge_calibration.v2',source_only=True,benchmark_data_used=False,global_world_units_per_metre=1.,calibration=calibrations,accepted_view_count=sum(r['accepted'] for r in calibrations),geometry_receipt=dict(inputs=inputs,target_rgb_opened=False,benchmark_images_opened=False,benchmark_masks_opened=False,benchmark_labels_opened=False),validation=diagnostics,reference='Provided unlabelled reconstruction mesh. Mesh surface determines vertex visibility; MoGe depth is not a visibility criterion. Global vertex ID modulo5: 0 audit, 1 candidate admission, 2/3/4 fit. Identity fallback requires admission quality; affine additionally requires bounded fit and nonregressing median/p90/mean. Audit never selects candidates. Shared mesh/cameras are not independent sensor truth.',implementation_sha256=sha256_file(__file__),visibility_implementation_sha256=sha256_file('radio_gs/geometry/mesh_visibility.py'),fit_implementation_sha256=sha256_file('radio_gs/v4/geometry/depth_calibration.py'))
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
