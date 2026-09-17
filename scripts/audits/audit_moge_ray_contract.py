"""Compare native MoGe rays/normals against sealed COLMAP pinhole cameras."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
torch.set_num_threads(4);out=[]
for scene in ('figurines','ramen','teatime','waldo_kitchen'):
 report=Path('/mnt/pool/sqy/results/RADIO-GS/output/v4_fusion_contract_repair_20260914')/(scene+'.json')
 prior=MogeGeometryPrior(str(report),sha256_file(report));frame=min(prior.records);row=prior.records[frame]
 assert sha256_file(row['prediction'])==row['prediction_sha256']
 payload=torch.load(row['prediction'],map_location='cpu',weights_only=False)
 points=payload['point_map'].float();normals=payload['normals'].float();valid=payload['validity'].bool()
 h,w=valid.shape;cam=prior.cameras[frame];k=cam.intrinsic.float().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
 y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
 uv=points@k.T;uv=uv[...,:2]/uv[...,2:].clamp_min(1e-8)
 residual=(uv-torch.stack([x+.5,y+.5],dim=-1)).norm(dim=-1)
 ok=valid&torch.isfinite(points).all(-1)&(points[...,2]>0)
 dx=points[1:-1,2:]-points[1:-1,:-2];dy=points[2:,1:-1]-points[:-2,1:-1]
 derived=F.normalize(torch.cross(dx,dy,dim=-1),dim=-1)
 interior=ok[1:-1,1:-1]&ok[1:-1,2:]&ok[1:-1,:-2]&ok[2:,1:-1]&ok[:-2,1:-1]
 interior&=(dx.norm(dim=-1)<.03*points[1:-1,1:-1,2])&(dy.norm(dim=-1)<.03*points[1:-1,1:-1,2])
 cosine=F.cosine_similarity(derived,normals[1:-1,1:-1],dim=-1).abs().clamp(0,1)
 angle=cosine[interior].acos()*180/torch.pi
 out.append({'scene':scene,'frame':frame,'native_shape':[h,w],
             'median_ray_reprojection_error_pixels':float(residual[ok].median()),
             'p95_ray_reprojection_error_pixels':float(residual[ok].quantile(.95)),
             'normal_vs_pointmap_median_unsigned_angle_degrees':float(angle.median()),
             'input_pointmap_dtype':str(payload['point_map'].dtype),
             'prediction_sha256':row['prediction_sha256']})
Path('paper/artifacts/moge_ray_normal_audit_20260914.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
