"""Validate an explicitly materialized source-only geometry preprocessing result."""
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def load_source_geometry_override(record, *, frame, prediction_sha256, camera):
    path=record['path']
    if sha256_file(path)!=record['sha256']:raise ValueError('source geometry override identity differs')
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('schema')!='source_local_scale_geometry.v1' or data['frame']!=frame or data['prediction_sha256']!=prediction_sha256:
        raise ValueError('source geometry override provenance differs')
    if data['benchmark_data_used'] or not data['source_only']:raise ValueError('source-only geometry required')
    depth,normal,valid=data['depth'],data['normals'],data['valid']
    if depth.ndim!=2 or normal.shape!=(*depth.shape,3) or valid.shape!=depth.shape or valid.dtype!=torch.bool:
        raise ValueError('source geometry override axes differ')
    h,w=depth.shape;k=camera.intrinsic.clone();k[0]*=w/camera.width;k[1]*=h/camera.height
    if not torch.allclose(k,data['intrinsic'],atol=1e-5,rtol=1e-6) or not torch.equal(camera.world_to_camera,data['world_to_camera']):
        raise ValueError('source geometry override camera differs')
    if not torch.isfinite(depth).all() or not torch.isfinite(normal).all() or not (depth[valid]>0).all() or not (normal[valid].norm(dim=-1)>1e-6).all():
        raise ValueError('invalid materialized geometry')
    return data
