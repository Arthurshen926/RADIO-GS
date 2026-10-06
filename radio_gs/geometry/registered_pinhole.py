"""Registered RGB camera conversion, independent of COLMAP tracks or labels."""
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class RegisteredPinholeCamera:
    frame_index: int
    width: int
    height: int
    intrinsic: torch.Tensor
    world_to_camera: torch.Tensor
    camera_to_world: torch.Tensor


def blender_camera_to_opencv(frame, defaults):
    c2w=torch.as_tensor(frame['transform_matrix'],dtype=torch.float64).clone()
    if c2w.shape!=(4,4) or not torch.isfinite(c2w).all():
        raise ValueError('Finite 4x4 camera-to-world matrix required')
    if not torch.allclose(c2w[3],torch.tensor([0.,0.,0.,1.],dtype=c2w.dtype),atol=1e-6):
        raise ValueError('Invalid homogeneous camera pose')
    rotation=c2w[:3,:3]
    if not torch.allclose(rotation.T@rotation,torch.eye(3,dtype=c2w.dtype),atol=2e-3,rtol=0) or abs(float(torch.linalg.det(rotation))-1)>2e-3:
        raise ValueError('Registered camera rotation must be proper and orthogonal')
    # Invert the exact sign convention used by OpenGaussian scannet2blender.py.
    c2w[:3,1:3]*=-1
    values={key:frame.get(key,defaults.get(key)) for key in ['fl_x','fl_y','cx','cy','w','h']}
    if any(value is None for value in values.values()):raise ValueError('Missing pinhole intrinsic/dimensions')
    fx,fy,cx,cy=[float(values[key]) for key in ['fl_x','fl_y','cx','cy']]
    k=torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=torch.float64)
    width,height=int(values['w']),int(values['h'])
    if not torch.isfinite(k).all() or min(fx,fy,width,height)<=0:raise ValueError('Invalid pinhole camera')
    return dict(camera_to_world=c2w,world_to_camera=torch.linalg.inv(c2w),intrinsic=k,width=width,height=height)


def load_registered_camera_table(path):
    """Load an explicitly OpenCV table; never synthesize COLMAP observations."""
    import json
    from pathlib import Path
    data=json.loads(Path(path).read_text())
    if data.get('schema')!='registered_pinhole_source_authority.v1' or not data.get('camera_convention','').startswith('OpenCV:'):
        raise ValueError('Explicit registered OpenCV camera table required')
    cameras={}
    for row in data['records']:
        frame=int(row['frame_index'])
        if frame in cameras:raise ValueError('Duplicate registered camera frame')
        cv=torch.as_tensor(row['camera_to_world'],dtype=torch.float64)
        if cv.shape!=(4,4):raise ValueError('Invalid registered pose dimensions')
        gl=cv.clone();gl[:3,1:3]*=-1
        k=torch.as_tensor(row['intrinsic'],dtype=torch.float64)
        if k.shape!=(3,3) or not torch.allclose(k[2],torch.tensor([0.,0.,1.],dtype=k.dtype)) or k[0,1]!=0 or k[1,0]!=0:
            raise ValueError('Invalid registered pinhole intrinsic matrix')
        result=blender_camera_to_opencv(dict(transform_matrix=gl,fl_x=k[0,0],fl_y=k[1,1],cx=k[0,2],cy=k[1,2],w=row['width'],h=row['height']),{})
        recorded=torch.as_tensor(row['world_to_camera'],dtype=torch.float64)
        if recorded.shape!=(4,4) or not torch.allclose(recorded,result['world_to_camera'],atol=1e-6,rtol=1e-6):raise ValueError('Registered pose inverse mismatch')
        cameras[frame]=RegisteredPinholeCamera(frame,result['width'],result['height'],result['intrinsic'].float(),result['world_to_camera'].float(),result['camera_to_world'].float())
    if set(cameras)!=set(data['source_frame_indices']):raise ValueError('Registered camera authority frame mismatch')
    return cameras
