"""Registered first-surface support derived only from the Gaussian carrier.

This transfers v10's geometric input function, not its observation-map solver.
It supplies one estimated support to the existing common member function and
does not turn a pixel into an exact positive 3D membership constraint.
"""
import numpy as np
from scipy.spatial import cKDTree
import torch

from .contracts import QueryPacket, Support


class SurfacePointAdapter:
    def __init__(self, field):
        self.field=field;self.xyz=field.xyz.detach().cpu().numpy()
        if len(self.xyz)<4:raise ValueError('surface registration requires at least four carrier points')
        distance,neighbors=cKDTree(self.xyz).query(self.xyz,k=min(12,len(self.xyz)),workers=4)
        offset=self.xyz[neighbors]-self.xyz[:,None,:]
        covariance=np.einsum('nki,nkj->nij',offset,offset)
        _,vectors=np.linalg.eigh(covariance)
        self.normal=vectors[:,:,0];self.radius=distance[:,3]

    def packet(self,pixel_xy,intrinsic,camera_to_world,*,scale=None):
        pose=np.asarray(camera_to_world,dtype=np.float64);k=np.asarray(intrinsic,dtype=np.float64)
        pixel=np.asarray(pixel_xy,dtype=np.float64)
        if pose.shape!=(4,4) or k.shape!=(3,3) or pixel.shape!=(2,) or not all(np.isfinite(x).all() for x in [pose,k,pixel]):
            raise ValueError('invalid registered surface query')
        if k[0,0]<=0 or k[1,1]<=0:raise ValueError('invalid registered surface camera')
        direction=pose[:3,:3]@np.linalg.solve(k,np.r_[pixel,1.])
        direction/=np.linalg.norm(direction)
        denominator=self.normal@direction
        travel=np.divide(np.sum(self.normal*(self.xyz-pose[:3,3]),axis=1),denominator,
            out=np.full(len(self.xyz),np.inf),where=np.abs(denominator)>1e-5)
        candidates=np.flatnonzero((travel>0)&np.isfinite(travel))
        hit=pose[:3,3]+travel[candidates,None]*direction
        inside=np.linalg.norm(hit-self.xyz[candidates],axis=1)<=self.radius[candidates]
        candidates=candidates[inside]
        if len(candidates):
            anchor=int(candidates[np.argmin(travel[candidates])]);rule='first_local_surface_intersection'
        else:
            camera=(self.xyz-pose[:3,3])@pose[:3,:3];ids=np.flatnonzero(camera[:,2]>.001)
            if not len(ids):raise ValueError('surface query has no carrier in front of camera')
            points=camera[ids];uv=points@k.T;uv=uv[:,:2]/uv[:,2,None]
            angular=np.linalg.norm(uv-pixel,axis=1);near=np.flatnonzero(angular<=2.)
            local=near[np.argmin(points[near,2])] if len(near) else int(np.argmin(angular))
            anchor=int(ids[local]);rule='frontmost_within_two_pixels_else_nearest_ray'
        support=Support(torch.tensor([anchor],device=self.field.xyz.device),torch.ones(1,device=self.field.xyz.device))
        return QueryPacket('point_2d',positive=support,scale=scale),dict(
            anchor=anchor,rule=rule,geometry_only=True,source_access=False,
            depth_or_rgb_input=False,hard_positive_constraint=False)
