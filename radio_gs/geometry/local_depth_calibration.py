"""Source-track local multiplicative depth correction, experimental and bounded.

This uses geometric sparse references, not semantic features. Fit and validation
point identities must be disjoint at the caller, across all views.
"""
import math
import torch


def local_depth_scales(train_xy, train_depth, reference_depth, query_xy, query_depth,
                       *, fallback_scale, neighbors=16, minimum_neighbors=4,
                       spatial_radius=.1, log_depth_radius=.2, maximum_ratio=1.2):
    xy=torch.as_tensor(train_xy,dtype=torch.float64)
    p=torch.as_tensor(train_depth,dtype=torch.float64).reshape(-1)
    r=torch.as_tensor(reference_depth,dtype=torch.float64).reshape(-1)
    qxy=torch.as_tensor(query_xy,dtype=torch.float64)
    q=torch.as_tensor(query_depth,dtype=torch.float64).reshape(-1)
    if xy.shape!=(len(p),2) or r.shape!=p.shape or qxy.shape!=(len(q),2):
        raise ValueError('local calibration sample axes differ')
    if not (math.isfinite(fallback_scale) and fallback_scale>0 and neighbors>=minimum_neighbors>=1
            and spatial_radius>0 and log_depth_radius>0 and maximum_ratio>=1):
        raise ValueError('invalid local calibration policy')
    valid=torch.isfinite(xy).all(1)&torch.isfinite(p)&torch.isfinite(r)&(p>0)&(r>0)
    xy,p,r=xy[valid],p[valid],r[valid]
    out=torch.full_like(q,float(fallback_scale));supported=torch.zeros_like(q,dtype=torch.bool)
    if len(p)<minimum_neighbors:return out,supported
    ratio=(r/p).log()
    # Chunking avoids Q x all-training-points storage for dense map materialization.
    for start in range(0,len(q),2048):
        stop=min(start+2048,len(q));qq=q[start:stop];uv=qxy[start:stop]
        distance=torch.cdist(uv,xy)/spatial_radius
        depth_distance=(qq.clamp_min(1e-12).log()[:,None]-p.log()[None]).abs()/log_depth_radius
        eligible=(distance<=1)&(depth_distance<=1)&torch.isfinite(uv).all(1)[:,None]&torch.isfinite(qq)[:,None]&(qq>0)[:,None]
        cost=(distance.square()+depth_distance.square()).masked_fill(~eligible,float('inf'))
        cost,index=cost.topk(min(neighbors,len(p)),dim=1,largest=False)
        weights=torch.exp(-.5*cost);values=ratio[index]
        values,order=values.sort(1);weights=weights.gather(1,order)
        cumulative=weights.cumsum(1);middle=(cumulative>=.5*weights.sum(1,keepdim=True)).long().argmax(1)
        estimate=values.gather(1,middle[:,None]).squeeze(1).exp()
        known=(weights>0).sum(1)>=minimum_neighbors
        estimate=estimate.clamp(fallback_scale/maximum_ratio,fallback_scale*maximum_ratio)
        out[start:stop]=torch.where(known,estimate,out[start:stop]);supported[start:stop]=known
    return out,supported


def warp_normals_for_image_scale(points, normals, scales, intrinsic):
    """Transform normals with inverse-transpose Jacobian of X'=s(u,v) X.

    Pixel derivatives use the materialized positive scale map; a constant scale
    preserves normals. Input points/normals are in source-camera coordinates.
    """
    if points.shape!=normals.shape or points.shape!=(*scales.shape,3) or scales.ndim!=2 or min(scales.shape)<2:
        raise ValueError('normal warp requires aligned HxW geometry')
    if not bool(torch.isfinite(scales).all() & (scales>0).all()):
        raise ValueError('positive finite scale map required')
    dv,du=torch.gradient(scales.log())
    z=points[...,2].clamp_min(1e-8)
    gx=du*intrinsic[0,0]/z;gy=dv*intrinsic[1,1]/z
    gz=-(gx*points[...,0]+gy*points[...,1])/z
    gradient=torch.stack([gx,gy,gz],-1)
    result=normals-gradient*(points*normals).sum(-1,keepdim=True)
    return torch.nn.functional.normalize(result,dim=-1)
