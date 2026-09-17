"""Robust, anchored joint depth-scale fit with fixed cameras/correspondences."""
from __future__ import annotations
import torch


def fit_joint_depth_scales(view_i, view_j, point_i, point_j, translation_delta, *, num_views,
                           prior_weight=0.01, maximum_deviation=0.1, iterations=8):
    """Fit s_i*p_i - s_j*p_j + (t_i-t_j)=0, with a unit-scale anchor.

    Points are rotated into world axes but still relative to their cameras.
    Correspondences must represent the same surface, not arbitrary occlusion
    pairs. Per-edge balance prevents a large image dominating the fit. Bounds
    preserve the independent metric calibration and fix scale-collapse risk.
    """
    if num_views <= 0 or not 0 < maximum_deviation < 1 or prior_weight <= 0 or iterations <= 0:
        raise ValueError('invalid scale optimization policy')
    i, j = torch.as_tensor(view_i).long(), torch.as_tensor(view_j).long()
    p, q, t = [torch.as_tensor(x, dtype=torch.float64, device=i.device) for x in (point_i, point_j, translation_delta)]
    if i.ndim != 1 or j.shape != i.shape or any(x.shape != (i.numel(), 3) for x in (p,q,t)):
        raise ValueError('scale correspondences must be aligned [M]/[M,3]')
    if not all(bool(torch.isfinite(x).all()) for x in (p,q,t)):
        raise ValueError('nonfinite correspondence')
    if i.numel() and (min(int(i.min()),int(j.min())) < 0 or max(int(i.max()),int(j.max())) >= num_views or bool((i==j).any())):
        raise ValueError('invalid view indices')
    scale = torch.ones(num_views, dtype=torch.float64, device=i.device)
    if not i.numel():
        return scale.float()
    _, inv, counts = torch.unique(i*num_views+j, return_inverse=True, return_counts=True)
    balance = 1/counts[inv].double()
    # Normalize metric residuals so the regularizer is invariant to world units.
    unit = ((p.norm(dim=1)+q.norm(dim=1))*.5).clamp_min(1e-8)
    p,q,t = [x/unit[:,None] for x in (p,q,t)]
    for _ in range(iterations):
        residual = (scale[i,None]*p-scale[j,None]*q+t).norm(dim=1)
        robust = torch.minimum(torch.ones_like(residual), .02/residual.clamp_min(1e-12))
        weight = balance*robust
        matrix = torch.eye(num_views, dtype=p.dtype, device=p.device)*prior_weight
        target = torch.ones_like(scale)*prior_weight
        matrix.index_put_((i,i),weight*p.square().sum(1),accumulate=True)
        matrix.index_put_((j,j),weight*q.square().sum(1),accumulate=True)
        cross = -weight*(p*q).sum(1)
        matrix.index_put_((i,j),cross,accumulate=True)
        matrix.index_put_((j,i),cross,accumulate=True)
        target.index_add_(0,i,-weight*(p*t).sum(1))
        target.index_add_(0,j,weight*(q*t).sum(1))
        scale = torch.linalg.solve(matrix,target).clamp(1-maximum_deviation,1+maximum_deviation)
    return scale.float()


def recover_unanchored_view_scales(views, *, maximum_neighbors=8, minimum_pairs=3,
                                  maximum_log_dispersion=.08):
    """Recover rejected local calibrations using only trusted mapping views.

    Per-neighbor median depth ratios supply independent scale proposals. No
    heldout depth is permitted in ``views``. A view is admitted only if both
    within-view ratios and neighbor proposals agree; unsuccessful views are
    excluded instead of injecting a wrong global-scale fallback.
    """
    from torch.nn import functional as F
    trusted=[v for v in views if v['local_calibration_accepted']]
    output=[];report=[]
    for target in views:
        if target['local_calibration_accepted']:
            output.append(target);continue
        neighbors=sorted(trusted,key=lambda v:float((v['pose'][:3,3]-target['pose'][:3,3]).norm()))[:maximum_neighbors]
        estimates=[];dispersions=[]
        for src in neighbors:
            mask=src['valid'].clone();mask[1::2]=False;mask[:,1::2]=False
            y,x=torch.where(mask)
            p=src['rays'][y,x]*src['depth'][y,x,None]
            world=p@src['pose'][:3,:3].T+src['pose'][:3,3]
            camera=(world-target['pose'][:3,3])@target['pose'][:3,:3]
            uv=camera@target['k'].T
            px=(uv[:,0]/uv[:,2].clamp_min(1e-8)).floor().long();py=(uv[:,1]/uv[:,2].clamp_min(1e-8)).floor().long()
            h,w=target['depth'].shape
            inside=(camera[:,2]>0)&(px>=0)&(px<w)&(py>=0)&(py<h)
            px,py=px.clamp(0,w-1),py.clamp(0,h-1)
            d=target['depth'][py,px]
            n1=src['normals'][y,x]@src['pose'][:3,:3].T
            n2=target['normals'][py,px]@target['pose'][:3,:3].T
            valid=inside&target['valid'][py,px]&torch.isfinite(d)&(d>0)&(F.cosine_similarity(n1,n2).abs()>.8)
            if int(valid.sum())<128:continue
            ratio=(camera[valid,2]/d[valid]).log()
            median=ratio.median();mad=(ratio-median).abs().median()
            if float(mad)<=maximum_log_dispersion:
                estimates.append(median);dispersions.append(float(mad))
        accepted=False;value=1.;spread=None
        if len(estimates)>=minimum_pairs:
            e=torch.stack(estimates);centre=e.median();spread=float((e-centre).abs().median())
            value=float(centre.exp())
            accepted=spread<maximum_log_dispersion/2 and .25<=value<=4.
        report.append({'frame':target['frame'],'accepted':accepted,'scale':value,
                       'trusted_neighbor_count':len(estimates),'neighbor_log_dispersion':spread,
                       'maximum_within_neighbor_log_dispersion':max(dispersions) if dispersions else None})
        if accepted:
            corrected={**target,'depth':target['depth']*value,'recovered_scale':value}
            output.append(corrected)
    return output,report
