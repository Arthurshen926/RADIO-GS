"""Privileged source-query intervention. Never a deployed scene representation."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def select_parts(masks,instance):
    if masks.ndim!=3 or masks.dtype!=torch.bool or not len(masks):raise ValueError('invalid candidate masks')
    h,w=masks.shape[-2:];regions=masks.flatten(1)
    if not len(instance['scores']):
        target=torch.zeros(h*w,dtype=torch.bool)
    else:
        if not torch.isfinite(instance['masks']).all():raise ValueError('invalid source detection')
        target=(F.interpolate(instance['masks'].float().amax(0)[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0]>.5).flatten()
    union=torch.zeros_like(target);best=0.;chosen=[]
    for _ in range(len(regions)):
        candidates=regions|union[None]
        iou=(candidates&target).sum(1)/(candidates|target).sum(1).clamp_min(1)
        value,index=iou.max(0)
        if float(value)<=best:break
        best=float(value);chosen.append(int(index));union=candidates[index]
    # 0.5 is zero positive preference, NOT a reliable negative region label.
    probability=torch.full((len(masks),),.5)
    probability[chosen]=1.
    return probability,dict(teacher_present=bool(target.any()),selected=chosen,source_feasible_iou=best,
        acceptance_strength='binary privileged selection, not calibrated native confidence')
