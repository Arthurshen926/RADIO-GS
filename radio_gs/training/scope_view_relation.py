"""Source-only visible-scope relations; these do not prove semantic identity."""
import torch


def classify_visible_scope(target, projection, visible, *, min_pixels=20, containment=.75, disjoint=.25):
    if target.dtype!=torch.bool or projection.dtype!=torch.bool or visible.dtype!=torch.bool or target.shape!=projection.shape or target.shape!=visible.shape:
        raise ValueError('aligned Boolean source masks required')
    total=int(target.sum());t=target&visible;p=projection&visible
    nt=int(t.sum());np_=int(p.sum());hit=int((t&p).sum())
    metrics=dict(target_pixels=total,visible_target_pixels=nt,projection_pixels=np_,intersection_pixels=hit,target_visible_fraction=nt/max(total,1),teacher_contained=hit/max(nt,1),projection_contained=hit/max(np_,1))
    if total<min_pixels or nt<min_pixels or np_<min_pixels or nt/max(total,1)<.5:
        return 'weak_visibility',metrics
    a,b=metrics['teacher_contained'],metrics['projection_contained']
    if a>=containment and b>=containment:return 'support',metrics
    if a>=containment:return 'current_part',metrics
    if b>=containment:return 'current_whole',metrics
    if a<disjoint and b<disjoint:return 'disjoint',metrics
    return 'uncertain',metrics


def exclude_view(numerator, denominator, count, view_numerator, view_denominator, view_valid):
    """The only observation cannot validate itself, including subtraction noise."""
    remaining=(count-view_valid.to(count.dtype)).clamp_min(0)
    den=(denominator-view_denominator).clamp_min(0)
    num=(numerator-view_numerator).clamp_min(0).minimum(den)
    valid=(remaining>0)&(den>1e-6)
    return torch.where(valid,num/den.clamp_min(1e-8),0),valid
