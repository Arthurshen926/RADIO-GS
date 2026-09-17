"""Conservative source-only region association, not verified instance identity."""
import torch
from torch.nn import functional as F


def associate_source_groups(memberships, descriptors, views, *, minimum_iou=.25,
                            minimum_cosine=.8, minimum_size_ratio=.5, visibility=None,
                            minimum_shared_elements=1):
    if memberships.ndim!=2 or memberships.dtype!=torch.bool or descriptors.ndim!=2 or len(descriptors)!=len(memberships) or views.shape!=(len(memberships),):
        raise ValueError('group evidence axes differ')
    if not torch.isfinite(descriptors).all() or not memberships.any(1).all():
        raise ValueError('finite descriptors and nonempty groups required')
    size=memberships.sum(1);values=memberships.float();intersection=values@values.T
    if visibility is None:
        left_size=size[:,None].expand_as(intersection)
    else:
        if visibility.shape!=memberships.shape or visibility.dtype!=torch.bool or (memberships&~visibility).any():
            raise ValueError('group membership must have source visibility evidence')
        # Missing surfaces in another camera are unknown, not negative. Count
        # supports only in the co-visible domain for pairwise association.
        left_size=values@visibility.float().T
    right_size=left_size.T
    iou=intersection/(left_size+right_size-intersection).clamp_min(1)
    ratio=torch.minimum(left_size,right_size)/torch.maximum(left_size,right_size).clamp_min(1)
    unit=F.normalize(descriptors,dim=-1);similarity=unit@unit.T
    compatible=(intersection>=minimum_shared_elements)&(views[:,None]!=views[None])&(iou>=minimum_iou)&(similarity>=minimum_cosine)&(ratio>=minimum_size_ratio)
    # Mutual best partner separately for every other source view. All evidence
    # is source-side; benchmark query words and target cameras are absent.
    selected=torch.zeros_like(compatible)
    for view in views.unique():
        candidates=compatible&(views[None]==view)
        score=iou.masked_fill(~candidates,-1)
        rows=torch.where(candidates.any(1))[0]
        selected[rows,score[rows].argmax(1)]=True
    edges=torch.triu(selected&selected.T,diagonal=1).nonzero().tolist()
    edges.sort(key=lambda pair:(-float(iou[pair[0],pair[1]]),pair[0],pair[1]))
    components={i:{i} for i in range(len(views))};owner=list(range(len(views)))
    for left,right in edges:
        a,b=owner[left],owner[right]
        if a==b:continue
        ca,cb=components[a],components[b]
        # Do not merge different proposals from one view, and reject transitive
        # bridges unless every pair agrees. This intentionally favors precision.
        if set(views[list(ca)].tolist())&set(views[list(cb)].tolist()):continue
        if not all(bool(compatible[i,j]) for i in ca for j in cb):continue
        ca.update(cb)
        for i in cb:owner[i]=a
        del components[b]
    return [sorted(group) for group in components.values() if len(group)>1]
