"""Query-independent semantic attribution over fixed renderer contributions."""
import torch
from torch.nn import functional as F


def visibility_gate(pixel_ids, weights, pixels, mass_fraction=.7, retained_fraction=.5, floor=1e-4):
    """Per-ray mass prefix, capped by a per-view contribution quantile.

    VALA-inspired adaptation: the published view/ray notation is ambiguous;
    here coverage is explicitly per ray and the quantile explicitly per view.
    Gate writes only. The forward renderer is never altered.
    """
    order = weights.argsort(descending=True, stable=True)
    order = order[pixel_ids[order].argsort(stable=True)]
    pid, w = pixel_ids[order], weights[order]
    total = torch.zeros(pixels, device=w.device).scatter_add_(0, pid, w)
    cumulative = w.double().cumsum(0)
    start = torch.ones(len(w), dtype=torch.bool, device=w.device)
    start[1:] = pid[1:] != pid[:-1]
    offsets = torch.zeros(pixels, dtype=torch.float64, device=w.device)
    offsets[pid[start]] = cumulative[start] - w[start]
    previous = cumulative - offsets[pid] - w
    threshold = torch.quantile(w[w > 0], 1-retained_fraction)
    selected = (previous < mass_fraction*total[pid]) & (w >= threshold) & (w >= floor)
    keep = torch.zeros_like(selected)
    keep[order] = selected
    return keep


def geometric_median(records, initial, iterations=5):
    """Spherical normalized Weiszfeld over per-view unit observations.

    Uses chord distance sqrt(2-2cos), not an exact VALA streaming reproduction.
    """
    center = initial.float()
    for _ in range(iterations):
        total = torch.zeros_like(center)
        for ids, features, mass in records:
            distance = (2-2*(center[ids]*features).sum(-1)).clamp_min(1e-4).sqrt()
            weight = mass/distance
            total.index_add_(0, ids, weight[:, None]*features)
        center = F.normalize(total, dim=-1)
    return center


def two_view_prototypes(records, initial, iterations=3):
    """Keep two view modes; a singleton secondary mode falls back to primary.

    No query or class identity participates in the construction. Distinct
    modes are observations, not a claim to have identified object/part groups.
    """
    first=initial.float();second=first.clone();worst=torch.ones(len(first),device=first.device)
    for ids,features,mass in records:
        cosine=(first[ids]*features).sum(-1);take=cosine<worst[ids]
        second[ids[take]]=features[take];worst[ids[take]]=cosine[take]
    for _ in range(iterations):
        sums=[torch.zeros_like(first),torch.zeros_like(first)];counts=torch.zeros(len(first),2,device=first.device)
        for ids,features,mass in records:
            assignment=((features*second[ids]).sum(-1)>(features*first[ids]).sum(-1)).long()
            for k in range(2):
                chosen=assignment==k;row=ids[chosen];sums[k].index_add_(0,row,features[chosen]*mass[chosen,None]);counts[:,k].index_add_(0,row,torch.ones(len(row),device=first.device))
        first=F.normalize(sums[0],dim=-1);second=F.normalize(sums[1],dim=-1)
    first=torch.where((counts[:,0]>0)[:,None],first,initial)
    second=torch.where((counts[:,1]>=2)[:,None],second,first)
    return torch.stack([first,second],1)
