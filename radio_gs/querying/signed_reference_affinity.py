"""Transport authorized signed reference evidence without an extent mask gate."""
import math
import torch
from torch.nn import functional as F


def patch_authority(positive, negative, size, seed=None):
    """Mixed scribble patches remain unknown; optional mask gives a pseudo control."""
    p,n=torch.as_tensor(positive),torch.as_tensor(negative)
    if p.dtype!=torch.bool or n.dtype!=torch.bool or p.ndim!=2 or p.shape!=n.shape or (p&n).any():
        raise ValueError('disjoint signed boolean masks required')
    pool=lambda x:F.adaptive_avg_pool2d(x.float()[None,None],size)[0,0]
    pp,nn=pool(p)>0,pool(n)>0
    if seed is None:
        labels=torch.zeros(size,dtype=torch.int64,device=p.device)
    else:
        s=torch.as_tensor(seed,device=p.device)
        if s.dtype!=torch.bool or s.shape!=p.shape:raise ValueError('seed shape/dtype differs')
        labels=torch.where(pool(s)>=.5,1,-1)
    labels[pp&~nn]=1;labels[nn&~pp]=-1;labels[pp&nn]=0
    return labels


def signed_reference_probability(reference, target, authority):
    """Nearest cosine margin, with one source-only leave-one-out scale.

    No class prior from sample counts, location restriction or target labels.
    Unknown reference patches supply neither positive nor negative evidence.
    The scale maps a median correctly signed LOO margin to probability 0.9.
    A missing class or no positive source margin yields a neutral response.
    """
    r,t=torch.as_tensor(reference).float(),torch.as_tensor(target).float()
    labels=torch.as_tensor(authority,device=r.device).flatten()
    if r.ndim!=2 or t.ndim!=2 or r.shape[1]!=t.shape[1] or len(labels)!=len(r):
        raise ValueError('feature/authority shapes differ')
    if not torch.isfinite(r).all() or not torch.isfinite(t).all() or not torch.isin(labels,labels.new_tensor([-1,0,1])).all():
        raise ValueError('invalid features/authority')
    r,t=F.normalize(r,dim=1),F.normalize(t,dim=1)
    pos,neg=r[labels==1],r[labels==-1]
    receipt=dict(positive_patches=len(pos),negative_patches=len(neg),unknown_patches=int((labels==0).sum()),
                 source_only_scale=True,reference_extent_gate=False)
    if min(len(pos),len(neg))<2:
        return t.new_full((len(t),),.5),dict(receipt,fallback='insufficient_signed_support')
    pp=pos@pos.T;nn=neg@neg.T;pn=pos@neg.T
    pp.fill_diagonal_(-torch.inf);nn.fill_diagonal_(-torch.inf)
    loo=torch.cat([pp.amax(1)-pn.amax(1),nn.amax(1)-pn.amax(0)])
    scale=loo.median()
    receipt.update(source_loo_accuracy=float((loo>0).float().mean()),median_signed_loo_margin=float(scale))
    if scale<=0:return t.new_full((len(t),),.5),dict(receipt,fallback='nonpositive_source_margin')
    margin=(t@pos.T).amax(1)-(t@neg.T).amax(1)
    probability=torch.sigmoid(margin*math.log(9)/scale)
    return probability,dict(receipt,fallback=None,temperature=float(scale/math.log(9)))
