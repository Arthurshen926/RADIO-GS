"""Query-derived proposal domains, separate from forced positive membership.

Candidate support licenses a search space; it does not label every member as
foreground. Unknown/outside support is not converted into negative evidence.
The shuffled and prior-only modes are diagnostic controls, not deployment rules.
"""
import numpy as np


def region_union(pointer,index,chosen,available):
    pointer=np.asarray(pointer);index=np.asarray(index);chosen=np.asarray(chosen);available=np.asarray(available)
    if (pointer.shape!=(len(available)+1,) or available.dtype!=bool or chosen.dtype!=bool
            or pointer[0]!=0 or pointer[-1]!=len(index) or (np.diff(pointer)<0).any()
            or (index<0).any() or (index>=len(chosen)).any()):
        raise ValueError('unaligned regional candidate support')
    members=np.repeat(np.arange(len(available)),np.diff(pointer))
    union=np.zeros(len(available),bool);union[members[chosen[index]]]=True
    return union & available


def observation_strata(count,index,weight,target,valid):
    """Match controls by current observed foreground/background/tied/unknown mass."""
    row,col=np.asarray(index);weight=np.asarray(weight,dtype=float);target=np.asarray(target)
    active=weight*np.asarray(valid)[row]
    mass=np.bincount(col,weights=active,minlength=count)
    positive=np.bincount(col,weights=active*target[row],minlength=count)
    strata=np.zeros(count,np.int64);known=mass>1e-8
    strata[known]=1
    strata[known & (positive>.5*mass)]=2
    strata[known & (positive<.5*mass)]=3
    return strata


def proposal_domain(candidate,initial,available,strata,*,mode='scope',seed=20261012):
    arrays=[np.asarray(v) for v in [candidate,initial,available]]
    candidate,initial,available=arrays;strata=np.asarray(strata)
    if (any(v.dtype!=bool or v.shape!=available.shape for v in arrays)
            or available.ndim!=1 or strata.shape!=available.shape or strata.dtype.kind not in 'iu'
            or (initial&~available).any() or (candidate&~available).any()):
        raise ValueError('invalid proposal domain')
    if mode not in {'scope','shuffled','prior_only'}:raise ValueError('unknown proposal policy')
    if (initial&~candidate).any():raise ValueError('candidate domain must retain the initial feasible set')
    allowed=initial.copy() if mode=='prior_only' else candidate.copy()
    if mode=='shuffled':
        rng=np.random.default_rng(seed)
        for group in np.unique(strata):
            ids=np.flatnonzero(available&~initial&(strata==group))
            allowed[ids]=candidate[rng.permutation(ids)]
    return allowed,dict(policy=mode,candidate_members=int(candidate.sum()),
        allowed_members=int(allowed.sum()),allowed_new_members=int((allowed&~initial).sum()),
        initial_members=int(initial.sum()),initial_members_locked=False,
        outside_support_is_negative=False,shuffle_seed=seed if mode=='shuffled' else None,
        observation_strata_counts={str(int(s)):int((allowed&~initial&(strata==s)).sum()) for s in np.unique(strata)})
