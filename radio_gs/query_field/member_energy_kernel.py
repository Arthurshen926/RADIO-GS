"""Compiled execution of the reference's identical bounded decision neighborhood."""
import numpy as np
from numba import njit


@njit
def loss(x,r,p,n,cutoff):
    z=min(x/cutoff,1.)
    return p[r]*(1-z)+n[r]*z


@njit
def pair_delta(g,h,ptr,idx,val,coverage,p,n,cutoff,apply):
    i,end=ptr[g],ptr[g+1];j,stop=ptr[h],ptr[h+1];delta=0.
    while i<end or j<stop:
        r=min(idx[i] if i<end else len(coverage),idx[j] if j<stop else len(coverage))
        change=0.
        if i<end and idx[i]==r:change-=val[i];i+=1
        if j<stop and idx[j]==r:change+=val[j];j+=1
        new=max(coverage[r]+change,0.)
        delta+=loss(new,r,p,n,cutoff)-loss(coverage[r],r,p,n,cutoff)
        if apply:coverage[r]=new
    return delta


@njit
def run(ptr,idx,val,row_ptr,row_ids,prior,initial,selected,touched,p,n,cost,disagreement,coverage,
        cutoff,support_weight,prior_weight,max_passes,budget):
    trace=np.zeros(max_passes+1);flips=0;exchanges=0;pairs=0;changed=0
    for step in range(max_passes+1):
        e=0.
        for r in range(len(coverage)):e+=loss(coverage[r],r,p,n,cutoff)
        for g in range(len(selected)):
            if selected[g]:e+=support_weight*cost[g]
            if selected[g]!=initial[g]:e+=prior_weight*disagreement[g]
        trace[step]=e
        if step==max_passes or (step>0 and changed==0):break
        changed=0
        for g in range(len(selected)):
            if not touched[g]:continue
            sign=-1. if selected[g] else 1.;delta=0.
            for k in range(ptr[g],ptr[g+1]):
                r=idx[k];new=max(coverage[r]+sign*val[k],0.)
                delta+=loss(new,r,p,n,cutoff)-loss(coverage[r],r,p,n,cutoff)
            delta+=support_weight*sign*cost[g]
            delta+=prior_weight*disagreement[g]*(1 if selected[g]==initial[g] else -1)
            if delta < -1e-12:
                selected[g]=not selected[g];changed+=1;flips+=1
                for k in range(ptr[g],ptr[g+1]):
                    r=idx[k];coverage[r]=max(coverage[r]+sign*val[k],0.)
        if budget:
            snapshot=selected.copy()
            for g in range(len(selected)):
                if not snapshot[g] or not touched[g]:continue
                candidates=np.full(budget,-1,np.int64);size=0
                for k in range(ptr[g],ptr[g+1]):
                    r=idx[k]
                    if p[r]<=0:continue
                    for j in range(row_ptr[r],row_ptr[r+1]):
                        h=row_ids[j]
                        if selected[h] or not touched[h]:continue
                        if size==budget:
                            last=candidates[size-1]
                            if cost[h]>cost[last] or (cost[h]==cost[last] and h>=last):break
                        duplicate=False
                        for t in range(size):
                            if candidates[t]==h:duplicate=True;break
                        if duplicate:continue
                        pos=size
                        for t in range(size):
                            old=candidates[t]
                            if cost[h]<cost[old] or (cost[h]==cost[old] and h<old):pos=t;break
                        for t in range(min(size,budget-1),pos,-1):candidates[t]=candidates[t-1]
                        candidates[pos]=h;size=min(size+1,budget)
                best=-1;best_delta=-1e-12
                for t in range(size):
                    h=candidates[t];pairs+=1
                    delta=pair_delta(g,h,ptr,idx,val,coverage,p,n,cutoff,False)
                    delta+=support_weight*(cost[h]-cost[g])
                    delta+=prior_weight*(disagreement[g]*(1 if initial[g] else -1)+disagreement[h]*(-1 if initial[h] else 1))
                    if delta<best_delta:best=h;best_delta=delta
                if best>=0:
                    pair_delta(g,best,ptr,idx,val,coverage,p,n,cutoff,True)
                    selected[g]=False;selected[best]=True;changed+=1;exchanges+=1;flips+=2
    return selected,trace[:step+1],flips,exchanges,pairs,step,changed


def execute(matrix,row_matrix,prior,initial,selected,touched,p,n,cost,disagreement,coverage,
            cutoff,support_weight,prior_weight,max_passes,budget):
    # Sorting each factor once is equivalent to forming the full union and sorting
    # it for each member. Dynamic selected filtering stays inside the kernel.
    ids=row_matrix.indices.copy()
    for r in range(row_matrix.shape[0]):
        a,b=row_matrix.indptr[r:r+2];v=ids[a:b].copy()
        ids[a:b]=v[np.lexsort((v,cost[v]))]
    return run(matrix.indptr,matrix.indices,matrix.data,row_matrix.indptr,ids,prior,initial,
               selected,touched,p,n,cost,disagreement,coverage,cutoff,support_weight,
               prior_weight,max_passes,budget)
