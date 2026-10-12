"""Finite membership decisions preserving observation and selected-set functions.

The same binary set must explain normalized full-scene observation mass A_obs S
and its counterfactual selected-only rendering R_set(S). Neither is a target-view
answer or a hard lock on inferred members. This is an experimental objective.
"""
import numpy as np
from numba import njit
from scipy import sparse
from .extent_decision_linked import LinkedExtentDecision, response, probability_at_cutoff, solve
from .extent_decision import binary_cross_entropy


@njit(cache=True)
def block_response(head,following,alpha,index,row,group,positions,start,stop,remove):
    """Merge all added hits, or omit all removed hits, in canonical depth order."""
    t=np.float32(1.);k=head[row];q=stop if remove else start
    while k>=0 or q<stop:
        if q<stop and (k<0 or positions[q]<k):hit=positions[q];q+=1
        else:
            hit=k;k=following[k]
            if remove and index[hit]==group:continue
        a=min(alpha[hit],np.float32(.999))
        if a<1/255:continue
        next_t=np.float32(t*np.float32(np.float32(1)-a))
        if next_t<=1e-4:break
        t=next_t
    return np.float32(1)-t


@njit(cache=True)
def coupled_decisions(ptr,idx,alpha,colptr,colrows,colposition,rowview,initial,
                      target,known,coefficient,weight,cost,max_passes,decision_cutoff,
                      optr,orows,oval,observation_weight,block_pointer=None,proposal_domain=None):
    selected=initial.copy();views=len(weight);rows=len(rowview)
    head=np.full(rows,-1,np.int32);following=np.full(len(idx),-1,np.int32)
    prob=np.zeros((2,rows));bce=np.zeros((2,views));mass=np.zeros((2,views))
    overlap=np.zeros((2,views));truth=np.zeros(views);observed=np.zeros(rows)
    mix=np.array([1-observation_weight,observation_weight])
    for g in range(len(selected)):
        if selected[g]:
            for k in range(optr[g],optr[g+1]):observed[orows[k]]+=oval[k]
    for r in range(rows):
        if not known[r]:continue
        for k in range(ptr[r+1]-1,ptr[r]-1,-1):
            if selected[idx[k]]:following[k]=head[r];head[r]=k
        v=rowview[r];truth[v]+=target[r]
        prob[0,r]=probability_at_cutoff(response(head,following,alpha,r),decision_cutoff)
        prob[1,r]=min(max(observed[r],1e-6),1-1e-6)
        for channel in range(2):
            p=prob[channel,r];bce[channel,v]+=coefficient[r]*binary_cross_entropy(p,target[r])
            mass[channel,v]+=p;overlap[channel,v]+=p*target[r]
    trace=np.zeros(max_passes+1);flips=0;changed=0;evaluated=0
    dm=np.zeros((2,views));dt=np.zeros((2,views));db=np.zeros((2,views))
    scratch=np.zeros(int(np.max(colptr[1:]-colptr[:-1])) if len(initial) else 0)
    for step in range(max_passes+1):
        energy=.1*np.dot(cost,selected.astype(np.float64))
        for channel in range(2):
            for v in range(views):
                energy+=mix[channel]*weight[v]*(bce[channel,v]+1-(2*overlap[channel,v]+1)/(mass[channel,v]+truth[v]+1))
        trace[step]=energy
        if step==max_passes or (step>0 and changed==0):break
        changed=0
        for g in range(len(selected)):
            if proposal_domain is not None and not proposal_domain[g]:continue
            if colptr[g]==colptr[g+1] and optr[g]==optr[g+1] and cost[g]==0:continue
            dm[:]=0.;dt[:]=0.;db[:]=0.;sign=-1. if selected[g] else 1.
            delta=.1*sign*cost[g]
            for k in range(colptr[g],colptr[g+1]):
                r=colrows[k]
                if not known[r]:continue
                v=rowview[r]
                if block_pointer is None:a=response(head,following,alpha,r,colposition[k],selected[g])
                else:a=block_response(head,following,alpha,idx,r,g,colposition,block_pointer[k],block_pointer[k+1],selected[g])
                p=probability_at_cutoff(a,decision_cutoff)
                scratch[k-colptr[g]]=p;diff=p-prob[0,r]
                dm[0,v]+=diff;dt[0,v]+=diff*target[r]
                db[0,v]+=coefficient[r]*(binary_cross_entropy(p,target[r])-binary_cross_entropy(prob[0,r],target[r]))
            for k in range(optr[g],optr[g+1]):
                r=orows[k]
                if not known[r]:continue
                v=rowview[r];p=min(max(observed[r]+sign*oval[k],1e-6),1-1e-6);diff=p-prob[1,r]
                dm[1,v]+=diff;dt[1,v]+=diff*target[r]
                db[1,v]+=coefficient[r]*(binary_cross_entropy(p,target[r])-binary_cross_entropy(prob[1,r],target[r]))
            for channel in range(2):
                for v in range(views):
                    delta+=mix[channel]*weight[v]*(db[channel,v]+(2*overlap[channel,v]+1)/(mass[channel,v]+truth[v]+1)
                        -(2*(overlap[channel,v]+dt[channel,v])+1)/(mass[channel,v]+dm[channel,v]+truth[v]+1))
            evaluated+=1
            if delta < -1e-12:
                selected[g]=not selected[g];changed+=1;flips+=1;mass+=dm;overlap+=dt;bce+=db
                for k in range(optr[g],optr[g+1]):
                    r=orows[k];observed[r]+=sign*oval[k]
                    if known[r]:prob[1,r]=min(max(observed[r],1e-6),1-1e-6)
                for k in range(colptr[g],colptr[g+1]):
                    r=colrows[k]
                    if not known[r]:continue
                    prob[0,r]=scratch[k-colptr[g]]
                    start=k if block_pointer is None else block_pointer[k]
                    stop=k+1 if block_pointer is None else block_pointer[k+1]
                    for q in range(start,stop):
                        position=colposition[q];previous=-1;current=head[r]
                        while current>=0 and current<position:previous=current;current=following[current]
                        if selected[g]:
                            following[position]=current
                            if previous<0:head[r]=position
                            else:following[previous]=position
                        else:
                            assert current==position
                            if previous<0:head[r]=following[position]
                            else:following[previous]=following[position]
                            following[position]=-1
    return selected,trace[:step+1],flips,evaluated,changed==0


class CoupledExtentDecision(LinkedExtentDecision):
    def __init__(self,factors,preparation,count):
        super().__init__(factors,count)
        if len(preparation.views)!=len(self.sizes):raise ValueError('unaligned observation views')
        views=[]
        for size,v in zip(self.sizes,preparation.views):
            if len(v.valid)!=size or v.count!=count:raise ValueError('unaligned observation operator')
            views.append(sparse.coo_matrix((v.weight.double().cpu().numpy(),v.index.cpu().numpy()),shape=(size,count)))
        op=sparse.vstack(views).tocsc();op.sum_duplicates();op.sort_indices()
        self.optr=op.indptr;self.orows=op.indices;self.oval=op.data

    def solve(self,initial,target,positive,negative,confidence,cost,*,max_passes=4,
              decision_cutoff=.5,observation_weight=.5):
        if not np.isfinite(observation_weight) or not 0<=observation_weight<=1:
            raise ValueError('observation weight must be in [0,1]')
        if observation_weight==0:
            return super().solve(initial,target,positive,negative,confidence,cost,
                max_passes=max_passes,decision_cutoff=decision_cutoff)
        def kernel(*args):
            return coupled_decisions(*args,self.optr,self.orows,self.oval,float(observation_weight))
        selected,report=solve(self,initial,target,positive,negative,confidence,cost,
            max_passes=max_passes,decision_cutoff=decision_cutoff,_kernel=kernel)
        report.update(observation_weight=float(observation_weight),same_members_for_both_operators=True)
        return selected,report
