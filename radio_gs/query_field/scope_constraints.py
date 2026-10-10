"""Field-owned union constraints, composed BEFORE lifting to Gaussian rows.

Atoms are equivalence classes of partial relations, with no pixel coordinates,
source rasters, cameras or query answers. This is evidence, not a member solver.
"""
import torch
from torch import nn


class ConstraintView(nn.Module):
    def __init__(self,state,count):
        super().__init__();self.count=count
        keys={'pointer','index','value','positive','negative','interior','regions','candidates'}
        if set(state)!=keys:raise ValueError('incomplete scope constraints')
        for k,v in state.items():self.register_buffer(k,v.detach())
        atoms=len(self.positive)
        if self.positive.shape!=(atoms,len(self.regions)) or self.negative.shape!=self.positive.shape or self.interior.shape!=(atoms,len(self.candidates)):
            raise ValueError('unaligned relation incidence')
        if any(x.dtype!=torch.bool for x in [self.positive,self.negative,self.interior]):raise ValueError('relations must be boolean')
        if bool((self.positive&self.negative).any()):raise ValueError('contradictory partial labels')
        if any(x.dtype!=torch.long or x.ndim!=1 for x in [self.pointer,self.index,self.regions,self.candidates]):raise ValueError('invalid sparse coordinates')
        if self.pointer.shape!=(count+1,) or int(self.pointer[0])!=0 or int(self.pointer[-1])!=len(self.value) or bool((self.pointer.diff()<0).any()):raise ValueError('invalid lift pointer')
        if self.index.shape!=self.value.shape or bool(((self.index<0)|(self.index>=atoms)).any()):raise ValueError('invalid atom indices')
        if not bool(torch.isfinite(self.value).all()) or bool((self.value<0).any()):raise ValueError('invalid lift measure')

    def forward(self,accepted,permitted,expanded=True):
        chosen=accepted[self.regions]
        if not bool(chosen.any()):return self.value.new_zeros(self.count,2)
        positive=self.positive[:,chosen].any(1)
        negative=self.negative[:,chosen].all(1)
        if expanded:
            eligible=permitted[self.candidates]
            if bool(eligible.any()):negative=negative|self.interior[:,eligible].any(1)
        if bool((positive&negative).any()):raise ValueError('permission conflicts with positive atoms')
        labels=torch.stack([positive,negative],1).to(self.value.dtype)
        lift=torch.sparse_csr_tensor(self.pointer,self.index,self.value,size=(self.count,len(positive)))
        return torch.sparse.mm(lift,labels)


    def paired(self,accepted,permitted):
        chosen=accepted[self.regions]
        if not bool(chosen.any()):return self.value.new_zeros(self.count,3)
        p=self.positive[:,chosen].any(1);n=self.negative[:,chosen].all(1)
        allowed=permitted[self.candidates]
        expanded=n|self.interior[:,allowed].any(1) if bool(allowed.any()) else n
        if bool((p&expanded).any()):raise ValueError('permission conflicts with positive atoms')
        labels=torch.stack([p,n,expanded],1).to(self.value.dtype)
        lift=torch.sparse_csr_tensor(self.pointer,self.index,self.value,size=(self.count,len(p)))
        return torch.sparse.mm(lift,labels)


class ScopeConstraints(nn.Module):
    def __init__(self,state,count,permissions):
        super().__init__()
        keys={int(k.split('.')[1]) for k in state if k.startswith('views.')}
        if not keys or keys!=set(range(len(keys))):raise ValueError('invalid constraint views')
        self.views=nn.ModuleList([ConstraintView({k[len(f'views.{i}.'):]:v for k,v in state.items() if k.startswith(f'views.{i}.')},count) for i in range(len(keys))])
        if set(state)!=set(self.state_dict()):raise ValueError('unknown constraint state')
        for i,v in enumerate(self.views):
            if not torch.equal(v.regions,torch.where(permissions.accepted_views==i)[0]) or not torch.equal(v.candidates,torch.where(permissions.candidate_views==i)[0]):raise ValueError('permission coordinates changed')

    @torch.no_grad()
    def forward(self,accepted,permissions,*,expanded=True):
        permitted=permissions(accepted)
        values=[v(accepted,permitted,expanded) for v in self.views]
        p=torch.stack([x[:,0] for x in values],1);n=torch.stack([x[:,1] for x in values],1)
        return p,n,dict(admitted_regions=int(accepted.sum()),scope_exclusion=expanded,
            union_before_lifting=True,field_owned_constraints=True)


    @torch.no_grad()
    def paired(self,accepted,permissions):
        permitted=permissions(accepted)
        values=[v.paired(accepted,permitted) for v in self.views]
        p,n,expanded=[torch.stack([x[:,k] for x in values],1) for k in range(3)]
        def result(negative,flag):
            return p,negative,dict(admitted_regions=int(accepted.sum()),scope_exclusion=flag,
                union_before_lifting=True,field_owned_constraints=True,shared_positive_reduction=True)
        return {False:result(n,False),True:result(expanded,True)}


def compile_view(transfer,positive,negative,interior,regions,candidates):
    import numpy as np
    from scipy import sparse
    p=np.asarray(positive,dtype=bool);n=np.asarray(negative,dtype=bool);e=np.asarray(interior,dtype=bool)
    if p.ndim!=2 or p.shape!=n.shape or e.ndim!=2 or e.shape[1]!=p.shape[1] or transfer.shape[1]!=p.shape[1]:raise ValueError('unaligned source relations')
    packed=np.packbits(np.concatenate([p,n,e],axis=0).T,axis=1)
    _,first,inverse=np.unique(packed,axis=0,return_index=True,return_inverse=True)
    assignment=sparse.csr_matrix((np.ones(len(inverse),np.float32),(np.arange(len(inverse)),inverse)),shape=(len(inverse),len(first)))
    lift=(transfer@assignment).tocsr();lift.eliminate_zeros()
    return dict(pointer=torch.from_numpy(lift.indptr.astype(np.int64)),index=torch.from_numpy(lift.indices.astype(np.int64)),
        value=torch.from_numpy(lift.data.astype(np.float32)),positive=torch.from_numpy(p[:,first].T.copy()),
        negative=torch.from_numpy(n[:,first].T.copy()),interior=torch.from_numpy(e[:,first].T.copy()),
        regions=torch.as_tensor(regions,dtype=torch.long),candidates=torch.as_tensor(candidates,dtype=torch.long))
