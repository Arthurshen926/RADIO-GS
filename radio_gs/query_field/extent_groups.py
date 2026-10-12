"""Reversible blocks from field-owned region-incidence equivalence classes.

Shared incidence is a partial grouping hypothesis, not an instance label. Split
by the current initial bit to preserve that initializer exactly. Empty evidence
keeps singleton freedom; it never creates one giant 'unknown object'.
"""
import numpy as np
from scipy import sparse
from .extent_decision_coupled import CoupledExtentDecision,coupled_decisions
from .extent_decision_linked import solve


def incidence_partition(pointer,index,regions,domain,initial,*,shuffle=False,seed=20261012):
    pointer=np.asarray(pointer);index=np.asarray(index);domain=np.asarray(domain);initial=np.asarray(initial)
    count=len(domain)
    if (pointer.shape!=(count+1,) or domain.dtype!=bool or initial.shape!=domain.shape or initial.dtype!=bool
            or pointer[0]!=0 or pointer[-1]!=len(index) or np.any(np.diff(pointer)<0)
            or regions<1 or np.any(index<0) or np.any(index>=regions) or np.any(initial&~domain)):
        raise ValueError('invalid field incidence or initial domain')
    covered=np.diff(pointer)>0;members=np.flatnonzero(domain&covered)
    packed=np.zeros((count,(regions+7)//8+1),np.uint8)
    rows=np.repeat(np.arange(count),np.diff(pointer))
    np.bitwise_or.at(packed,(rows,index//8),(1<<(index%8)).astype(np.uint8))
    packed[:,-1]=initial
    groups=np.full(count,-1,np.int64)
    if len(members):
        _,inverse=np.unique(packed[members],axis=0,return_inverse=True);groups[members]=inverse
        next_id=int(inverse.max())+1
    else:next_id=0
    unknown=np.flatnonzero(domain&~covered);groups[unknown]=np.arange(next_id,next_id+len(unknown))
    # Canonical group order is increasing first Gaussian, independent of hash or
    # arbitrary region numbering. The shuffled control preserves these slots.
    if domain.any():
        _,first=np.unique(groups[domain],return_index=True)
        order=np.argsort(first);remap=np.empty(len(order),np.int64);remap[order]=np.arange(len(order))
        groups[domain]=remap[groups[domain]]
    if shuffle:
        rng=np.random.default_rng(seed)
        for bit in [False,True]:
            ids=np.flatnonzero(domain&covered&(initial==bit));groups[ids]=groups[rng.permutation(ids)]
    sizes=np.bincount(groups[domain])
    return groups,dict(groups=len(sizes),grouped_members=int(sizes[sizes>1].sum()),
        singleton_groups=int((sizes==1).sum()),max_group_size=int(sizes.max()) if len(sizes) else 0,
        no_region_members_are_singletons=int(len(unknown)),initial_state_preserved=True,
        shuffled=bool(shuffle),shuffle_seed=seed if shuffle else None,
        equivalence='same positive region incidence and current initial bit',instance_identity_certified=False)


class GroupedExtentDecision(CoupledExtentDecision):
    def __init__(self,factors,preparation,count,groups):
        super().__init__(factors,preparation,count)
        self.member_count=count;self.groups=np.asarray(groups)
        if self.groups.shape!=(count,) or self.groups.dtype!=np.int64 or np.any(self.groups < -1):
            raise ValueError('invalid member partition')
        ids=np.unique(self.groups[self.groups>=0])
        if not np.array_equal(ids,np.arange(len(ids))) or not len(ids):raise ValueError('noncontiguous or empty partition')
        self.count=len(ids);rows=len(self.rowview)
        original_obs=sparse.csc_matrix((self.oval,self.orows,self.optr),shape=(rows,count)).tocoo()
        included=self.groups[original_obs.col]>=0
        op=sparse.coo_matrix((original_obs.data[included],(original_obs.row[included],self.groups[original_obs.col[included]])),shape=(rows,self.count)).tocsc()
        op.sum_duplicates();op.sort_indices();self.optr,self.orows,self.oval=op.indptr,op.indices,op.data
        self.index=self.groups[self.index]
        if np.any(self.index<0):raise ValueError('optical hits outside partition domain')
        ray=np.repeat(np.arange(rows),np.diff(self.pointer));key=self.index*rows+ray
        order=np.argsort(key,kind='stable');sorted_key=key[order]
        start=np.r_[0,np.flatnonzero(np.diff(sorted_key))+1] if len(order) else np.empty(0,np.int64)
        self.block_pointer=np.r_[start,len(order)].astype(np.int64)
        self.colposition=order.astype(np.int32)
        self.colrows=(sorted_key[start]%rows).astype(np.int32)
        sizes=np.bincount(sorted_key[start]//rows,minlength=self.count)
        self.colptr=np.r_[0,np.cumsum(sizes)].astype(np.int64)

    def solve(self,initial,target,positive,negative,confidence,cost,*,max_passes=4,decision_cutoff=.5,observation_weight=.5):
        initial=np.asarray(initial);cost=np.asarray(cost,dtype=float);domain=self.groups>=0
        if (initial.shape!=(self.member_count,) or initial.dtype!=bool or cost.shape!=initial.shape
                or np.any(initial[~domain]) or np.any(cost[~domain]!=0)
                or not np.isfinite(observation_weight) or not 0<=observation_weight<=1):
            raise ValueError('invalid grouped initializer/cost/weight')
        size=np.bincount(self.groups[domain],minlength=self.count)
        positives=np.bincount(self.groups[domain],weights=initial[domain],minlength=self.count)
        if np.any((positives!=0)&(positives!=size)):raise ValueError('groups must preserve the supplied initial state')
        group_initial=positives>0;group_cost=np.bincount(self.groups[domain],weights=cost[domain],minlength=self.count)
        def kernel(*args):
            return coupled_decisions(*args,self.optr,self.orows,self.oval,float(observation_weight),self.block_pointer)
        result,report=solve(self,group_initial,target,positive,negative,confidence,group_cost,
            max_passes=max_passes,decision_cutoff=decision_cutoff,_kernel=kernel)
        selected=np.zeros(self.member_count,bool);selected[domain]=result[self.groups[domain]]
        report.update(group_added=report.get('added',0),group_removed=report.get('removed',0),
            added=int((selected&~initial).sum()),removed=int((initial&~selected).sum()),
            finite_decision_unit='reversible field-incidence group',observation_weight=float(observation_weight),
            same_members_for_both_operators=True,prior_members_locked=False)
        return selected,report
