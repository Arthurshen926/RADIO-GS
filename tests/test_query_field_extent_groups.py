import numpy as np
import pytest
import torch
from radio_gs.query_field.extent_groups import incidence_partition,GroupedExtentDecision
from radio_gs.query_field.extent_factors import ExtentFactors
from radio_gs.query_field.extent_preparation import ExtentPreparation
from radio_gs.query_field.extent_decision import render_binary
from radio_gs.query_field.extent_decision_coupled import block_response


def test_incidence_unknown_and_shuffle_preserve_initial_and_group_sizes():
    pointer=np.array([0,2,4,6,8,9,10,10,10,10,10])
    index=np.array([0,2,0,2,0,2,0,2,1,1]);domain=np.ones(10,bool);domain[-1]=False
    initial=np.array([1,1,0,0,1,1,0,1,0,0],bool)
    g,info=incidence_partition(pointer,index,3,domain,initial)
    shuffled,_=incidence_partition(pointer,index,3,domain,initial,shuffle=True)
    assert g[0]==g[1] and g[2]==g[3] and g[0]!=g[2] and g[4]==g[5]
    assert len(set(g[6:9]))==3 and g[-1]==-1 and info['no_region_members_are_singletons']==3
    assert np.array_equal(np.bincount(g[domain]),np.bincount(shuffled[domain]))
    assert np.array_equal(g[6:],shuffled[6:])
    for part in [g,shuffled]:
        assert all(len(np.unique(initial[part==i]))==1 for i in np.unique(part[domain]))
    again,_=incidence_partition(pointer,index,3,domain,initial,shuffle=True)
    assert np.array_equal(again,shuffled)


def test_grouped_all_moves_match_original_gaussian_full_objective():
    rng=np.random.default_rng(20261012);groups=np.array([0,0,1,2,2,-1,3,3],np.int64)
    n=len(groups);members=np.flatnonzero(groups>=0);rows=9;state={};prep={};matrices=[]
    for v in range(2):
        index=np.concatenate([rng.permutation(members) for _ in range(rows)])
        alpha=rng.uniform(.004,.999,len(index)).astype(np.float32)
        # Repeated hits of each group per row, including alpha early termination.
        a=dict(pointer=torch.arange(0,(rows+1)*len(members),len(members)),index=torch.from_numpy(index),
            alpha=torch.from_numpy(alpha),visibility=torch.from_numpy(alpha),atom=torch.zeros(rows,dtype=torch.long))
        obs=rng.random((rows,n)).astype(np.float32);obs[:,5]=0;obs/=obs.sum(1,keepdims=True)
        rr,cc=np.nonzero(obs);b=dict(index=torch.from_numpy(np.stack([rr,cc])),weight=torch.from_numpy(obs[rr,cc]),
            valid=torch.ones(rows,dtype=torch.bool),footprint=torch.ones(n))
        state.update({f'views.{v}.'+k:t for k,t in a.items()});prep.update({f'views.{v}.'+k:t for k,t in b.items()});matrices.append(obs)
    factors=ExtentFactors(state,n);preparation=ExtentPreparation(prep,n)
    engine=GroupedExtentDecision(factors,preparation,n,groups)
    for case in range(6):
        bits=rng.random(4)>.5;initial=np.zeros(n,bool);initial[members]=bits[groups[members]]
        target=rng.random(2*rows);positive=target>.6;negative=target<.4;cost=rng.random(n)/n;cost[5]=0
        confidence=np.array([.8,0. if case%2 else .3]);active=confidence>0
        def objective(proposal,mix):
            total=.1*cost@proposal
            for v in range(2):
                if not active[v]:continue
                a=render_binary(state[f'views.{v}.pointer'].numpy(),state[f'views.{v}.index'].numpy(),state[f'views.{v}.alpha'].numpy(),proposal)
                y=target[v*rows:(v+1)*rows];p=positive[v*rows:(v+1)*rows];neg=negative[v*rows:(v+1)*rows];known=p|neg
                coef=.5*(p/max(p.sum(),1)+neg/max(neg.sum(),1))
                for response,weight in [(a,1-mix),(matrices[v].astype(float)@proposal,mix)]:
                    q=np.clip(response.astype(float),1e-6,1-1e-6);b=np.clip(q,1e-5,1-1e-5)
                    ce=-y*np.log(b)-(1-y)*np.log1p(-b)
                    loss=(coef*ce*known).sum()+1-(2*(q*y*known).sum()+1)/((q*known).sum()+(y*known).sum()+1)
                    total+=weight*confidence[v]/active.sum()*loss
            return total
        for mix in [0.,.5,1.]:
            expected=initial.copy();trace=[objective(expected,mix)]
            for _ in range(4):
                changed=False
                for g in range(4):
                    trial=expected.copy();trial[groups==g]=~trial[groups==g]
                    if objective(trial,mix)-objective(expected,mix)<-1e-12:expected=trial;changed=True
                trace.append(objective(expected,mix))
                if not changed:break
            actual,report=engine.solve(initial,target,positive,negative,confidence,cost,observation_weight=mix)
            assert np.array_equal(actual,expected),(case,mix)
            np.testing.assert_allclose(report['energy_trace'],trace,atol=1e-11,rtol=0)
    bad=initial.copy();bad[0]=~bad[1]
    with pytest.raises(ValueError):engine.solve(bad,target,positive,negative,confidence,cost)


def test_group_can_remove_all_initial_positive_members():
    alpha=torch.tensor([.8,.8]);s=dict(pointer=torch.tensor([0,2]),index=torch.tensor([0,1]),alpha=alpha,visibility=alpha,atom=torch.tensor([0]))
    p=dict(index=torch.tensor([[0,0],[0,1]]),weight=torch.tensor([.5,.5]),valid=torch.tensor([True]),footprint=torch.ones(2))
    e=GroupedExtentDecision(ExtentFactors({'views.0.'+k:v for k,v in s.items()},2),ExtentPreparation({'views.0.'+k:v for k,v in p.items()},2),2,np.zeros(2,np.int64))
    selected,_=e.solve(np.ones(2,bool),np.zeros(1),np.zeros(1,bool),np.ones(1,bool),np.ones(1),np.ones(2))
    assert not selected.any()


def test_block_optics_early_exit_and_subthreshold_hits_in_all_states():
    group=np.array([0,1,0,2,0],np.int64);alpha=np.array([.999,.9,.0001,.999,.8],np.float32)
    for state in range(8):
        chosen=np.array([bool(state&(1<<g)) for g in range(3)])
        head=np.array([-1],np.int32);following=np.full(5,-1,np.int32)
        for hit in range(4,-1,-1):
            if chosen[group[hit]]:following[hit]=head[0];head[0]=hit
        for g in range(3):
            positions=np.flatnonzero(group==g).astype(np.int32)
            actual=block_response(head,following,alpha,group,0,g,positions,0,len(positions),chosen[g])
            proposal=chosen.copy();proposal[g]=~proposal[g]
            # Independent original-Gaussian membership, including shader stop
            # BEFORE the hit that would take transmittance below 1e-4.
            trans=np.float32(1.)
            for a,selected in zip(alpha,proposal[group]):
                if not selected or a<1/255:continue
                nxt=np.float32(trans*np.float32(1-a))
                if nxt<=1e-4:break
                trans=nxt
            assert actual==np.float32(1)-trans
