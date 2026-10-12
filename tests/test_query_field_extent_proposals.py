import numpy as np
import pytest
from test_query_field_extent_edit import make_engine
from radio_gs.query_field.extent_decision import render_binary
from radio_gs.query_field.extent_proposals import region_union,observation_strata,proposal_domain


def test_candidate_scope_is_not_a_required_foreground_group():
    pointer=np.array([0,1,3,4,4,5]);index=np.array([0,0,1,1,0])
    available=np.array([1,1,1,1,0],bool)
    candidate=region_union(pointer,index,np.array([1,0],bool),available)
    np.testing.assert_array_equal(candidate,[1,1,0,0,0])
    initial=np.array([1,0,0,0,0],bool)
    domain,info=proposal_domain(candidate,initial,available,np.zeros(5,int))
    assert domain[1] and not initial[1] and not info['initial_members_locked']
    engine=make_engine(np.full((1,5),.8,np.float32),np.full((1,5),.2,np.float32))
    selected,_=engine.solve(initial,np.zeros(1),np.zeros(1,bool),np.ones(1,bool),np.ones(1),np.full(5,.2),proposal_domain=domain)
    assert not selected.any()
    with pytest.raises(ValueError,match='contain'):engine.solve(initial,np.zeros(1),np.zeros(1,bool),np.ones(1,bool),np.ones(1),np.full(5,.2),proposal_domain=np.zeros(5,bool))


def test_stratified_control_preserves_counts_and_prior_without_unknown_labels():
    rng=np.random.default_rng(9);available=np.ones(300,bool);available[-10:]=False
    initial=(rng.random(300)<.2)&available;candidate=((rng.random(300)<.3)|initial)&available
    strata=rng.integers(0,4,300)
    shuffled,info=proposal_domain(candidate,initial,available,strata,mode='shuffled')
    assert (shuffled[initial]).all() and not shuffled[~available].any()
    for s in range(4):assert (shuffled&~initial&(strata==s)).sum()==(candidate&~initial&(strata==s)).sum()
    assert not np.array_equal(candidate,shuffled) and not info['outside_support_is_negative']
    repeated,_=proposal_domain(candidate,initial,available,strata,mode='shuffled');assert np.array_equal(repeated,shuffled)
    prior,_=proposal_domain(candidate,initial,available,strata,mode='prior_only');assert np.array_equal(prior,initial)
    ids=np.array([[0,0,1,1],[0,1,1,2]])
    np.testing.assert_array_equal(observation_strata(5,ids,np.ones(4),np.array([1.,0.]),np.array([1,1],bool)),[2,1,3,0,0])


def test_restricted_decisions_match_independent_full_objective_traces():
    rng=np.random.default_rng(4122026);rows=11;n=9
    for case in range(12):
        alpha=rng.uniform(.004,.99,(rows,n)).astype(np.float32)
        obs=rng.random((rows,n)).astype(np.float32);obs/=obs.sum(1,keepdims=True)
        engine=make_engine(alpha,obs);prior=rng.random(n)>.6;domain=prior|(rng.random(n)>.5)
        target=rng.random(rows);p=target>.65;neg=target<.4;known=p|neg
        coef=.5*(p/max(p.sum(),1)+neg/max(neg.sum(),1));cost=rng.random(n)/n
        def objective(bits):
            total=.1*cost@(bits!=prior)
            for q in [render_binary(engine.pointer,engine.index,engine.alpha,bits),obs.astype(float)@bits]:
                q=np.clip(q.astype(float),1e-6,1-1e-6);b=np.clip(q,1e-5,1-1e-5)
                total+=.5*((coef*(-target*np.log(b)-(1-target)*np.log1p(-b))*known).sum()
                    +1-(2*(q*target*known).sum()+1)/((q*known).sum()+(target*known).sum()+1))
            return total
        expected=prior.copy();trace=[objective(expected)]
        for _ in range(4):
            changed=False
            for g in np.flatnonzero(domain):
                trial=expected.copy();trial[g]=~trial[g]
                if objective(trial)-objective(expected)<-1e-12:expected=trial;changed=True
            trace.append(objective(expected))
            if not changed:break
        actual,report=engine.solve(prior,target,p,neg,np.ones(1),cost,proposal_domain=domain)
        assert np.array_equal(actual,expected) and not actual[~domain].any()
        np.testing.assert_allclose(report['energy_trace'],trace,atol=1e-11,rtol=0)
        free,rfree=engine.solve(prior,target,p,neg,np.ones(1),cost)
        full,rfull=engine.solve(prior,target,p,neg,np.ones(1),cost,proposal_domain=np.ones(n,bool))
        assert np.array_equal(free,full)
        np.testing.assert_array_equal(rfree['energy_trace'],rfull['energy_trace'])
