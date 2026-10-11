import numpy as np
import pytest
import torch
from radio_gs.query_field.reference_extent import reference_scope,prepare_reference
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision,probability_at_cutoff
from radio_gs.query_field.extent_decision import CUTOFF,probability,render_binary
from radio_gs.query_field.extent_factors import ExtentFactors
from radio_gs.query_field.extent_preparation import ExtentPreparation


def test_explicit_complete_reference_fractional_boundary_and_invalid_rays():
    valid=torch.tensor([True,True,True,False])
    s=reference_scope(torch.tensor([[1.,.25],[0.,0.]]),valid)
    assert s['target'].tolist()==[1.,.25,0.,0.]
    assert s['positive'].tolist()==[True,True,False,False]
    assert s['negative'].tolist()==[False,False,True,False]
    with pytest.raises(ValueError):reference_scope(torch.zeros(4),valid)


def test_strong_prior_is_not_silently_replaced_by_visible_lift():
    module=ExtentPreparation({'views.0.'+k:v for k,v in dict(
        index=torch.tensor([[0,1],[0,1]]),weight=torch.ones(2),valid=torch.ones(2,dtype=torch.bool),
        footprint=torch.tensor([2.,3.,4.])).items()},3)
    prior=torch.tensor([0.,1.,1.]);scope=reference_scope(torch.tensor([1.,0.]),module.views[0].valid)
    p,c,_=prepare_reference(module,prior,scope,'prior')
    lifted,lc,_=prepare_reference(module,prior,scope,'lifted')
    assert torch.equal(p,prior) and torch.equal(lifted,torch.tensor([1.,0.,1.]))
    assert float(c@p)==pytest.approx(1.) and float(lc@lifted)==pytest.approx(1.)
    assert torch.equal(c,torch.tensor([2.,3.,4.])/7)


def test_task_boundary_and_legacy_probability_exact():
    for a in np.r_[0.,np.linspace(.001,.999,101),1.]:
        assert probability_at_cutoff(a,CUTOFF)==probability(a)
    assert probability_at_cutoff(.5,.5)==.5
    assert probability_at_cutoff(CUTOFF,CUTOFF)==pytest.approx(.5)
    assert probability_at_cutoff(.2,.5)<.5<probability_at_cutoff(.2,CUTOFF)


def test_typed_linked_decisions_match_full_objective_flip_enumeration():
    rng=np.random.default_rng(61);n=7;r=13
    idx=np.tile(np.arange(n),r);alpha=rng.uniform(.01,.8,len(idx)).astype(np.float32)
    state=dict(pointer=torch.arange(0,(r+1)*n,n),index=torch.from_numpy(idx),alpha=torch.from_numpy(alpha),
               visibility=torch.from_numpy(alpha),atom=torch.zeros(r,dtype=torch.long))
    engine=LinkedExtentDecision(ExtentFactors({'views.0.'+k:v for k,v in state.items()},n),n)
    target=rng.uniform(0,1,r);p=target>.5;neg=~p;prior=rng.random(n)>.5;cost=rng.uniform(.01,.1,n)
    def objective(selected,cutoff):
        a=np.clip(render_binary(engine.pointer,engine.index,engine.alpha,selected).astype(np.float64),1e-6,1-1e-6)
        prob=a*(1-cutoff)/(a*(1-cutoff)+cutoff*(1-a));b=np.clip(prob,1e-5,1-1e-5)
        ce=-target*np.log(b)-(1-target)*np.log1p(-b)
        return .1*cost@selected+.5*(ce[p].mean()+ce[neg].mean())+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    for cutoff in [CUTOFF,.5,.8]:
        expected=prior.copy();trace=[objective(expected,cutoff)]
        for _ in range(4):
            changed=False
            for i in range(n):
                proposal=expected.copy();proposal[i]=~proposal[i]
                if objective(proposal,cutoff)-objective(expected,cutoff)<-1e-12:
                    expected=proposal;changed=True
            trace.append(objective(expected,cutoff))
            if not changed:break
        actual,report=engine.solve(prior,target,p,neg,np.ones(1),cost,decision_cutoff=cutoff)
        assert np.array_equal(actual,expected)
        np.testing.assert_allclose(report['energy_trace'],trace,atol=1e-12,rtol=0)
    for cutoff in [0.,1.,np.nan]:
        with pytest.raises(ValueError):engine.solve(prior,target,p,neg,np.ones(1),cost,decision_cutoff=cutoff)
