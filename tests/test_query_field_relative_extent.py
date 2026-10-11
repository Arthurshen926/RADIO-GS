import numpy as np
import pytest
import torch
from radio_gs.query_field.relative_extent_decision import RelativeEvidenceExtentDecision
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from radio_gs.query_field.extent_factors import ExtentFactors


def factors():
    state=dict(pointer=torch.tensor([0,2,3]),index=torch.tensor([0,1,2]),
               alpha=torch.tensor([.8,.7,.9]),visibility=torch.tensor([.8,.7,.9]),
               atom=torch.zeros(2,dtype=torch.long))
    return ExtentFactors({f'views.{i}.'+k:v for i in range(2) for k,v in state.items()},3)


def test_relative_confidence_global_scale_invariance():
    engine=RelativeEvidenceExtentDecision(factors(),3)
    initial=np.array([True,False,True]);y=np.array([1.,0.,1.,0.])
    p=y>.5;n=~p;cost=np.array([.2,.4,.3])
    first=None
    for scale in [1.,.5,.125]:
        result,info=engine.solve(initial,y,p,n,np.array([.5,1.])*scale,cost)
        if first is None:first=(result,info)
        else:
            assert np.array_equal(result,first[0])
            assert info['weighted_mean_energy_trace']==first[1]['weighted_mean_energy_trace']


def test_unit_confidence_reproduces_existing_decisions():
    f=factors();a=LinkedExtentDecision(f,3);b=RelativeEvidenceExtentDecision(f,3)
    args=(np.array([True,False,True]),np.array([1.,0.,1.,0.]),np.array([True,False,True,False]),
          np.array([False,True,False,True]),np.ones(2),np.array([.2,.4,.3]))
    x,u=a.solve(*args);y,v=b.solve(*args)
    assert np.array_equal(x,y) and u['energy_trace']==v['energy_trace']


def test_invalid_and_empty_confidence():
    e=RelativeEvidenceExtentDecision(factors(),3)
    args=(np.zeros(3,dtype=bool),np.zeros(4),np.zeros(4,dtype=bool),np.ones(4,dtype=bool))
    x,r=e.solve(*args,np.zeros(2),np.ones(3))
    assert not x.any() and r['fallback']
    with pytest.raises(ValueError):e.solve(*args,np.array([1.1,0]),np.ones(3))


def test_nonuniform_objective_matches_independent_rendered_reference():
    from test_query_field_extent_decision import reference
    f=factors();e=RelativeEvidenceExtentDecision(f,3)
    initial=np.array([True,True,True]);y=np.array([1.,0.,1.,0.]);p=y>.5;n=~p
    c=np.array([.1,.4]);cost=np.array([.2,.4,.3])
    selected,info=e.solve(initial,y,p,n,c,cost)
    for bits,value in [(initial,info['weighted_mean_energy_trace'][0]),(selected,info['weighted_mean_energy_trace'][-1])]:
        expected=reference(f,bits,y,p,n,c/c.mean(),cost)
        assert abs(expected-value)<2e-6
