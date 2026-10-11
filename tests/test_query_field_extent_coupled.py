import numpy as np
import pytest
import torch
from radio_gs.query_field.extent_factors import ExtentFactors
from radio_gs.query_field.extent_preparation import ExtentPreparation
from radio_gs.query_field.extent_decision import render_binary
from radio_gs.query_field.extent_decision_coupled import CoupledExtentDecision


def engine_for(alpha,observed):
    rows,count=alpha.shape
    optical=dict(pointer=torch.arange(0,(rows+1)*count,count),index=torch.arange(count).repeat(rows),
        alpha=torch.from_numpy(alpha.flatten()),visibility=torch.from_numpy(alpha.flatten()),atom=torch.zeros(rows,dtype=torch.long))
    row,col=np.nonzero(observed)
    prep=dict(index=torch.from_numpy(np.stack([row,col])),weight=torch.from_numpy(observed[row,col].astype(np.float32)),
        valid=torch.ones(rows,dtype=torch.bool),footprint=torch.ones(count))
    return CoupledExtentDecision(ExtentFactors({'views.0.'+k:v for k,v in optical.items()},count),
        ExtentPreparation({'views.0.'+k:v for k,v in prep.items()},count),count)


def test_observation_distinguishes_occluded_silhouette_substitutes():
    engine=engine_for(np.array([[.999,.999]],np.float32),np.array([[1.,0.]]))
    args=(np.ones(2,bool),np.ones(1),np.ones(1,bool),np.zeros(1,bool),np.ones(1),np.full(2,.5))
    subset,_=engine.solve(*args,observation_weight=0)
    coupled,_=engine.solve(*args,observation_weight=.5)
    assert subset.tolist()==[False,True]
    assert coupled.tolist()==[True,False]


def test_coupled_every_flip_matches_independent_full_objective():
    rng=np.random.default_rng(20261012)
    for case in range(8):
        n=7;rows=11;alpha=rng.uniform(.01,.95,(rows,n)).astype(np.float32)
        observed=rng.random((rows,n)).astype(np.float32);observed/=observed.sum(1,keepdims=True)
        engine=engine_for(alpha,observed);prior=rng.random(n)>.4;y=rng.random(rows)
        pos=y>.65;neg=y<.3;known=pos|neg;cost=rng.random(n)/n
        coef=.5*(pos/max(pos.sum(),1)+neg/max(neg.sum(),1))
        def objective(bits,mix):
            a=render_binary(engine.pointer,engine.index,engine.alpha,bits).astype(float)
            obs=observed.astype(float)@bits
            total=.1*cost@bits
            for p,w in [(a,1-mix),(obs,mix)]:
                p=np.clip(p,1e-6,1-1e-6);b=np.clip(p,1e-5,1-1e-5)
                loss=-y*np.log(b)-(1-y)*np.log1p(-b)
                total+=w*((coef*loss*known).sum()+1-(2*(p*y*known).sum()+1)/((p*known).sum()+(y*known).sum()+1))
            return total
        for mix in [0.,.5,1.]:
            expected=prior.copy();trace=[objective(expected,mix)]
            for _ in range(4):
                changed=False
                for g in range(n):
                    proposal=expected.copy();proposal[g]=~proposal[g]
                    if objective(proposal,mix)-objective(expected,mix)<-1e-12:expected=proposal;changed=True
                trace.append(objective(expected,mix))
                if not changed:break
            actual,report=engine.solve(prior,y,pos,neg,np.ones(1),cost,observation_weight=mix)
            assert np.array_equal(actual,expected),(case,mix)
            np.testing.assert_allclose(report['energy_trace'],trace,rtol=0,atol=1e-11)
    for mix in [float('nan'),-.1,1.1]:
        with pytest.raises(ValueError):engine.solve(prior,y,pos,neg,np.ones(1),cost,observation_weight=mix)
