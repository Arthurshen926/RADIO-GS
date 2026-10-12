import numpy as np
import torch
from radio_gs.query_field.extent_edit import EditExtentDecision
from radio_gs.query_field.extent_factors import ExtentFactors
from radio_gs.query_field.extent_preparation import ExtentPreparation
from radio_gs.query_field.extent_decision import render_binary


def make_engine(alpha,obs):
    rows,n=alpha.shape;rr,cc=np.nonzero(obs)
    a=dict(pointer=torch.arange(0,(rows+1)*n,n),index=torch.arange(n).repeat(rows),alpha=torch.from_numpy(alpha.ravel()),
        visibility=torch.from_numpy(alpha.ravel()),atom=torch.zeros(rows,dtype=torch.long))
    b=dict(index=torch.from_numpy(np.stack([rr,cc])),weight=torch.from_numpy(obs[rr,cc]),valid=torch.ones(rows,dtype=torch.bool),footprint=torch.ones(n))
    return EditExtentDecision(ExtentFactors({'views.0.'+k:v for k,v in a.items()},n),ExtentPreparation({'views.0.'+k:v for k,v in b.items()},n),n)


def test_edit_and_unpenalized_decisions_match_full_xor_objective():
    rng=np.random.default_rng(5122026);rows=13;n=8
    for case in range(6):
        a=rng.uniform(.004,.99,(rows,n)).astype(np.float32);obs=rng.random((rows,n)).astype(np.float32);obs/=obs.sum(1,keepdims=True)
        engine=make_engine(a,obs);prior=rng.random(n)>.5;target=rng.random(rows);p=target>.6;neg=target<.3;known=p|neg
        cost=rng.random(n)/n;coef=.5*(p/max(p.sum(),1)+neg/max(neg.sum(),1))
        def objective(bits,mode):
            total=.1*cost@(bits!=prior) if mode=='edit' else 0.
            optical=render_binary(engine.pointer,engine.index,engine.alpha,bits)
            for q in [optical,obs.astype(float)@bits]:
                q=np.clip(q.astype(float),1e-6,1-1e-6);b=np.clip(q,1e-5,1-1e-5)
                loss=-target*np.log(b)-(1-target)*np.log1p(-b)
                total+=.5*((coef*loss*known).sum()+1-(2*(q*target*known).sum()+1)/((q*known).sum()+(target*known).sum()+1))
            return total
        for mode in ['edit','none']:
            expected=prior.copy();trace=[objective(expected,mode)]
            for _ in range(4):
                changed=False
                for g in range(n):
                    trial=expected.copy();trial[g]=~trial[g]
                    if objective(trial,mode)-objective(expected,mode)<-1e-12:expected=trial;changed=True
                trace.append(objective(expected,mode))
                if not changed:break
            actual,report=engine.solve(prior,target,p,neg,np.ones(1),cost,regularizer=mode)
            assert np.array_equal(actual,expected),(case,mode)
            np.testing.assert_allclose(report['energy_trace'],trace,atol=1e-11,rtol=0)


def test_finite_edit_cost_can_remove_all_inferred_positive_members():
    engine=make_engine(np.full((1,2),.8,np.float32),np.full((1,2),.5,np.float32))
    selected,report=engine.solve(np.ones(2,bool),np.zeros(1),np.zeros(1,bool),np.ones(1,bool),np.ones(1),np.full(2,.5))
    assert not selected.any() and not report['prior_members_locked']
    assert report['removed']==2 and report['energy_trace'][-1]<report['energy_trace'][0]
