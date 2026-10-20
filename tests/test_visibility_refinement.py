import pytest
import torch
from radio_gs.querying.response_permissions import admit_response_observation
from radio_gs.querying.response_refinement import refine_response
from radio_gs.querying.visibility_refinement import VisibilityNormalizedRefiner


def item(y, positive=True, reliability=1.):
    target=torch.tensor([[y]])
    return admit_response_observation(torch.ones(1,1).to_sparse(),target,torch.tensor([[positive]]),
        torch.tensor([[not positive]]),torch.tensor(reliability))[0]


def test_one_disagreement_does_not_erase_nine_consistent_observations():
    prior=torch.tensor([[.8]])
    observations=[item(1.)]+[item(.8) for _ in range(9)]
    assert refine_response(prior,observations)[0].item()==1.
    r=VisibilityNormalizedRefiner(prior)
    for e in observations:r.add(e)
    assert torch.allclose(r.finish()[0],torch.tensor([[.82]]))


def test_unknown_constrains_step_without_becoming_negative_evidence():
    prior=torch.tensor([[.2]])
    r=VisibilityNormalizedRefiner(prior)
    r.add(item(1.))
    for _ in range(9):r.add(item(0.,positive=False))
    assert torch.allclose(r.finish()[0],torch.tensor([[.28]]))
    r=VisibilityNormalizedRefiner(prior);r.add(item(0.,positive=False))
    assert torch.equal(r.finish()[0],prior)


def test_absolute_reliability_does_not_cancel_out_of_single_observation():
    r=VisibilityNormalizedRefiner(torch.tensor([[.2]]));r.add(item(1.,reliability=.1))
    assert torch.allclose(r.finish()[0],torch.tensor([[.28]]))
    with pytest.raises(ValueError,match='bounded reliability'):
        r.add(item(1.,reliability=2.))


def test_valid_negative_can_still_lower_a_high_prior():
    target=torch.zeros(1,1)
    evidence,_=admit_response_observation(torch.ones(1,1).to_sparse(),target,target.bool(),~target.bool(),
        torch.tensor(1.),exclusion=~target.bool())
    r=VisibilityNormalizedRefiner(torch.tensor([[.95]]));r.add(evidence)
    assert r.finish()[0].item()==0.
