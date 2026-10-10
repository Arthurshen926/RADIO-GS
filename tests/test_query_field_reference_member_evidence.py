import pytest
import torch
from radio_gs.query_field.reference_scope import reference_member_evidence


def test_reference_member_evidence_preserves_unseen_and_ties_but_rejects_background():
    operator=torch.tensor([[1.,0,0,0,0],[0,1.,0,0,0],[0,0,.5,0,.5],[0,0,.5,0,.5]]).to_sparse()
    prior=torch.tensor([True,True,True,True,False])
    scores,selected,diagnostics=reference_member_evidence(operator,torch.ones(5,dtype=torch.bool),torch.tensor([1.,0,1,0]),prior)
    torch.testing.assert_close(scores,torch.tensor([1.,0,.5,1,0]))
    assert selected.tolist()==[True,False,True,True,False]
    assert diagnostics['reference_unseen_members']==1
    assert diagnostics['reference_unseen_changed']==0
    assert diagnostics['reference_rejected_members']==1
    assert diagnostics['reference_tied_members']==1


def test_unobserved_pixels_abstain_and_fractional_coverage_is_not_binarized():
    operator=torch.tensor([[.1,.9,0],[0,0,1.]]).to_sparse()
    prior=torch.tensor([True,False,True]);observed=torch.tensor([True,False,True])
    scores,selected,d=reference_member_evidence(operator,observed,torch.tensor([0.,.6]),prior)
    torch.testing.assert_close(scores,torch.tensor([1.,0,.6]))
    assert selected.tolist()==prior.tolist() and d['reference_unseen_members']==1
    with pytest.raises(ValueError):reference_member_evidence(operator,observed,torch.tensor([0.,float('nan')]),prior)


def test_visible_fraction_cannot_resolve_occluded_identity():
    # Both rows are labelled background in the mask, but the second primitive
    # could be a target behind a 99% opaque occluder. Tiny nonzero visibility
    # is still evidence here; do not claim this diagonal vote solves identity.
    operator=torch.tensor([[.99,.01,0.]]).to_sparse()
    prior=torch.tensor([False,True,True])
    scores,selected,d=reference_member_evidence(operator,torch.ones(3,dtype=torch.bool),torch.tensor([0.]),prior)
    assert selected.tolist()==[False,False,True]
    assert d['weak_visibility_is_not_identity'] and scores[-1]==1
