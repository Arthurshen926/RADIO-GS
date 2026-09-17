import torch
from radio_gs.training.grounded_text_supervision import corroborated_labels,grounded_pixel_evidence


def test_two_view_evidence_required_and_conflicts_abstain():
    pos,neg=corroborated_labels(torch.tensor([2,1,2,0]),torch.tensor([0,0,2,2]))
    assert pos.tolist()==[True,False,False,False]
    assert neg.tolist()==[False,False,False,True]


def test_no_detection_is_not_background_negative():
    p=torch.tensor([.95,.01,.01,.95])
    pos,neg=grounded_pixel_evidence(p,torch.tensor([False,False,True,False]),torch.ones(4),torch.ones(4),torch.tensor([1.,1.,1.,0.]))
    assert pos.tolist()==[True,False,False,False]
    assert neg.tolist()==[False,False,True,False]


def test_absent_query_abstains_even_with_other_positive_objects():
    pos,neg=grounded_pixel_evidence(torch.zeros(3),torch.ones(3,dtype=torch.bool),torch.ones(3),torch.ones(3),torch.ones(3))
    assert not pos.any() and not neg.any()


def test_pixel_rescue_requires_another_view_and_any_negative_vetoes():
    from radio_gs.training.grounded_text_supervision import independent_positive_carriers
    # Current-view-only evidence cannot confirm itself; a separate view can
    # confirm a pixel even when its carrier straddles this view's boundary.
    counts=torch.tensor([1,1,2,2,0])
    current=torch.tensor([True,False,True,False,False])
    negatives=torch.tensor([0,0,0,1,0])
    assert independent_positive_carriers(counts,current,negatives).tolist()==[False,True,True,False,False]


def test_invalid_independent_vote_counts_rejected():
    import pytest
    from radio_gs.training.grounded_text_supervision import independent_positive_carriers
    with pytest.raises(ValueError):
        independent_positive_carriers(torch.tensor([0]),torch.tensor([True]),torch.tensor([0]))
