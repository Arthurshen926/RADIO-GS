import torch
from scripts.audits.source_uplift_scope_next20 import local_correction
from scripts.audits.source_selected_calibration_next20 import grouped_predictions


def test_unknown_and_conflicting_evidence_cannot_rewrite_baseline():
    base=torch.tensor([0.,.2,.8,1.])
    result,valid=local_correction(base,torch.tensor([1.,0.,.5,0.]),torch.tensor([0.,.2,0.,0.]),torch.tensor([0,4,4,1]))
    assert not valid.any()
    assert torch.equal(result,base)


def test_consensus_changes_only_eligible_local_regions_and_is_bounded():
    base=torch.tensor([.4,.6,.6])
    result,valid=local_correction(base,torch.tensor([.95,.05,.9]),torch.zeros(3),torch.tensor([3,3,1]))
    assert valid.tolist()==[True,True,False]
    assert result[0]>.5 and result[1]<.5
    assert result[2]==base[2]
    assert (torch.logit(result[:2])-torch.logit(base[:2])).abs().max()<=2.197225


def test_duplicate_source_queries_are_excluded_together():
    q=torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    p1,groups=grouped_predictions(q,torch.tensor([.1,.1,.3,.5]))
    p2,_=grouped_predictions(q,torch.tensor([.9,.9,.3,.5]))
    assert groups[0]==groups[1]
    assert torch.equal(p1[:2],p2[:2])


def test_opt_in_runtime_preserves_other_modalities_and_unknown_rows():
    from radio_gs.querying.source_scope_consensus import refine_source_scope_consensus
    scores=torch.tensor([[.4,.8],[.6,.2],[.6,.7]])
    observed=torch.tensor([True,True,False]);mean=torch.tensor([[.95],[.05],[.9]])
    var=torch.zeros_like(mean);count=torch.tensor([[3],[3],[3]])
    args=(scores,observed,mean,var,count,torch.tensor([0,0]),torch.tensor([True,False]))
    assert refine_source_scope_consensus(*args,modality='text') is scores
    assert refine_source_scope_consensus(*args,modality='image',enabled=True) is scores
    assert refine_source_scope_consensus(*args,modality='prompt',enabled=True) is scores
    before=scores.clone();actual=refine_source_scope_consensus(*args,modality='text',enabled=True)
    assert torch.equal(scores,before)
    assert torch.equal(actual[:,1],scores[:,1]) and torch.equal(actual[2],scores[2])
    assert actual[0,0]>.5 and actual[1,0]<.5


def test_fractional_routes_are_rejected_instead_of_truncated():
    import pytest
    from radio_gs.querying.source_scope_consensus import refine_source_scope_consensus
    with pytest.raises(ValueError,match='integer'):
        refine_source_scope_consensus(torch.tensor([[.4]]),torch.tensor([True]),torch.tensor([[.9]]),torch.tensor([[0.]]),torch.tensor([[3]]),torch.tensor([.5]),torch.tensor([True]),modality='text',enabled=True)
