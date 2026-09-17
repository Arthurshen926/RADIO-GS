import torch
from scripts.audits.fit_cross_view_region_ranker import cross_view_quality


def test_candidate_origin_and_episode_target_do_not_supply_quality():
    scores=torch.tensor([[[1.,.1]],[[.2,1.]],[[.6,.8]]])
    present=torch.ones(3,1,dtype=torch.bool);origin=torch.tensor([0,1])
    quality,count=cross_view_quality(scores,present,origin,excluded=2)
    torch.testing.assert_close(quality,torch.tensor([[.2,.1]]))
    assert count.tolist()==[[1,1]]
    quality,count=cross_view_quality(scores,present,origin)
    torch.testing.assert_close(quality,torch.tensor([[.4,.45]]))


def test_undetected_views_are_unknown_and_no_evidence_is_not_division_by_zero():
    scores=torch.tensor([[[1.,.1]],[[.2,1.]],[[.6,.8]]])
    present=torch.tensor([[True],[False],[False]]);origin=torch.tensor([0,1])
    quality,count=cross_view_quality(scores,present,origin)
    torch.testing.assert_close(quality,torch.tensor([[0.,.1]]))
    assert count.tolist()==[[0,1]]
