import torch
from radio_gs.training.source_evidence_balancing import balance_source_views_and_origins


def test_repeating_an_entire_origin_inventory_cannot_increase_its_vote():
    d=torch.tensor([[1.,0.],[.5,.5],[0.,1.],[0.,1.]])
    w=torch.tensor([[1.,0.],[1.,0.],[1.,1.],[1.,1.]])
    views=torch.tensor([0,0,0,1]);origins=torch.tensor([0,0,1,0])
    expected,count=balance_source_views_and_origins(d,w,views,origins)
    indices=torch.tensor([0,1,0,1,0,1,2,3])
    actual,actual_count=balance_source_views_and_origins(d[indices],w[indices],views[indices],origins[indices])
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert torch.equal(count,actual_count)
    torch.testing.assert_close(expected,torch.tensor([[.1875,.8125],[0.,1.]]))


def test_absent_evidence_is_not_a_zero_feature_vote():
    d=torch.tensor([[1.,0.],[0.,1.]])
    w=torch.tensor([[1.,0.],[0.,0.]])
    value,count=balance_source_views_and_origins(d,w,torch.tensor([0,1]),torch.tensor([0,1]))
    assert torch.equal(value,torch.tensor([[1.,0.],[0.,0.]]))
    assert torch.equal(count,torch.tensor([1.,0.]))


def test_single_origin_without_view_balancing_equals_original_region_fusion():
    d=torch.tensor([[1.,0.],[0.,1.],[.5,.5]])
    w=torch.tensor([[1.,.25],[.5,1.],[.5,.25]])
    actual,count=balance_source_views_and_origins(d,w,torch.zeros(3,dtype=torch.long),torch.zeros(3,dtype=torch.long))
    assert torch.equal(actual,w.T@d/w.sum(0)[:,None])
    assert torch.equal(count,torch.ones(2))
