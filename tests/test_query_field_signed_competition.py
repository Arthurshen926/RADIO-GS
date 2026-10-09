import torch
from radio_gs.query_field.contracts import Support,QueryPacket
from radio_gs.query_field.positive_rays import mixture_logits,query
from radio_gs.query_field.negative_evidence import competitive_membership
from test_query_preserving_field import example


def test_joint_competition_keeps_null_and_compares_conflicting_evidence():
    pos=torch.tensor([3.,-10.]);neg=torch.tensor([2.,-10.])
    actual=competitive_membership(pos,neg)
    torch.testing.assert_close(actual,torch.softmax(torch.stack([pos,neg,torch.zeros_like(pos)],1),1)[:,0])
    assert actual[0]>.5 and (pos.sigmoid()*(1-neg.sigmoid()))[0]<.5
    assert actual[1]<.001


def test_expected_signed_runtime_matches_joint_three_way_readout():
    f=example();p=Support(torch.tensor([0,1]),torch.tensor([.6,.4]),torch.tensor([0,0]));n=Support(torch.tensor([3,5]),torch.tensor([1.,.4]),torch.tensor([0,1]))
    packet=QueryPacket('point_2d',positive=p,negative=n,scale=1.)
    expected=competitive_membership(mixture_logits(f,p,1.,aggregation='expected_union'),mixture_logits(f,n,1.,aggregation='expected_union'))
    actual=query(f,packet,aggregation='expected_union',negative_aggregation='expected_competition')
    torch.testing.assert_close(actual.membership_scores,expected)
    assert torch.equal(actual.selected,expected>f.threshold)
