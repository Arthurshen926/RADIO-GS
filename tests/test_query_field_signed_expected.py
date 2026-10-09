from dataclasses import replace
import torch
from radio_gs.query_field.contracts import Support,QueryPacket
from radio_gs.query_field.positive_rays import mixture_logits,query
from test_query_preserving_field import example


def test_signed_expected_uses_same_uncertain_kernel_on_each_sign():
    f=example();positive=Support(torch.tensor([0,1]),torch.tensor([.6,.4]),torch.tensor([0,0]))
    negative=Support(torch.tensor([4,5,7]),torch.tensor([.3,.4,1.]),torch.tensor([0,0,1]))
    p=QueryPacket('point_2d',positive=positive,negative=negative,scale=1.)
    expected=mixture_logits(f,positive,1.,aggregation='expected_union').sigmoid()*(1-mixture_logits(f,negative,1.,aggregation='expected_union').sigmoid())
    actual=query(f,p,aggregation='expected_union',negative_aggregation='expected_union')
    torch.testing.assert_close(actual.membership_scores,expected)
    assert torch.equal(actual.selected,expected>f.threshold)
    # Identical independent negative constraints cannot gain confidence by duplication.
    doubled=Support(torch.cat([negative.indices,negative.indices]),torch.cat([negative.weights,negative.weights]),torch.cat([negative.components,negative.components+2]))
    torch.testing.assert_close(query(f,replace(p,negative=doubled),aggregation='expected_union',negative_aggregation='expected_union').membership_scores,expected)
