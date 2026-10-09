import pytest
import torch
from radio_gs.query_field.contracts import Support
from radio_gs.query_field.positive_rays import mixture_logits
from test_query_preserving_field import example


def test_union_preserves_positive_parts_and_full_confidence_duplicates():
    f=example();p=Support(torch.tensor([0,1]),torch.ones(2),torch.tensor([0,1]))
    separate=torch.stack([f.member_logits(Support(torch.tensor([i]),torch.ones(1)),1.) for i in [0,1]],1)
    expected=separate.max(1).values
    torch.testing.assert_close(mixture_logits(f,p,1.,aggregation='union'),expected)
    torch.testing.assert_close(mixture_logits(f,p,1.,aggregation='confidence_union').sigmoid(),expected.sigmoid())
    mixed=mixture_logits(f,p,1.)
    assert bool((expected>=mixed-1e-6).all()) and bool((expected>mixed+1e-4).any())
    duplicate=Support(torch.tensor([0,1,0]),torch.ones(3),torch.tensor([0,1,2]))
    torch.testing.assert_close(mixture_logits(f,duplicate,1.,aggregation='confidence_union'),mixture_logits(f,p,1.,aggregation='confidence_union'))


def test_union_fractional_ray_confidence_is_not_normalized_away():
    f=example();p=Support(torch.tensor([0,1]),torch.tensor([1.,.01]),torch.tensor([0,1]))
    scores=torch.stack([f.member_logits(Support(torch.tensor([i]),torch.ones(1)),1.).sigmoid() for i in [0,1]],1)
    expected=(scores*torch.tensor([1.,.01])).max(1).values
    actual=mixture_logits(f,p,1.,aggregation='confidence_union')
    torch.testing.assert_close(actual.sigmoid(),expected)
    torch.testing.assert_close(actual,mixture_logits(f,p,1.,aggregation='confidence_union',chunk_size=1))
    with pytest.raises(ValueError,match='fractional'):
        mixture_logits(f,Support(torch.tensor([0]),torch.tensor([2.]),torch.tensor([0])),1.,aggregation='confidence_union')
