import copy

import pytest
import torch

from radio_gs.query_field.adapters import scribble_packet
from radio_gs.query_field.contracts import QueryPacket, Support
from radio_gs.query_field.negative_evidence import competitive_membership, ray_log_mixture
from radio_gs.query_field.storage import save_field, load_field
from test_query_preserving_field import example


def test_competition_matches_three_way_normalization_and_preserves_raw_confidence():
    positive = torch.tensor([-3., 2., 8., 100.], requires_grad=True)
    negative = torch.tensor([2., -1., 8., 90.], requires_grad=True)
    expected = torch.stack([torch.zeros_like(positive), positive, negative],1).softmax(1)[:,1]
    actual = competitive_membership(positive, negative)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert actual[2] > .49 and actual[3] > .999
    assert bool((actual <= positive.sigmoid()).all())
    actual.sum().backward()
    assert (positive.grad >= 0).all() and (negative.grad <= 0).all()


@pytest.mark.parametrize('local', ['none','native_cosine'])
def test_ray_mixture_is_explicit_mass_weighted_sum_with_split_invariance(local):
    field = example()
    field.spatial_metric = 'scale_normalized'; field.world_units_per_metre = 2.
    field.local_metric = local
    support = Support(torch.tensor([0,1,3,4]),torch.tensor([.2,.8,.6,1.4]),torch.tensor([0,0,1,1]))
    explicit = torch.stack([field.member_logits(Support(support.indices[:2],support.weights[:2]),.7),
                            field.member_logits(Support(support.indices[2:],support.weights[2:]),.7)],1)
    expected = torch.logsumexp(explicit+torch.tensor([1/3,2/3]).log(),1)
    actual = ray_log_mixture(field,support,.7)
    assert torch.allclose(actual,expected,atol=5e-6,rtol=1e-6)
    split = Support(support.indices.repeat(2),support.weights.repeat(2)*.5,
                    torch.cat([support.components,support.components+2]))
    assert torch.allclose(actual,ray_log_mixture(field,split,.7),atol=5e-6,rtol=1e-6)
    actual.sum().backward()
    assert torch.isfinite(field.group.grad).all() and field.group.grad.abs().sum()>0
    assert field.local.codes.grad is None


@pytest.mark.parametrize('rule',['competitive_pooled','competitive_rays'])
def test_unsigned_inputs_are_exact_and_signed_training_matches_query(rule,tmp_path):
    parent = example(); field = copy.deepcopy(parent); field.negative_rule = rule
    pos = Support(torch.tensor([0,1]),torch.tensor([.3,.7]))
    packets = [QueryPacket('text',task='union',semantic=field.semantic.codes[0]),
               QueryPacket('image',semantic=field.semantic.codes[1],local=field.local.codes[1:3]),
               QueryPacket('point_2d',positive=pos),
               QueryPacket('point_3d',positive=Support(torch.tensor([0]),torch.ones(1)),exact_positive=True)]
    for packet in packets:
        assert torch.equal(parent.query(packet).membership_scores,field.query(packet).membership_scores)
    neg = Support(torch.tensor([4,5]),torch.tensor([.1,.9]),torch.arange(2))
    member = field.support_membership(pos,.5,neg)
    query = field.query(QueryPacket('point_2d',positive=pos,negative=neg,scale=.5))
    assert torch.equal(member,query.membership_scores)
    save_field(field,tmp_path/'field',{}); loaded,_=load_field(tmp_path/'field')
    assert loaded.negative_rule==rule and set(parent.state_dict())==set(loaded.state_dict())
    assert torch.equal(member,loaded.support_membership(pos,.5,neg))


def test_retaining_negative_rays_does_not_change_positive_adapter():
    op=torch.sparse_coo_tensor(torch.tensor([[0,0,1,1],[0,2,2,3]]),torch.tensor([.2,.8,.4,.6]),(2,12)).coalesce()
    a,_=scribble_packet(op,torch.tensor([1.,.25]),torch.tensor([.1,1.]))
    b,_=scribble_packet(op,torch.tensor([1.,.25]),torch.tensor([.1,1.]),negative_rays=True)
    assert torch.equal(a.positive.indices,b.positive.indices)
    assert torch.equal(a.positive.weights,b.positive.weights)
    assert b.positive.components is None and b.negative.components.tolist()==[0,0,1,1]


def test_missing_observed_negatives_keep_positive_exact_and_3d_constraints_remain_hard():
    field=example();field.negative_rule='competitive_rays';field.observed[11]=False
    pos=Support(torch.tensor([0]),torch.ones(1))
    original=field.support_membership(pos,.5)
    missing=Support(torch.tensor([11]),torch.ones(1),torch.tensor([0]))
    assert torch.equal(original,field.support_membership(pos,.5,missing))
    result=field.query(QueryPacket('point_3d',positive=pos,negative=Support(torch.tensor([4]),torch.ones(1)),exact_positive=True))
    assert result.membership_scores[0]==1 and result.membership_scores[4]==0
