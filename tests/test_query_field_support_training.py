import torch
import pytest

from radio_gs.query_field.adapters import raster_support
from radio_gs.query_field.contracts import QueryPacket, Support
from radio_gs.query_field.support_training import TrainingRays, episode_support, sample_pixels
from test_query_preserving_field import example


def source_view():
    op = torch.sparse_coo_tensor(torch.tensor([[0, 0, 1, 2, 3, 4, 5], [0, 1, 1, 2, 3, 4, 5]]),
                                 torch.tensor([.2, .8, 1., 1., 1., 1., 1.]), (6, 12)).coalesce()
    return dict(operator=op, best_element=torch.tensor([1, 1, 2, 3, 4, 5]), training_rays=TrainingRays(op))


def test_source_ray_gather_matches_deployed_pool_and_preserves_uncertainty():
    view = source_view()
    expected, _ = raster_support(view['operator'], torch.tensor([1., 0., 1., 0., 0., 0.]))
    actual = view['training_rays'].support([0, 2], 'cpu')
    assert torch.equal(expected.indices, actual.indices)
    assert torch.equal(expected.weights, actual.weights)
    assert actual.indices.tolist() == [0, 1, 2]  # Non-dominant row zero survives.


def test_matched_support_draws_do_not_change_original_episode_rng():
    view = source_view()
    states, records = [], []
    for policy in ['mixed_unsigned', 'mixed_signed']:
        torch.manual_seed(131)
        generator = torch.Generator().manual_seed(19)
        trace = []
        for step in range(16):
            first = int(torch.randint(3, (1,)))
            positive, negative, kind = episode_support(policy, step, view, torch.arange(3), first,
                                                       torch.tensor([3, 4, 5]), generator, 'cpu')
            trace.append((first, positive.indices.clone(), positive.weights.clone(), negative, kind))
        states.append(torch.random.get_rng_state())
        records.append(trace)
    assert torch.equal(*states)
    for step, (a, b) in enumerate(zip(*records)):
        assert a[0] == b[0] and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        assert a[3] is None
        assert (b[3] is not None) == (step % 4 == 3)
        if b[3] is not None:
            assert set(b[3].indices.tolist()) <= {3, 4, 5}
    assert sample_pixels(torch.tensor([4, 7]), 4, torch.Generator().manual_seed(3)).tolist() == [4, 7]


def test_missing_negatives_keep_episode_unsigned_without_fabricated_support():
    view = source_view()
    positive, negative, kind = episode_support('mixed_signed', 3, view, torch.arange(3), 0,
                                               torch.empty(0, dtype=torch.long), torch.Generator(), 'cpu')
    assert negative is None and kind == 'pooled_missing_negative'
    assert len(positive.indices) == 3
    with pytest.raises(ValueError):
        episode_support('unsupported', 0, view, torch.arange(3), 0, torch.arange(3), torch.Generator(), 'cpu')


def test_training_signed_member_equals_deployment_and_backpropagates_both_signs():
    field = example()
    positive = Support(torch.tensor([0, 1]), torch.tensor([.2, .8]))
    negative = Support(torch.tensor([3, 4]), torch.tensor([.7, .3]))
    member = field.support_membership(positive, .5, negative)
    result = field.query(QueryPacket('point_2d', positive=positive, negative=negative, scale=.5))
    assert torch.equal(member.masked_fill(~field.observed, 0), result.membership_scores)
    expected = field.member_logits(positive, .5).sigmoid()*(1-field.member_logits(negative, .5).sigmoid())
    assert torch.equal(member, expected)
    member[[0, 3, 5]].sum().backward()
    assert torch.isfinite(field.group.grad).all() and field.group.grad[negative.indices].abs().sum() > 0
    assert field.semantic.codes.grad is None and field.local.codes.grad is None
