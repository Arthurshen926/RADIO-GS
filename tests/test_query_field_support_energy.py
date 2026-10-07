import copy

import pytest
import torch

from radio_gs.query_field.adapters import raster_support, scribble_packet
from radio_gs.query_field.contracts import QueryPacket, Support
from radio_gs.query_field.storage import save_field, load_field
from test_query_preserving_field import example


@pytest.mark.parametrize('native', ['none', 'native_cosine'])
def test_expected_energy_equals_weighted_singleton_logits_with_spatial_variance(native):
    field = example()
    field.spatial_metric = 'scale_normalized'
    field.world_units_per_metre = 2.
    field.local_metric = native
    indices, weights = torch.tensor([0, 2, 4]), torch.tensor([.2, .3, .5])
    expected = sum(field.member_logits(Support(i[None], torch.ones(1)), .7)*w
                   for i, w in zip(indices, weights))
    old = field.member_logits(Support(indices, weights), .7)
    field.support_aggregation = 'expected_energy'
    actual = field.member_logits(Support(indices, weights), .7)
    assert torch.allclose(actual, expected, atol=4e-6, rtol=1e-6)
    assert not torch.allclose(old, actual, atol=1e-4)
    actual.sum().backward()
    assert torch.isfinite(field.group.grad).all()
    assert field.group.grad[indices].abs().sum() > 0
    assert field.semantic.codes.grad is None and field.local.codes.grad is None


def test_ray_constraints_do_not_cancel_or_strengthen_when_duplicated():
    field = example()
    field.support_aggregation = 'expected_energy'
    negative = Support(torch.tensor([2, 3, 4, 5]), torch.tensor([.3, .7, .4, .6]), torch.tensor([0, 0, 1, 1]))
    pos = torch.ones(len(field.xyz))
    expected = 1-torch.stack([field.member_logits(Support(negative.indices[a:b], negative.weights[a:b]), .5).sigmoid()
                             for a, b in [(0, 2), (2, 4)]]).max(0).values
    actual = field.exclude_membership(pos, negative, .5)
    assert torch.allclose(actual, expected, atol=2e-7)
    repeated = Support(negative.indices.repeat(2), negative.weights.repeat(2),
                       torch.cat([negative.components, negative.components+2]))
    assert torch.equal(actual, field.exclude_membership(pos, repeated, .5))
    one = field.exclude_membership(pos, Support(negative.indices[:2], negative.weights[:2]), .5)
    assert bool((actual <= one+2e-7).all())
    assert not torch.allclose(actual, field.exclude_membership(pos, Support(negative.indices, negative.weights), .5))


def test_structured_adapter_retains_ray_identity_and_old_pool_equivalence():
    op = torch.sparse_coo_tensor(torch.tensor([[0, 0, 1, 1], [0, 2, 2, 3]]),
                                 torch.tensor([.2, .8, .4, .6]), (3, 12)).coalesce()
    evidence = torch.tensor([.5, 1., .5])
    pooled, coverage = raster_support(op, evidence)
    rays, ray_coverage = raster_support(op, evidence, preserve_rays=True)
    assert coverage == ray_coverage == .75
    assert rays.components.tolist() == [0, 0, 1, 1]
    combined = torch.bincount(rays.indices, weights=rays.weights, minlength=12)
    assert torch.allclose(combined[pooled.indices], pooled.weights)
    packet, receipt = scribble_packet(op, evidence, preserve_rays=True)
    assert receipt['positive_gaussians'] == 3 and receipt['separate_ray_constraints']
    packet.validate(12, 6, 4)
    with pytest.raises(ValueError, match='components'):
        Support(torch.tensor([0]), torch.ones(1), torch.tensor([-1])).validate(12)


def test_singleton_queries_remain_exact_and_expected_export_is_self_contained(tmp_path):
    legacy = example()
    field = copy.deepcopy(legacy)
    field.support_aggregation = 'expected_energy'
    packets = [QueryPacket('text', semantic=field.semantic.decode(torch.tensor([0]))[0]),
               QueryPacket('point_3d', positive=Support(torch.tensor([0]), torch.ones(1)), exact_positive=True)]
    for packet in packets:
        assert torch.equal(legacy.query(packet).membership_scores, field.query(packet).membership_scores)
    manifest = save_field(field, tmp_path/'field', {})
    loaded, _ = load_field(tmp_path/'field')
    assert loaded.support_aggregation == 'expected_energy'
    assert manifest['external_scene_dependencies'] == []
    assert set(legacy.state_dict()) == set(field.state_dict())
    packet = QueryPacket('point_2d', positive=Support(torch.tensor([0, 1]), torch.tensor([.4, .6])),
                        negative=Support(torch.tensor([4, 5]), torch.ones(2), torch.arange(2)))
    assert torch.equal(field.query(packet).membership_scores, loaded.query(packet).membership_scores)


def test_zero_observed_negative_ray_is_ignored_without_inventing_evidence():
    field = example()
    field.support_aggregation = 'expected_energy'
    field.observed[5] = False
    support = Support(torch.tensor([4, 5]), torch.ones(2), torch.tensor([0, 1]))
    actual = field.exclude_membership(torch.ones(len(field.xyz)), support, .5)
    expected = field.exclude_membership(torch.ones(len(field.xyz)), Support(torch.tensor([4]), torch.ones(1)), .5)
    assert torch.allclose(actual, expected, atol=2e-7)
