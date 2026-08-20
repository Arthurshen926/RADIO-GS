import torch

from radio_gs.querying.sam_siglip_object_posterior import (
    sam_siglip_object_posterior,
)


def _inputs():
    base = torch.tensor([[1.0], [0.8], [0.2], [0.1]])
    rows = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    props = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    weights = torch.ones(8)
    views = torch.tensor([0, 0, 1, 1])
    parents = torch.tensor([1, -1, 3, -1])
    descriptor = torch.tensor([[0.8], [0.8], [0.8], [0.8]])
    context = torch.zeros_like(descriptor)
    return base, rows, props, weights, views, parents, descriptor, context


def test_sam_siglip_object_posterior_selects_multiview_object_and_ascends():
    values = _inputs()
    posterior, stats = sam_siglip_object_posterior(
        *values,
        minimum_object_views=2,
        minimum_descriptor_score=0.5,
    )
    assert torch.equal(posterior[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert stats["fallback_query_count"] == 0
    assert stats["parent_ascent_steps"] == [2]


def test_sam_siglip_object_posterior_falls_back_bitwise_with_one_view():
    values = list(_inputs())
    values[6] = torch.tensor([[0.8], [0.8], [0.2], [0.2]])
    posterior, stats = sam_siglip_object_posterior(
        *values,
        minimum_object_views=3,
        minimum_descriptor_score=0.5,
    )
    assert torch.equal(posterior, values[0])
    assert stats["fallback_query_count"] == 1


def test_query_listwise_descriptor_gate_is_invariant_to_low_query_offset():
    values = list(_inputs())
    values[6] = torch.tensor([[0.40], [0.40], [0.40], [0.40]])
    absolute, absolute_stats = sam_siglip_object_posterior(
        *values,
        minimum_object_views=2,
        minimum_descriptor_score=0.5,
    )
    listwise, listwise_stats = sam_siglip_object_posterior(
        *values,
        minimum_object_views=2,
        minimum_descriptor_score=0.5,
        descriptor_gate="query_listwise",
        descriptor_listwise_margin=0.12,
    )
    assert torch.equal(absolute, values[0])
    assert absolute_stats["fallback_query_count"] == 1
    assert torch.equal(listwise[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert listwise_stats["fallback_query_count"] == 0


def test_latent_proposal_marginal_keeps_explicit_null_and_is_probabilistic():
    values = _inputs()
    posterior, stats = sam_siglip_object_posterior(
        *values,
        minimum_object_views=2,
        minimum_descriptor_score=0.5,
        association_mode="latent_proposal_marginal",
        require_field_peak_anchor=False,
    )
    assert bool(((posterior >= 0) & (posterior <= 1)).all())
    assert 0.0 < stats["null_probability"][0] < 1.0
    assert stats["fallback_query_count"] == 0


def test_latent_proposal_marginal_falls_back_without_cross_view_evidence():
    values = list(_inputs())
    values[4] = torch.zeros(4, dtype=torch.long)
    posterior, stats = sam_siglip_object_posterior(
        *values,
        minimum_descriptor_score=0.5,
        association_mode="latent_proposal_marginal",
        require_field_peak_anchor=False,
    )
    assert torch.equal(posterior, values[0])
    assert stats["fallback_query_count"] == 1
