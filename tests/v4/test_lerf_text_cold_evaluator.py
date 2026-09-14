import pytest
import torch

from radio_gs.v4.evaluation.lerf_text_cold_evaluator import (
    compose_object_queries,
    prototype_cosine_set_posterior,
)


def test_query_composition_uses_canonical_sum_not_max():
    membership = torch.tensor([[0.4, 0.3], [0.8, 0.1]])
    token_probability = torch.tensor([[0.6, 0.3]])
    posterior, audit = compose_object_queries(membership, token_probability)
    # Dense rows are completed to a simplex with explicit unknown mass, so the
    # first element is 0.4*.6 + 0.3*.3 = .33 rather than max(.24, .09).
    assert posterior[:, 0].tolist() == pytest.approx([0.33, 0.51])
    assert audit["selection_mode"] == "multi_instance"
    assert audit["composition"].endswith("mixture_sum")


def test_top3_is_explicit_diagnostic_not_primary_query_contract():
    membership = torch.eye(4)
    probability = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    full, full_audit = compose_object_queries(membership, probability)
    bounded, bounded_audit = compose_object_queries(
        membership, probability, maximum_query_tokens=3
    )
    assert full_audit["query_token_capacity"] is None
    assert bounded_audit["query_token_capacity"] == 3
    assert full[0, 0] == pytest.approx(0.1)
    assert bounded[0, 0] == 0


def test_cosine_set_posterior_has_soft_bounded_total_mass():
    posterior = prototype_cosine_set_posterior(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
        torch.tensor([0, 0, 1]),
        torch.tensor([[1.0, 0.0]]),
        num_tokens=2,
        temperature=0.1,
        set_mass=1.0,
    )
    assert posterior.shape == (1, 2)
    assert posterior.sum() <= 1.0
    assert posterior[0, 0] > posterior[0, 1]
