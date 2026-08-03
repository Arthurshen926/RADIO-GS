import pytest
import torch

from radio_gs.querying.reliability_fusion import (
    geometric_consensus_unary,
    symmetric_bernoulli_product_of_experts,
)


def test_bernoulli_product_is_symmetric_and_neutral() -> None:
    values = torch.tensor([0.0, 0.1, 0.5, 0.8, 1.0])
    neutral = torch.full_like(values, 0.5)
    torch.testing.assert_close(
        symmetric_bernoulli_product_of_experts(values, neutral),
        values,
    )
    torch.testing.assert_close(
        symmetric_bernoulli_product_of_experts(values, neutral),
        symmetric_bernoulli_product_of_experts(neutral, values),
    )


def test_bernoulli_product_boundary_policy_is_explicit() -> None:
    left = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5])
    right = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.5])
    expected = torch.tensor([0.0, 1.0, 0.5, 0.5, 0.5])
    torch.testing.assert_close(
        symmetric_bernoulli_product_of_experts(left, right),
        expected,
    )


def test_bernoulli_product_is_monotone_in_each_expert() -> None:
    increasing = torch.linspace(0.0, 1.0, 101)
    fixed = torch.full_like(increasing, 0.73)
    first = symmetric_bernoulli_product_of_experts(increasing, fixed)
    second = symmetric_bernoulli_product_of_experts(fixed, increasing)
    assert bool((first[1:] >= first[:-1]).all())
    assert bool((second[1:] >= second[:-1]).all())


def test_geometric_consensus_unary_matches_logit_product() -> None:
    field = torch.tensor([-0.7, -0.2, 0.1, 0.8], dtype=torch.float64)
    direct = torch.tensor([0.4, -0.3, 0.7, -0.1], dtype=torch.float64)
    pooled = geometric_consensus_unary(
        field,
        direct,
        unary_temperature=0.1,
        chunk_size=2,
    )
    torch.testing.assert_close(pooled, field + direct, atol=1e-10, rtol=1e-10)


def test_geometric_consensus_preserves_neutral_rows_bitwise() -> None:
    field = torch.tensor([-0.75, 0.0, 0.625], dtype=torch.float32)
    pooled = geometric_consensus_unary(
        field,
        torch.zeros_like(field),
        unary_temperature=0.1,
        chunk_size=1,
    )
    assert torch.equal(pooled, field)


@pytest.mark.parametrize(
    "first,second,message",
    [
        (torch.tensor([float("nan")]), torch.tensor([0.5]), "NaN or infinity"),
        (torch.tensor([-0.1]), torch.tensor([0.5]), "first Bernoulli"),
        (torch.tensor([0.5]), torch.tensor([1.1]), "second Bernoulli"),
        (torch.tensor([0.5]), torch.tensor([0.5, 0.5]), "matching shapes"),
    ],
)
def test_bernoulli_product_fails_closed(first, second, message) -> None:
    with pytest.raises(ValueError, match=message):
        symmetric_bernoulli_product_of_experts(first, second)


def test_geometric_consensus_unary_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        geometric_consensus_unary(
            torch.zeros(2),
            torch.tensor([0.0, 1.1]),
            unary_temperature=0.1,
        )
    with pytest.raises(ValueError, match="positive"):
        geometric_consensus_unary(
            torch.zeros(2),
            torch.zeros(2),
            unary_temperature=0.0,
        )
