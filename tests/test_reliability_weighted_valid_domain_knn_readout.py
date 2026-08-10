from __future__ import annotations

import pytest
import torch

from radio_gs.querying import reliability_weighted_valid_domain_knn_readout as v2
from radio_gs.querying import valid_domain_knn_readout as v1


def test_uniform_policy_is_bitwise_v1() -> None:
    generator = torch.Generator().manual_seed(20260809)
    scores = torch.rand(17, 5, generator=generator)
    xyz = torch.rand(17, 3, generator=generator)
    valid = torch.tensor([True, False, True, True, False] + [True] * 12)
    reliability = torch.rand(17, generator=generator)
    reliability[~valid] = 0
    expected = v1.valid_domain_knn_smoothed_scores(
        scores, xyz, k=7, chunk_size=4, valid_mask=valid
    )
    actual = v2.reliability_weighted_valid_domain_knn_smoothed_scores(
        scores,
        xyz,
        reliability,
        policy_id="uniform",
        k=7,
        chunk_size=4,
        valid_mask=valid,
    )
    assert torch.equal(actual, expected)


def test_reliability_precision_prefers_trusted_neighbor() -> None:
    scores = torch.tensor([[0.0], [1.0], [0.0]])
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    reliability = torch.tensor([0.0, 1.0, 0.0])
    actual = v2.reliability_weighted_valid_domain_knn_smoothed_scores(
        scores,
        xyz,
        reliability,
        policy_id="reliability_precision",
        k=3,
    )
    assert actual[:, 0].tolist() == pytest.approx([0.5, 1.0, 0.5])


def test_all_zero_reliability_falls_back_to_spatial_kernel() -> None:
    distances = torch.tensor([[0.0, 0.5, 1.0]])
    reliability = torch.zeros_like(distances)
    actual = v2.normalized_neighbor_weights(
        distances,
        reliability,
        policy_id="gaussian_reliability_precision",
    )
    expected = v2.normalized_neighbor_weights(
        distances, reliability, policy_id="gaussian"
    )
    assert torch.equal(actual, expected)
    assert float(actual.sum()) == pytest.approx(1.0)


def test_leave_self_out_retrieves_k_other_neighbors() -> None:
    scores = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    xyz = torch.arange(4, dtype=torch.float32)[:, None].repeat(1, 3)
    valid = torch.ones(4, dtype=torch.bool)
    reliability = torch.ones(4)
    actual = v2.reliability_weighted_valid_domain_knn_smoothed_scores(
        scores,
        xyz,
        reliability,
        policy_id="uniform_leave_self_out",
        k=2,
        valid_mask=valid,
    )
    # Centre zero retrieves self plus two others, removes self, then averages
    # the two independent neighbours.  The raw outer self keeps total 0.5.
    assert float(actual[0, 0]) == pytest.approx(0.5)
    assert float(actual[1, 0]) == pytest.approx(0.25)


def test_source_view_reliability_reuses_existing_formula() -> None:
    actual = v2.source_view_reliability(
        torch.tensor([0, 1, 2, 4]),
        torch.tensor([0.0, 1.0, 0.75, 1.0]),
    )
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "broken",
    ["unknown_policy", "invalid_reliability", "invalid_xyz", "invalid_mask"],
)
def test_weighted_readout_fails_closed(broken: str) -> None:
    scores = torch.zeros(3, 1)
    xyz = torch.zeros(3, 3)
    reliability = torch.zeros(3)
    valid = torch.ones(3, dtype=torch.bool)
    policy = "uniform"
    if broken == "unknown_policy":
        policy = "scene_tuned"
    elif broken == "invalid_reliability":
        reliability[0] = 1.1
    elif broken == "invalid_xyz":
        xyz[0, 0] = torch.nan
    else:
        valid[0] = False
        reliability[0] = 0.1
    with pytest.raises(ValueError):
        v2.reliability_weighted_valid_domain_knn_smoothed_scores(
            scores,
            xyz,
            reliability,
            policy_id=policy,
            valid_mask=valid,
        )


def test_multiscale_output_is_bounded_and_invalid_zero() -> None:
    positive = torch.tensor(
        [
            [[0.8], [0.7], [0.6]],
            [[0.0], [0.0], [0.0]],
            [[0.2], [0.3], [0.4]],
        ]
    )
    negative = torch.zeros(3, 3, 4)
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    valid = torch.tensor([True, False, True])
    reliability = torch.tensor([1.0, 0.0, 0.2])
    result = v2.reliability_weighted_valid_domain_multiscale_readout(
        positive,
        negative,
        xyz,
        valid,
        reliability,
        policy_id="gaussian_reliability_precision",
        k=2,
    )
    assert result.scores.shape == (3, 1)
    assert torch.isfinite(result.scores).all()
    assert bool((result.scores >= 0).all())
    assert bool((result.scores <= 1).all())
    assert torch.equal(result.scores[~valid], torch.zeros(1, 1))
