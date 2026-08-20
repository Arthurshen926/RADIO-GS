import numpy as np
import pytest
import torch

from radio_gs.querying import (
    QueryAbstention,
    deterministic_visible_signed_points,
    marginalize_synchronous_multiview_candidates,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def test_candidate_marginal_is_convex_and_order_invariant_bitwise():
    fields = torch.tensor(
        [
            [[0.0, 0.2], [1.0, 0.6]],
            [[0.8, 1.0], [0.4, 0.0]],
        ],
        dtype=torch.float32,
    )
    precision = torch.log(torch.tensor([[1.0, 3.0], [2.0, 2.0]]))
    logits = torch.log(torch.tensor([3.0, 1.0]))
    first = marginalize_synchronous_multiview_candidates(
        fields,
        precision,
        logits,
        candidate_digests=[_digest(2), _digest(1)],
        view_digests=[_digest(4), _digest(3)],
        expected_candidates=2,
    )
    second = marginalize_synchronous_multiview_candidates(
        fields[[1, 0]][:, [1, 0]],
        precision[[1, 0]][:, [1, 0]],
        logits[[1, 0]],
        candidate_digests=[_digest(1), _digest(2)],
        view_digests=[_digest(3), _digest(4)],
        expected_candidates=2,
    )
    assert torch.equal(first.probability, second.probability)
    assert torch.equal(first.candidate_field, second.candidate_field)
    assert 0.0 <= first.probability.min() <= first.probability.max() <= 1.0
    assert torch.allclose(first.candidate_probability.sum(), torch.tensor(1.0))
    assert torch.allclose(first.view_probability.sum(dim=1), torch.ones(2))


def test_incomplete_k_abstains_instead_of_falling_back():
    with pytest.raises(QueryAbstention, match="incomplete"):
        marginalize_synchronous_multiview_candidates(
            torch.zeros((1, 2, 3)),
            torch.zeros((1, 2)),
            torch.zeros(1),
            candidate_digests=[_digest(1)],
            view_digests=[_digest(2), _digest(3)],
            expected_candidates=10,
        )


def test_signed_points_are_stable_and_have_xy_label_order():
    posterior = np.array(
        [[0.1, 0.9, 0.8], [0.2, 0.7, 0.3], [0.6, 0.4, 0.95]],
        dtype=np.float32,
    )
    visible = np.ones_like(posterior, dtype=bool)
    first = deterministic_visible_signed_points(
        posterior,
        visible,
        candidate_digest=_digest(11),
        view_digest=_digest(22),
        points_per_sign=3,
    )
    second = deterministic_visible_signed_points(
        posterior.copy(),
        visible.copy(),
        candidate_digest=_digest(11),
        view_digest=_digest(22),
        points_per_sign=3,
    )
    assert np.array_equal(first[0], second[0])
    assert first[1].tolist() == [1, 1, 1, 0, 0, 0]
    for (x, y), label in zip(*first):
        assert bool(posterior[int(y), int(x)] >= 0.5) == bool(label)


def test_signed_points_abstain_when_a_sign_is_not_visible():
    with pytest.raises(QueryAbstention, match="signed visible support"):
        deterministic_visible_signed_points(
            np.ones((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=bool),
            candidate_digest=_digest(1),
            view_digest=_digest(2),
        )
