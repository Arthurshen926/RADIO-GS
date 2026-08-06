import numpy as np
import pytest

from radio_gs.querying.sam3_reference_completion import (
    aggregate_completed_positive,
    deterministic_positive_points,
    entropy_reliability_soft_observation,
    probability_preserving_entropy_observation,
)


def test_deterministic_positive_points_are_unique_and_repeatable():
    mask = np.zeros((6, 8), dtype=bool)
    mask.reshape(-1)[[1, 2, 5, 9, 12, 18, 21, 30, 33, 41]] = True
    first = deterministic_positive_points(mask, count=6)
    second = deterministic_positive_points(mask.copy(), count=6)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert len({tuple(point) for point in first.tolist()}) == 6
    assert all(mask[int(y), int(x)] for x, y in first)


def test_deterministic_positive_points_fail_closed():
    with pytest.raises(ValueError, match="2D"):
        deterministic_positive_points(np.ones((1, 2, 3)), count=1)
    with pytest.raises(ValueError, match="required"):
        deterministic_positive_points(np.eye(2, dtype=bool), count=3)


def test_completed_positive_applies_signed_scribble_authority():
    trials = np.asarray(
        [
            [[0, 1, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 1]],
        ],
        dtype=bool,
    )
    positive = np.zeros((2, 3), dtype=bool)
    positive[0, 0] = True
    negative = np.zeros((2, 3), dtype=bool)
    negative[0, 2] = True
    aggregate, completed = aggregate_completed_positive(
        trials, positive, negative, threshold=0.5
    )
    np.testing.assert_allclose(
        aggregate, np.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    )
    np.testing.assert_array_equal(
        completed,
        np.asarray([[1, 1, 0], [0, 1, 1]], dtype=bool),
    )


def test_entropy_reliability_has_registered_boundaries_and_symmetry():
    q = np.asarray([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    empty = np.zeros_like(q, dtype=bool)
    reliability, observation = entropy_reliability_soft_observation(q, empty, empty)
    np.testing.assert_allclose(reliability[:, [0, 4]], 1.0, rtol=0, atol=0)
    np.testing.assert_allclose(reliability[:, 2], 0.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(reliability[:, 1], reliability[:, 3], rtol=0, atol=1e-7)
    np.testing.assert_allclose(observation, q * reliability, rtol=0, atol=1e-7)
    assert bool(((reliability >= 0) & (reliability <= 1)).all())


def test_entropy_soft_observation_applies_raw_scribble_authority():
    q = np.asarray([[0.5, 1.0], [0.0, 0.9]], dtype=np.float32)
    positive = np.asarray([[1, 0], [0, 0]], dtype=bool)
    negative = np.asarray([[0, 1], [0, 0]], dtype=bool)
    _, observation = entropy_reliability_soft_observation(q, positive, negative)
    assert observation[0, 0] == 1.0
    assert observation[0, 1] == 0.0
    assert observation[1, 0] == 0.0
    assert 0.0 < observation[1, 1] < 0.9


def test_entropy_soft_observation_fails_closed_on_invalid_input():
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="finite"):
        entropy_reliability_soft_observation(
            np.asarray([[0.0, 1.1], [0.0, 0.0]]), mask, mask
        )
    overlap = mask.copy()
    overlap[0, 0] = True
    with pytest.raises(ValueError, match="overlap"):
        entropy_reliability_soft_observation(np.zeros((2, 2)), overlap, overlap)


def test_probability_preserving_observation_separates_q_from_confidence():
    q = np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32)
    empty = np.zeros_like(q, dtype=bool)

    probability, reliability = probability_preserving_entropy_observation(
        q, empty, empty
    )

    np.testing.assert_array_equal(probability, q)
    assert reliability[0, 1] == pytest.approx(0.0, abs=1e-7)
    assert probability[0, 1] == 0.5
    assert reliability[0, 0] == pytest.approx(reliability[0, 2], abs=1e-7)
    # The old q*c representation would incorrectly turn q=0.5 into zero.
    assert probability[0, 1] != q[0, 1] * reliability[0, 1]


def test_probability_preserving_observation_applies_signed_authority_to_both_terms():
    q = np.asarray([[0.5, 1.0]], dtype=np.float32)
    positive = np.asarray([[1, 0]], dtype=bool)
    negative = np.asarray([[0, 1]], dtype=bool)

    probability, reliability = probability_preserving_entropy_observation(
        q, positive, negative
    )

    np.testing.assert_array_equal(probability, np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(reliability, np.ones((1, 2), dtype=np.float32))
