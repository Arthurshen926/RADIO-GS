import pytest
import torch

from radio_gs.evaluation.view_consistency import (
    consistency_from_sums,
    merge_training_partials,
    pearson_spearman,
)


def _partial(count, weight, cosine, cosine_square):
    return {
        "observation_count": torch.tensor(count),
        "weight_sum": torch.tensor(weight),
        "weight_square_sum": torch.tensor(weight),
        "cosine_sum": torch.tensor(cosine),
        "cosine_square_sum": torch.tensor(cosine_square),
        "weighted_cosine_sum": torch.tensor(cosine),
        "weighted_cosine_square_sum": torch.tensor(cosine_square),
    }


def test_merge_and_consistency_recover_mean_and_variance():
    first = _partial([1, 1], [1.0, 1.0], [0.8, 0.4], [0.64, 0.16])
    second = _partial([1, 1], [1.0, 1.0], [0.6, 0.8], [0.36, 0.64])

    result = consistency_from_sums(merge_training_partials([first, second]))

    torch.testing.assert_close(result["mean_cosine"], torch.tensor([0.7, 0.6]))
    torch.testing.assert_close(result["cosine_variance"], torch.tensor([0.01, 0.04]))
    torch.testing.assert_close(result["view_disagreement"], torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(result["effective_views"], torch.tensor([2.0, 2.0]))


def test_merge_rejects_row_mismatch():
    one = _partial([1], [1.0], [0.5], [0.25])
    two = _partial([1, 1], [1.0, 1.0], [0.5, 0.5], [0.25, 0.25])

    with pytest.raises(ValueError, match="row counts differ"):
        merge_training_partials([one, two])


def test_correlations_track_monotonic_disagreement_and_error():
    result = pearson_spearman(torch.arange(10), torch.arange(10).square())

    assert result["samples"] == 10
    assert result["pearson"] > 0.9
    assert result["spearman"] == pytest.approx(1.0)

