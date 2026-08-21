import torch

from radio_gs.scripts.train_lerf_source_physical_track_calibrator_nested import (
    balanced_probability_log_score,
    jeffreys_probability,
)


def test_jeffreys_shrinkage_moves_probability_toward_half() -> None:
    raw = jeffreys_probability(torch.tensor([-4.0, 4.0]), 1.0, 0.0)
    shrunk = jeffreys_probability(torch.tensor([-4.0, 4.0]), 1.0, 1.0)
    assert torch.all(torch.abs(shrunk - 0.5) < torch.abs(raw - 0.5))


def test_probability_log_score_prefers_correct_probabilities() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    assert balanced_probability_log_score(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels) < balanced_probability_log_score(torch.full((4,), 0.5), labels)
