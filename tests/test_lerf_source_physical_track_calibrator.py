import torch

from radio_gs.scripts.train_lerf_source_physical_track_calibrator import (
    balanced_log_score,
    rank_auc,
)


def test_rank_auc_is_exact_for_separated_scores() -> None:
    assert rank_auc(torch.tensor([-2.0, -1.0, 1.0, 2.0]), torch.tensor([0, 0, 1, 1])) == 1.0


def test_balanced_log_score_rewards_correct_logits() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    assert balanced_log_score(torch.tensor([-2.0, -1.0, 1.0, 2.0]), labels) < balanced_log_score(torch.zeros(4), labels)
