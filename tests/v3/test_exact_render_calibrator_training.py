import torch

from radio_gs.v3.training.fit_exact_render_null_calibrated_posterior import (
    _exact_mask_loss,
)


def test_exact_mask_loss_rewards_positive_and_empty_calibration():
    truth = torch.tensor([1, 0], dtype=torch.bool)
    known = torch.ones(2, dtype=torch.bool)
    good, empty = _exact_mask_loss(torch.tensor([0.9, 0.1]), truth, known)
    bad, _ = _exact_mask_loss(torch.tensor([0.1, 0.9]), truth, known)
    assert not empty
    assert good < bad

    empty_truth = torch.zeros(2, dtype=torch.bool)
    low, is_empty = _exact_mask_loss(torch.tensor([0.1, 0.2]), empty_truth, known)
    high, _ = _exact_mask_loss(torch.tensor([0.8, 0.9]), empty_truth, known)
    assert is_empty
    assert low < high
