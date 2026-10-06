import torch
import pytest
from radio_gs.geometry.depth_opacity import depth_distribution_loss


def test_front_back_mixture_cannot_hide_behind_correct_mean_depth():
    # Equal weights at 1m and 3m have correct mean 2m but wrong surfaces.
    target = torch.tensor([[2.]])
    valid = torch.ones_like(target, dtype=torch.bool)
    alpha = torch.ones_like(target)
    exact, _ = depth_distribution_loss(torch.tensor([[[2., 4.]]]), alpha, target, valid)
    mixture, _ = depth_distribution_loss(torch.tensor([[[2., 5.]]]), alpha, target, valid)
    assert mixture > exact+1


def test_invalid_pixels_do_not_supervise_opacity_or_depth():
    moments = torch.tensor([[[2., 4.], [1., 50.]]], requires_grad=True)
    alpha = torch.tensor([[1., .2]], requires_grad=True)
    loss, _ = depth_distribution_loss(moments, alpha, torch.tensor([[2., float('nan')]]), torch.ones(1, 2, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(moments.grad[0, 1], torch.zeros(2))
    assert alpha.grad[0, 1] == 0


def test_empty_reference_fails_closed():
    with pytest.raises(ValueError, match='no valid'):
        depth_distribution_loss(torch.zeros(1, 1, 2), torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))
