import torch

from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_alignment_loss


class ScaleAdaptor(torch.nn.Module):
    def forward(self, x):
        return x * 2.0


def test_compute_radio_adaptor_alignment_loss_returns_zero_without_adaptors():
    decoded = torch.randn(1, 4, 2, 2)
    target = decoded.clone()

    loss, stats = compute_radio_adaptor_alignment_loss(decoded, target, {})

    assert loss.item() == 0.0
    assert stats == {}


def test_compute_radio_adaptor_alignment_loss_matches_identical_features():
    decoded = torch.randn(1, 4, 2, 2)
    target = decoded.clone()

    loss, stats = compute_radio_adaptor_alignment_loss(
        decoded,
        target,
        {"sam3": ScaleAdaptor()},
    )

    assert loss.item() < 1e-6
    assert "sam3" in stats
