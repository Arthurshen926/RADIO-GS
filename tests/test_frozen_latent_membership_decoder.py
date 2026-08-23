import pytest
import torch

from radio_gs.models.frozen_latent_membership_decoder import (
    FrozenLatentMembershipDecoder,
)


def test_membership_decoder_is_pair_aligned_and_differentiable():
    model = FrozenLatentMembershipDecoder(latent_dim=8, query_dim=6, hidden_dim=4)
    latent = torch.randn(7, 8, requires_grad=True)
    query = torch.randn(7, 6, requires_grad=True)
    logits = model(latent, query)
    assert logits.shape == (7,)
    logits.sum().backward()
    assert latent.grad is not None
    assert query.grad is not None


def test_membership_decoder_rejects_unaligned_pairs():
    model = FrozenLatentMembershipDecoder(latent_dim=8, query_dim=6, hidden_dim=4)
    with pytest.raises(ValueError, match="aligned matrices"):
        model(torch.randn(3, 8), torch.randn(2, 6))
