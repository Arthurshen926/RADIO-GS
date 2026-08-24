import torch

from radio_gs.models.frozen_latent_split_score_decoder import (
    FrozenReliabilityEligibilityGate,
    FrozenLatentSplitScoreDecoder,
)


def test_zero_initialization_replays_each_score_block() -> None:
    model = FrozenLatentSplitScoreDecoder(latent_dim=4, hidden_dim=3, split_dims=(3, 2))
    latent = torch.randn(7, 4)
    baseline = torch.randn(7, 5)
    expected = torch.cat(
        [torch.nn.functional.normalize(value, dim=-1) for value in baseline.split((3, 2), dim=-1)],
        dim=-1,
    )
    assert torch.equal(model(latent, baseline), expected)


def test_eligibility_gate_has_three_outputs() -> None:
    model = FrozenReliabilityEligibilityGate(
        latent_dim=4, reliability_dim=2, score_dim=5, hidden_dim=3
    )
    assert model(torch.randn(7, 4), torch.randn(7, 2), torch.randn(7, 5)).shape == (7, 3)
