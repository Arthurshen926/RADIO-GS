import torch

from radio_gs.models.frozen_latent_capability_decoder import (
    FrozenLatentCapabilityDecoder,
)
from radio_gs.scripts.train_frozen_latent_direct_capability_decoder import (
    cosine_metrics,
)


def test_zero_initialized_decoder_is_exact_normalized_baseline() -> None:
    model = FrozenLatentCapabilityDecoder(
        latent_dim=4, hidden_dim=3, capability_dim=5
    )
    latent = torch.randn(6, 4)
    baseline = torch.randn(6, 5)
    result = model(latent, baseline)
    expected = torch.nn.functional.normalize(baseline, dim=-1)
    assert torch.equal(result, expected)


def test_cosine_metrics_report_identical_features() -> None:
    features = torch.randn(16, 7)
    metrics = cosine_metrics(features, features)
    assert abs(metrics["mean_cosine"] - 1.0) < 1e-6
    assert abs(metrics["p05_cosine"] - 1.0) < 1e-6

