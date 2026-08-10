import torch

from radio_gs.scripts.materialize_lerf_o1_native_v3_coverage_deficit_fusion import (
    pair_observation_evidence,
)


def test_pair_observation_evidence_uses_native_then_v2_fallback() -> None:
    feature = {
        "native_feature_names": ["minimum_mean_observation_evidence"],
        "feature_names": ["minimum_core_observation_evidence"],
        "native_pair_features": torch.tensor([[0.8], [0.0]]),
        "v2_pair_features": torch.tensor([[0.4], [0.6]]),
        "native_pair_active_mask": torch.tensor([True, False]),
    }
    assert torch.equal(pair_observation_evidence(feature), torch.tensor([0.8, 0.6]))
