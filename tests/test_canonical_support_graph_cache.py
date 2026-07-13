import torch

from radio_gs.scripts.build_canonical_support_graph import (
    deterministic_feature_hash,
)


def test_deterministic_feature_hash_is_repeatable_and_normalized():
    features = torch.arange(60, dtype=torch.float32).reshape(6, 10) + 1.0
    first = deterministic_feature_hash(features, 7, batch_size=2)
    second = deterministic_feature_hash(features, 7, batch_size=4)

    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(6), atol=1e-6)
