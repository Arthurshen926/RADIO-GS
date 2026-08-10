from __future__ import annotations

import pytest
import torch

from radio_gs.models.region_comembership_native_v3 import (
    NATIVE_PAIR_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    V2_PAIR_FEATURE_NAMES,
    RegionCoMembershipNativeV3,
)


def test_native_v3_is_opt_in_30d_and_preserves_exact_epoch_zero() -> None:
    assert len(V2_PAIR_FEATURE_NAMES) == 21
    assert len(NATIVE_PAIR_FEATURE_NAMES) == 9
    assert len(PAIR_FEATURE_NAMES) == 30
    model = RegionCoMembershipNativeV3(torch.zeros(30), torch.ones(30))
    features = torch.randn(17, 30)
    probability = model.probability(features)
    assert torch.equal(probability, torch.full((17,), 0.5))


def test_native_v3_rejects_legacy_width_and_invalid_scale() -> None:
    model = RegionCoMembershipNativeV3(torch.zeros(30), torch.ones(30))
    with pytest.raises(ValueError):
        model(torch.randn(4, 21))
    with pytest.raises(ValueError):
        RegionCoMembershipNativeV3(torch.zeros(30), torch.zeros(30))
