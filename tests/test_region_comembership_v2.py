from __future__ import annotations

import pytest
import torch

from radio_gs.models.region_comembership_v1 import (
    PAIR_FEATURE_NAMES as V1_PAIR_FEATURE_NAMES,
)
from radio_gs.models.region_comembership_v2 import (
    CAPABILITY_PAIR_FEATURE_NAMES,
    HIDDEN_DIMENSIONS,
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV2,
)


def test_v2_feature_order_appends_the_six_registered_capability_channels() -> None:
    assert PAIR_FEATURE_NAMES[:15] == V1_PAIR_FEATURE_NAMES
    assert PAIR_FEATURE_NAMES[15:] == CAPABILITY_PAIR_FEATURE_NAMES
    assert len(PAIR_FEATURE_NAMES) == 21
    assert HIDDEN_DIMENSIONS == (64, 32)


def test_v2_epoch_zero_is_exactly_half_and_two_steps_train_hidden_layers() -> None:
    torch.manual_seed(0)
    model = RegionCoMembershipV2(torch.zeros(21), torch.ones(21))
    features = torch.randn(12, 21)
    target = torch.tensor([0.0, 1.0] * 6)
    assert torch.equal(model(features), torch.zeros(12))
    assert torch.equal(model.probability(features), torch.full((12,), 0.5))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(features), target
        )
        loss.backward()
        optimizer.step()
    assert model.network[0].weight.grad is not None
    assert float(model.network[0].weight.grad.abs().sum()) > 0
    assert not torch.equal(model(features), torch.zeros(12))


@pytest.mark.parametrize(
    "median,scale",
    [
        (torch.zeros(20), torch.ones(20)),
        (torch.zeros(21), torch.zeros(21)),
        (torch.full((21,), torch.nan), torch.ones(21)),
    ],
)
def test_v2_rejects_invalid_normalization(
    median: torch.Tensor, scale: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="normalization"):
        RegionCoMembershipV2(median, scale)


def test_v2_rejects_wrong_pair_feature_shape_and_dtype() -> None:
    model = RegionCoMembershipV2(torch.zeros(21), torch.ones(21))
    with pytest.raises(ValueError, match=r"\[P,21\]"):
        model(torch.zeros(2, 20))
    with pytest.raises(ValueError, match=r"\[P,21\]"):
        model(torch.zeros(2, 21, dtype=torch.int64))
