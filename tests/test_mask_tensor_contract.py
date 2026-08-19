import pytest
import torch

from radio_gs.models.mask_tensor_contract import (
    MASK_TENSOR_LOGIT,
    MASK_TENSOR_PROBABILITY,
    mask_tensor_to_binary,
    mask_tensor_to_probability,
)


def test_probability_semantics_is_identity_not_second_sigmoid() -> None:
    value = torch.tensor([0.0, 0.25, 1.0])
    observed = mask_tensor_to_probability(
        value,
        semantics=MASK_TENSOR_PROBABILITY,
    )
    assert torch.equal(observed, value)


def test_logit_semantics_applies_exactly_one_sigmoid() -> None:
    value = torch.tensor([-2.0, 0.0, 2.0])
    observed = mask_tensor_to_probability(value, semantics=MASK_TENSOR_LOGIT)
    assert torch.equal(observed, torch.sigmoid(value))


def test_missing_semantics_fails_closed_even_for_values_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="explicit mask_tensor_semantics"):
        mask_tensor_to_probability(torch.tensor([0.0, 1.0]), semantics=None)


def test_declared_probability_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="outside \\[0,1\\]"):
        mask_tensor_to_probability(
            torch.tensor([-0.01, 1.01]),
            semantics=MASK_TENSOR_PROBABILITY,
        )


def test_binary_threshold_is_defined_in_probability_space() -> None:
    assert torch.equal(
        mask_tensor_to_binary(
            torch.tensor([-1.0, 0.0, 1.0]),
            semantics=MASK_TENSOR_LOGIT,
        ),
        torch.tensor([False, True, True]),
    )
