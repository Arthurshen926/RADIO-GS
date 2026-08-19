import torch

from radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships_v2 import (
    lift_probability_masks_with_exact_mpr,
)


def test_probability_mask_is_not_passed_through_sigmoid() -> None:
    rows, proposals, weights = lift_probability_masks_with_exact_mpr(
        torch.tensor([[[0.0, 1.0]]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.ones(2),
        num_gaussians=2,
        feature_height=1,
        feature_width=2,
        min_membership=0.5,
    )
    assert rows.tolist() == [1]
    assert proposals.tolist() == [0]
    assert torch.equal(weights, torch.ones(1))


def test_probability_contract_rejects_logit_space() -> None:
    try:
        lift_probability_masks_with_exact_mpr(
            torch.tensor([[[-2.0, 2.0]]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.ones(2),
            num_gaussians=2,
            feature_height=1,
            feature_width=2,
        )
    except ValueError as error:
        assert "probability space" in str(error)
    else:
        raise AssertionError("logit-space tensor was silently accepted")
