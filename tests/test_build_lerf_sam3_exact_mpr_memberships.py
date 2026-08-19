import torch

from radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships import (
    exact_mpr_target_weights,
    lift_masks_with_exact_mpr,
)


def test_exact_mpr_target_weights_match_closed_form():
    pixels = torch.tensor([0, 0, 1])
    base = torch.tensor([0.6, 0.4, 0.5])
    observed = exact_mpr_target_weights(pixels, base, num_pixels=2)
    expected = torch.tensor([0.36, 0.16, 0.5])
    assert torch.allclose(observed, expected)


def test_lift_masks_uses_occlusion_aware_sparse_weights():
    logits = torch.tensor([[[8.0, -8.0]]])
    # Gaussian 0 contributes only to the foreground-mask pixel. Gaussian 1 is
    # visible in both pixels and therefore receives a mixed membership.
    gaussian_ids = torch.tensor([0, 1, 1])
    pixel_ids = torch.tensor([0, 0, 1])
    base_weights = torch.tensor([0.9, 0.1, 1.0])
    rows, proposals, weights = lift_masks_with_exact_mpr(
        logits,
        gaussian_ids,
        pixel_ids,
        base_weights,
        num_gaussians=2,
        feature_height=1,
        feature_width=2,
        min_membership=0.5,
    )
    assert rows.tolist() == [0]
    assert proposals.tolist() == [0]
    assert float(weights[0]) > 0.99


def test_lift_masks_applies_proposal_confidence():
    rows, proposals, weights = lift_masks_with_exact_mpr(
        torch.full((1, 1, 1), 8.0),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1.0]),
        num_gaussians=1,
        feature_height=1,
        feature_width=1,
        proposal_scores=torch.tensor([0.8]),
        min_membership=0.5,
    )
    assert rows.tolist() == [0]
    assert proposals.tolist() == [0]
    assert torch.allclose(weights, torch.tensor([0.8]), atol=1e-3)
