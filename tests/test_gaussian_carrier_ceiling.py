import pytest
import torch

from radio_gs.evaluation.gaussian_carrier_ceiling import (
    binary_iou,
    binary_membership_entropy,
    soft_iou,
    threshold_iou_curve,
    weighted_carrier_mixing_summary,
)
from radio_gs.scripts.audit_gaussian_carrier_ceiling import _sample_pixels


def test_membership_entropy_separates_pure_and_mixed_rows() -> None:
    foreground = torch.tensor([0.0, 0.5, 1.0, 0.0])
    total = torch.tensor([1.0, 1.0, 1.0, 0.0])

    membership, entropy = binary_membership_entropy(foreground, total)

    torch.testing.assert_close(membership, torch.tensor([0.0, 0.5, 1.0, 0.0]))
    assert float(entropy[1]) == pytest.approx(1.0)
    assert float(entropy[[0, 2, 3]].max()) < 1e-5


def test_weighted_mixing_counts_only_observed_ambiguous_rows() -> None:
    summary = weighted_carrier_mixing_summary(
        torch.tensor([0.0, 0.5, 1.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )

    assert summary["observed_rows"] == 3
    assert summary["ambiguous_rows"] == 1
    assert summary["ambiguous_row_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["ambiguous_total_mass_fraction"] == pytest.approx(1.0 / 3.0)


def test_threshold_curve_and_soft_iou_have_fixed_score_semantics() -> None:
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.6], [0.2, 0.7]])
    target = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool)

    curve = threshold_iou_curve(scores, target, [0.5, 0.75])

    torch.testing.assert_close(curve[:, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(curve[:, 1], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(binary_iou(scores >= 0.5, target), curve[0])
    assert bool((soft_iou(scores, target) >= 0).all())


def test_invalid_masses_fail_closed() -> None:
    with pytest.raises(ValueError, match="0 <= fg <= total"):
        binary_membership_entropy(torch.tensor([2.0]), torch.tensor([1.0]))


def test_zero_stratified_caps_mean_uniform_only() -> None:
    foreground = torch.zeros(20, 2, dtype=torch.bool)
    foreground[:10, 0] = True
    boundary = torch.zeros_like(foreground)
    boundary[5:15, 1] = True

    selected = _sample_pixels(
        foreground,
        boundary,
        foreground_cap=0,
        boundary_cap=0,
        random_cap=7,
        seed=3,
    )

    assert selected.numel() == 7
