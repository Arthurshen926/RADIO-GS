import pytest
import torch

from radio_gs.evaluation.render_ceiling import (
    PixelMetricAccumulator,
    contribution_coverage,
    coverage_bin_masks,
    normalize_premultiplied,
    parse_coverage_edges,
)


def test_contribution_coverage_uses_total_alpha_and_zeros_empty_pixels():
    total = torch.tensor([[0.8, 0.4, 0.0]])
    valid = torch.tensor([[0.6, 0.1, 0.0]])

    result = contribution_coverage(valid, total)

    torch.testing.assert_close(result, torch.tensor([[0.75, 0.25, 0.0]]))


def test_conditional_normalization_divides_by_valid_mass_not_total_alpha():
    numerator = torch.tensor([[[0.3, 0.2]], [[0.6, 0.1]]])
    valid_mass = torch.tensor([[0.5, 0.0]])

    result = normalize_premultiplied(numerator, valid_mass)

    torch.testing.assert_close(result[:, 0, 0], torch.tensor([0.6, 1.2]))
    torch.testing.assert_close(result[:, 0, 1], torch.zeros(2))


def test_coverage_bins_are_disjoint_and_cover_base_mask():
    coverage = torch.tensor([[0.0, 0.25, 0.74, 0.95, 1.0]])
    base = torch.tensor([[True, True, False, True, True]])
    masks = coverage_bin_masks(
        coverage, base, parse_coverage_edges("0,.25,.5,.75,.95,1.000001")
    )

    stacked = torch.stack(list(masks.values())).long().sum(dim=0)

    torch.testing.assert_close(stacked, base.long())
    assert all(
        not bool((left & right).any())
        for index, left in enumerate(masks.values())
        for right in list(masks.values())[index + 1 :]
    )


def test_metric_accumulator_reports_positive_coverage_error_correlation():
    target = torch.ones(2, 1, 4)
    prediction = torch.tensor([[[0.0, 0.2, 0.8, 1.0]], [[1.0, 1.0, 1.0, 1.0]]])
    coverage = torch.tensor([[0.0, 0.3, 0.7, 1.0]])
    accumulator = PixelMetricAccumulator()
    accumulator.update(
        prediction,
        target,
        torch.ones(1, 4, dtype=torch.bool),
        coverage=coverage,
    )

    summary = accumulator.summary()

    assert summary["pixels"] == 4
    assert summary["coverage_error_pearson"] is not None
    assert summary["coverage_error_pearson"] < 0


def test_coverage_edges_reject_missing_unit_interval():
    with pytest.raises(ValueError, match="start at 0"):
        parse_coverage_edges("0.1,0.5,1")
