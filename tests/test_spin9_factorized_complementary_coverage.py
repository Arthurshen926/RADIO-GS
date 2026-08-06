from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.build_spin9_factorized_source_quantile_target_prediction import (
    _paired_complementary_coverage,
)


LEGACY_TOLERANCE = float(16 * torch.finfo(torch.float32).eps)


def paired(visible, invisible, alpha):
    return _paired_complementary_coverage(
        torch.tensor(visible, dtype=torch.float32),
        torch.tensor(invisible, dtype=torch.float32),
        torch.tensor(alpha, dtype=torch.float32),
        legacy_tolerance=LEGACY_TOLERANCE,
    )


def test_exact_zero_denominator_is_unsupported_background() -> None:
    coverage, diagnostic = paired([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    assert torch.equal(coverage, torch.zeros(2, dtype=torch.float32))
    assert diagnostic["unsupported_zero_pixels"] == 2
    assert diagnostic["supported_positive_pixels"] == 0


def test_extremely_small_positive_denominator_uses_fp64_ratio() -> None:
    coverage, diagnostic = paired([1e-30, 3e-35], [1e-30, 1e-35], [0.0, 0.0])
    assert torch.allclose(coverage, torch.tensor([0.5, 0.75]), atol=1e-7, rtol=0)
    assert diagnostic["unsupported_zero_pixels"] == 0
    assert diagnostic["minimum_positive_denominator"] > 0
    assert diagnostic["maximum_fp32_vs_fp64_oracle_absolute_difference"] <= 3e-8


def test_negative_mass_fails_closed_without_clipping() -> None:
    with pytest.raises(RuntimeError, match="negative paired mass"):
        paired([0.5], [-1e-8], [0.5])


def test_legacy_admissible_domain_parity() -> None:
    coverage, diagnostic = paired([0.2, 0.7], [0.8, 0.3], [1.0, 1.0])
    assert torch.allclose(coverage, torch.tensor([0.2, 0.7]), atol=1e-7, rtol=0)
    assert diagnostic["legacy_admissible_pixels"] == 2
    assert (
        diagnostic["maximum_corrected_vs_legacy_admissible_absolute_difference"]
        <= 1e-7
    )


def test_legacy_overshoot_is_repaired_by_algebra_not_clip() -> None:
    visible = 1.0 + 8.344650268554688e-6
    coverage, diagnostic = paired([visible], [0.0], [1.0])
    assert coverage.item() == 1.0
    assert diagnostic["legacy_positive_overshoot"] > LEGACY_TOLERANCE
    assert diagnostic["corrected_maximum"] == 1.0

