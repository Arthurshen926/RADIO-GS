from __future__ import annotations

import pytest
import torch

from radio_gs.losses.generic_region_text_response import (
    FrozenGenericRegionTextBundle,
    generic_region_text_response_loss,
)


def _bundle(dimension: int = 6) -> FrozenGenericRegionTextBundle:
    generator = torch.Generator().manual_seed(4)
    primary = torch.nn.functional.normalize(
        torch.randn(8, dimension, generator=generator), dim=-1
    )
    synonym = torch.nn.functional.normalize(
        torch.randn(4, dimension, generator=generator), dim=-1
    )
    return FrozenGenericRegionTextBundle(
        primary_text=primary,
        synonym_text=synonym,
        synonym_left_primary_indices=torch.tensor([0, 2, 4, 6]),
        synonym_right_indices=torch.tensor([0, 1, 2, 3]),
        sibling_left_primary_indices=torch.tensor([0, 2, 4]),
        sibling_right_primary_indices=torch.tensor([1, 3, 5]),
        relation_authority_sha256="a" * 64,
        relation_content_authority_sha256="b" * 64,
        primary_file_sha256="c" * 64,
        primary_embedding_sha256="d" * 64,
        synonym_file_sha256="e" * 64,
        synonym_embedding_sha256="f" * 64,
    )


def test_identical_region_maps_have_zero_response_loss() -> None:
    teacher = torch.randn(1, 6, 16, 16, generator=torch.Generator().manual_seed(7))
    alpha = torch.ones(16, 16)

    loss, stats = generic_region_text_response_loss(
        teacher.clone(), teacher, alpha, _bundle(), alpha_threshold=0.02
    )

    assert float(loss) == pytest.approx(0.0, abs=2e-6)
    assert float(stats["profile_cosine"]) == pytest.approx(1.0, abs=2e-6)
    assert stats["regions"] == 64


def test_response_loss_aligns_one_row_full_extent_rounding_difference() -> None:
    teacher = torch.randn(1, 6, 46, 62, generator=torch.Generator().manual_seed(11))
    predicted = torch.nn.functional.interpolate(
        teacher,
        size=(45, 62),
        mode="bilinear",
        align_corners=False,
    )

    loss, stats = generic_region_text_response_loss(
        predicted,
        teacher,
        torch.ones(45, 62),
        _bundle(),
        alpha_threshold=0.02,
    )

    assert float(loss) == pytest.approx(0.0, abs=2e-6)
    assert float(stats["profile_cosine"]) == pytest.approx(1.0, abs=2e-6)


def test_response_loss_is_differentiable_and_detects_spatial_permutation() -> None:
    teacher = torch.randn(1, 6, 16, 16, generator=torch.Generator().manual_seed(9))
    predicted = teacher.flip(-1).clone().requires_grad_(True)

    loss, stats = generic_region_text_response_loss(
        predicted,
        teacher,
        torch.ones(16, 16),
        _bundle(),
        alpha_threshold=0.02,
    )
    loss.backward()

    assert float(loss) > 0.05
    assert float(stats["profile_cosine"]) < 0.9
    assert predicted.grad is not None
    assert bool(torch.isfinite(predicted.grad).all())
    assert float(predicted.grad.abs().sum()) > 0.0


def test_fewer_than_two_visible_regions_returns_exact_zero() -> None:
    predicted = torch.randn(1, 6, 8, 8, requires_grad=True)
    alpha = torch.zeros(8, 8)
    alpha[0, 0] = 1.0

    loss, stats = generic_region_text_response_loss(
        predicted,
        predicted.detach(),
        alpha,
        _bundle(),
        alpha_threshold=0.02,
    )
    loss.backward()

    assert float(loss) == 0.0
    assert stats["regions"] == 1
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))


def test_contract_rejects_descriptor_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        generic_region_text_response_loss(
            torch.randn(1, 5, 8, 8),
            torch.randn(1, 5, 8, 8),
            torch.ones(8, 8),
            _bundle(),
            alpha_threshold=0.02,
        )
