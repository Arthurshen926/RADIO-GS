import pytest
import torch

from radio_gs.scripts.build_surface_region_semantic_cache import (
    directional_anchor_tokens,
    project_surface_codebook_slots,
    surface_region_radio_tokens,
)
from radio_gs.scripts.materialize_gauge_projected_canonical_field import _safe


def test_directional_anchor_replaces_only_selected_token() -> None:
    context = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    anchors = torch.tensor([[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0]])
    result = directional_anchor_tokens(context, anchors, torch.tensor([0, 2]))
    assert torch.equal(result[0, 0], anchors[0])
    assert torch.equal(result[1, 2], anchors[1])
    assert torch.equal(result[0, 1:], context[0, 1:])
    assert torch.equal(result[1, :2], context[1, :2])
    assert torch.equal(context, torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))


def test_directional_anchor_rejects_invalid_index() -> None:
    with pytest.raises(IndexError, match="outside"):
        directional_anchor_tokens(
            torch.zeros(1, 2, 3), torch.ones(1, 3), torch.tensor([2])
        )


def test_surface_region_radio_tokens_enforces_training_direction_gauge() -> None:
    values = torch.tensor(
        [[[3.0, 4.0, 0.0], [0.0, 0.0, 7.0]]], dtype=torch.float16
    )
    result = surface_region_radio_tokens(
        values,
        normalization="l2_direction",
    )
    assert result.dtype == values.dtype
    assert torch.allclose(
        result.float().norm(dim=-1),
        torch.ones(1, 2),
        atol=5e-4,
    )
    assert torch.equal(
        surface_region_radio_tokens(values, normalization="legacy_raw"),
        values,
    )


def test_surface_region_radio_tokens_rejects_unknown_gauge() -> None:
    with pytest.raises(ValueError, match="normalization"):
        surface_region_radio_tokens(
            torch.ones(1, 1, 3),
            normalization="implicit",
        )


def test_project_surface_codebook_slots_uses_independent_single_slot_calls() -> None:
    class RecordingHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shapes: list[tuple[int, ...]] = []

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.shapes.append(tuple(values.shape))
            return values[..., :8]

    head = RecordingHead()
    tokens = torch.randn(3, 4, 1280)
    result = project_surface_codebook_slots(head, tokens)
    assert result.shape == (3, 4, 8)
    assert head.shapes == [(3, 1, 1280)] * 4
    assert torch.allclose(result.norm(dim=-1), torch.ones(3, 4), atol=1e-6)


def test_project_surface_codebook_slots_rejects_wrong_slot_shape() -> None:
    with pytest.raises(ValueError, match=r"\[B,4,1280\]"):
        project_surface_codebook_slots(
            torch.nn.Identity(), torch.zeros(2, 3, 1280)
        )


def test_gauge_safety_legacy_fallback_is_narrow() -> None:
    _safe(
        {
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "mpr_cache_metadata": {"benchmark_images_opened": False},
        },
        "legacy",
    )
    with pytest.raises(ValueError, match="task contaminated"):
        _safe(
            {
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
                "mpr_cache_metadata": {},
            },
            "unsafe legacy",
        )
