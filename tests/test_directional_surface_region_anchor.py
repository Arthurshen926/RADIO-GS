import pytest
import torch

from radio_gs.scripts.build_surface_region_semantic_cache import (
    directional_anchor_tokens,
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
