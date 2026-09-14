from __future__ import annotations

import pytest
import torch

from radio_gs.v4.contracts.lerf_fragment_surface_memory import (
    SCHEMA,
    _hierarchy_depth,
    fragment_evidence,
    validate_memory,
    source_mask_raster,
)


def test_native_mask_sampling_does_not_inherit_semantic_feature_grid():
    payload = {"mask_shape": [728, 986]}
    assert source_mask_raster(payload, (46, 62), "native") == (728, 986)
    assert source_mask_raster(payload, (46, 62), "reference") == (46, 62)
    with pytest.raises(ValueError):
        source_mask_raster({"mask_shape": [0, 1]}, (46, 62), "native")


def _payload():
    return {
        "schema": SCHEMA,
        "source_frames": [10, 20],
        "fragment_positive": torch.tensor([[0.8, 0.0, 0.2], [0.0, 0.6, 0.0]]),
        "view_visibility": torch.tensor([[True, False, True], [False, True, False]]),
        "fragment_view_index": torch.tensor([0, 1]),
        "fragment_frame_id": torch.tensor([10, 20]),
        "fragment_local_index": torch.tensor([0, 0]),
        "quality": torch.tensor([0.9, 0.8]),
        "stability": torch.tensor([0.95, 0.85]),
        "parent_index": torch.tensor([-1, -1]),
        "hierarchy_depth": torch.tensor([0, 0]),
        "information_policy": {
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_rgb_opened": False,
        },
    }


def test_fragment_evidence_reconstructs_three_state_source_facts():
    positive, negative, unknown = fragment_evidence(_payload())
    assert torch.equal(positive + negative + unknown, torch.ones_like(positive))
    assert negative[0].tolist() == pytest.approx([0.2, 0.0, 0.8])
    assert unknown[0].tolist() == [0.0, 1.0, 0.0]
    assert unknown[1].tolist() == [1.0, 0.0, 1.0]


def test_fragment_memory_rejects_positive_outside_view_visibility():
    payload = _payload()
    payload["fragment_positive"][0, 1] = 0.2
    with pytest.raises(ValueError, match="outside its source visibility"):
        validate_memory(payload)


def test_hierarchy_depth_detects_cycles_and_preserves_levels():
    assert _hierarchy_depth(torch.tensor([-1, 0, 1, 0])).tolist() == [0, 1, 2, 1]
    with pytest.raises(ValueError, match="cycle"):
        _hierarchy_depth(torch.tensor([1, 0]))
