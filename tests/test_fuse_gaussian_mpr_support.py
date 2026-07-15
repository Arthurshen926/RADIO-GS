from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.fuse_gaussian_mpr_support import fuse_primary_with_support


def _cache(valid: list[bool], offset: float = 0.0) -> dict:
    count = len(valid)
    return {
        "xyz": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        "features": torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
        + offset,
        "valid": torch.tensor(valid),
        "view_counts": torch.arange(1, count + 1),
        "reliability": torch.ones(count, 3) * (1.0 + offset),
        "metadata": {
            "feature_space": "radio",
            "construction": f"cache_{offset}",
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }


def test_fusion_preserves_primary_and_only_fills_uncovered_rows() -> None:
    primary = _cache([True, False, True, False])
    support = _cache([True, True, False, False], offset=100.0)

    fused, report = fuse_primary_with_support(primary, support)

    assert fused["valid"].tolist() == [True, True, True, False]
    assert torch.equal(fused["features"][0], primary["features"][0])
    assert torch.equal(fused["features"][1], support["features"][1])
    assert torch.equal(fused["features"][2], primary["features"][2])
    assert report["fallback_valid_count"] == 1
    assert report["primary_rows_preserved"] is True
    assert fused["reliability"][:, 2].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_fusion_rejects_geometry_mismatch() -> None:
    primary = _cache([True, False])
    support = _cache([False, True])
    support["xyz"][1, 0] += 1.0

    with pytest.raises(ValueError, match="row-aligned geometry"):
        fuse_primary_with_support(primary, support)


def test_fusion_rejects_benchmark_contamination() -> None:
    primary = _cache([True, False])
    support = _cache([False, True])
    support["metadata"]["benchmark_masks_opened"] = True

    with pytest.raises(ValueError, match="benchmark-contaminated"):
        fuse_primary_with_support(primary, support)
