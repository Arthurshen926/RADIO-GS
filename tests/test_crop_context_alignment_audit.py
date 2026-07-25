from __future__ import annotations

import torch

from radio_gs.benchmarks.scannet_pfpr.audit_crop_context_alignment import (
    crop_context_descriptors,
)


def test_crop_context_descriptors_keep_center_global_and_summary_normalized() -> None:
    spatial = torch.arange(2 * 4 * 5 * 3, dtype=torch.float32).reshape(2, 4, 5, 3)
    summary = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) + 1.0

    views = crop_context_descriptors(summary, spatial)

    assert set(views) == {"center3x3", "spatial_global", "official_crop_summary"}
    for values in views.values():
        assert values.shape == (2, 4)
        assert torch.allclose(values.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_crop_context_descriptors_ignore_incompatible_summary_shape() -> None:
    spatial = torch.ones(1, 4, 3, 3)
    views = crop_context_descriptors(torch.ones(1, 2, 4), spatial)
    assert "official_crop_summary" not in views
