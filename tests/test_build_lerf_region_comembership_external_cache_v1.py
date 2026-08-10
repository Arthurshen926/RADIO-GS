from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.build_lerf_region_comembership_external_cache_v1 import (
    validate_scale_major_alignment,
)


def test_scale_major_alignment_accepts_frozen_radii_and_indices() -> None:
    validate_scale_major_alignment(
        canonical_region_indices=torch.tensor([0, 4, 8]),
        scale_indices=torch.tensor([0, 1, 2]),
        anchor_count=4,
        o0_scale_radii_m=(0.25, 0.45, 0.7),
    )


def test_scale_major_alignment_rejects_swapped_o0_radii() -> None:
    with pytest.raises(ValueError, match="scale-major"):
        validate_scale_major_alignment(
            canonical_region_indices=torch.tensor([0, 4, 8]),
            scale_indices=torch.tensor([0, 1, 2]),
            anchor_count=4,
            o0_scale_radii_m=(0.45, 0.25, 0.7),
        )
