import random

import numpy as np
import torch

from radio_gs.scripts.build_scannet_surface_region_cache import (
    _region_indices,
    _surface_radius_graph,
    _teacher_medoid,
    _voxel_fuse,
)


def test_voxel_fusion_and_physical_surface_regions() -> None:
    xyz = torch.tensor([[0.00, 0, 0], [0.01, 0, 0], [0.05, 0, 0], [0.10, 0, 0]])
    features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    fused_xyz, fused_features, footprint, count = _voxel_fuse(
        xyz, features, torch.full((4,), 0.02), 0.04
    )
    assert len(fused_xyz) == 3
    assert count.max() == 2
    graph = _surface_radius_graph(fused_xyz, 0.04)
    rows = _region_indices(graph, 0, 0.12, min_tokens=2, max_tokens=8, rng=random.Random(0))
    assert rows is not None and len(rows) >= 2
    assert torch.isfinite(fused_features).all() and torch.isfinite(footprint).all()


def test_teacher_medoid_selects_consensus_view() -> None:
    tokens = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    assert _teacher_medoid(tokens) in {0, 1}
