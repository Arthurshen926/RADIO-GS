from __future__ import annotations

import torch

from radio_gs.querying.bounded_region_comembership_readout import (
    bounded_regions_for_seed,
    bridge_free_component_ids,
    thresholded_adjacency,
)


def _two_triangles_with_bridge():
    pairs = torch.tensor(
        [
            [0, 0, 1, 3, 3, 4, 2],
            [1, 2, 2, 4, 5, 5, 3],
        ],
        dtype=torch.int64,
    )
    probability = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.95])
    return thresholded_adjacency(
        region_count=6,
        pair_indices=pairs,
        pair_probabilities=probability,
        threshold=0.8,
    )


def test_dual_path_removes_single_false_positive_bridge() -> None:
    adjacency = _two_triangles_with_bridge()
    components = bridge_free_component_ids(adjacency)
    widest = bounded_regions_for_seed(
        method="widest_path",
        seed_region_index=0,
        adjacency=adjacency,
        maximum_regions=8,
    )
    dual = bounded_regions_for_seed(
        method="dual_path_widest",
        seed_region_index=0,
        adjacency=adjacency,
        maximum_regions=8,
        bridge_free_components=components,
    )
    assert set(widest) == set(range(6))
    assert set(dual) == {0, 1, 2}
    assert int(components[2]) != int(components[3])


def test_multipoint_consistency_rejects_single_bridge() -> None:
    adjacency = _two_triangles_with_bridge()
    selected = bounded_regions_for_seed(
        method="multipoint_consistency",
        seed_region_index=0,
        adjacency=adjacency,
        maximum_regions=8,
    )
    assert set(selected) == {0, 1, 2}


def test_bounded_readout_never_exceeds_global_k() -> None:
    adjacency = _two_triangles_with_bridge()
    for method in ("maximum_product", "widest_path", "multipoint_consistency"):
        selected = bounded_regions_for_seed(
            method=method,
            seed_region_index=0,
            adjacency=adjacency,
            maximum_regions=2,
        )
        assert len(selected) == 2
