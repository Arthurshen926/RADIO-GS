import torch

from radio_gs.scripts.build_surface_region_semantic_cache import (
    _adjacency,
    completion_primary_valid,
    preserve_primary_region_tokens,
    two_hop_physical_regions,
)
from radio_gs.scripts.eval_scannet_canonical_text_query import (
    load_primitive_multiscale_features,
    load_primitive_semantic_cache,
)


def test_two_hop_regions_are_unique_and_physical_scale_clipped() -> None:
    graph = {"xyz": torch.zeros(4, 3),
             "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
             "raw_affinity": torch.ones(6)}
    adjacency = _adjacency(graph, 2)
    xyz = torch.tensor([[0., 0, 0], [.1, 0, 0], [.2, 0, 0], [.5, 0, 0]])
    rows, mask = two_hop_physical_regions(torch.tensor([0]), adjacency, xyz, 0.25)
    kept = rows[0, mask[0]]
    assert set(kept.tolist()) == {0, 1, 2}
    assert len(kept) == len(torch.unique(kept))


def test_completed_surface_regions_preserve_primary_context() -> None:
    primary = torch.tensor([True, True, False, False])
    rows = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    mask = torch.ones_like(rows, dtype=torch.bool)

    kept = preserve_primary_region_tokens(
        rows,
        mask,
        centers=torch.tensor([0, 2]),
        primary_valid=primary,
    )

    assert torch.equal(kept[0], torch.tensor([True, True, False, False]))
    assert torch.equal(kept[1], torch.tensor([True, True, True, False]))


def test_completed_mpr_primary_partition_is_fail_closed() -> None:
    valid = torch.tensor([True, True, True, False])
    mpr = {
        "reliability": torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ),
        "metadata": {
            "construction": "dominant_primary_with_query_free_support_completion",
            "primary_valid_count": 2,
        },
    }

    primary = completion_primary_valid(mpr, valid)

    assert torch.equal(primary, torch.tensor([True, False, True, False]))


def test_sparse_v4_cache_expands_losslessly(tmp_path) -> None:
    path = tmp_path / "semantic.pt"
    xyz = torch.randn(5, 3)
    rows = torch.tensor([1, 4])
    valid = torch.zeros(5, dtype=torch.bool); valid[rows] = True
    sparse = torch.randn(2, 1536).half()
    torch.save({
        "xyz": xyz, "features": sparse, "summary_features": sparse,
        "global_rows": rows, "valid": valid,
        "metadata": {
            "schema_version": 4,
            "source": "canonical_radio_surface_region_readout",
            "official_summary_head": True,
            "custom_text_projection": False,
            "query_set_invariant": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }, path)
    loaded_xyz, loaded_valid, features, _ = load_primitive_semantic_cache(path)
    assert torch.equal(loaded_xyz, xyz)
    assert torch.equal(loaded_valid, valid)
    assert torch.equal(features[rows], sparse)
    assert not bool(features[~valid].any())


def test_sparse_v5_multiscale_cache_expands_losslessly(tmp_path) -> None:
    path = tmp_path / "semantic_multiscale.pt"
    xyz = torch.randn(5, 3)
    rows = torch.tensor([1, 4])
    valid = torch.zeros(5, dtype=torch.bool); valid[rows] = True
    sparse = torch.randn(2, 1536).half()
    scales = torch.randn(2, 3, 1536).half()
    torch.save({
        "xyz": xyz, "features": sparse, "summary_features": sparse,
        "features_by_scale": scales, "global_rows": rows, "valid": valid,
        "metadata": {
            "schema_version": 5,
            "source": "canonical_radio_surface_region_readout",
            "official_summary_head": True,
            "custom_text_projection": False,
            "query_set_invariant": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }, path)
    _, loaded_valid, features, _ = load_primitive_semantic_cache(path)
    loaded_scales = load_primitive_multiscale_features(path, valid=loaded_valid)
    assert torch.equal(features[rows], sparse)
    assert loaded_scales is not None
    assert torch.equal(loaded_scales[rows], scales)
    assert not bool(loaded_scales[~valid].any())
