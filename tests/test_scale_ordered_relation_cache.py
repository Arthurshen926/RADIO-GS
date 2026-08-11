import json

import torch
import pytest

from radio_gs.scripts.build_scannet_scale_ordered_relation_cache import (
    _align_masks_to_raster,
    _assert_query_free_graph_provenance,
    _mask_cache_paths,
    _load_responsibility_assignments,
    _select_relation_graph_rows,
    _sha256_tensor,
    raster_responsibility_membership,
)


def test_raster_responsibility_membership_uses_mass_not_primitive_centres() -> None:
    masks = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)
    # Primitive 0 receives one inside and one outside pixel with equal mass;
    # primitive 1 receives only the inside pixel.  The result is deliberately
    # soft, allowing the caller to leave the 0.5 boundary primitive uncertain.
    membership, observed, resized = raster_responsibility_membership(
        masks,
        primitive_ids=torch.tensor([0, 0, 1]),
        pixel_ids=torch.tensor([0, 1, 3]),
        weights=torch.tensor([1.0, 1.0, 2.0]),
        primitive_count=2, image_height=2, image_width=2,
    )
    assert not resized
    torch.testing.assert_close(membership, torch.tensor([[0.5, 1.0]]))
    assert observed.tolist() == [True, True]


def test_responsibility_global_rows_map_exactly_to_valid_canonical_subset(tmp_path) -> None:
    full_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    capability = tmp_path / "capability.pt"
    torch.save({"xyz": full_xyz}, capability)
    responsibility = tmp_path / "responsibility.pt"
    torch.save({
        "schema_version": 1,
        "metadata": {
            "xyz_sha256": _sha256_tensor(full_xyz),
            "selected_frame_indices": [7], "feature_height": 1, "feature_width": 3,
        },
        "assignments": [{
            "gaussian_ids": torch.tensor([0, 1, 2]),
            "pixel_ids": torch.tensor([0, 1, 2]),
            "weights": torch.ones(3),
        }],
    }, responsibility)
    graph = {
        "xyz": full_xyz[[0, 2]], "global_rows": torch.tensor([0, 2]),
        "metadata": {"capability_cache": str(capability)},
    }
    assignments, metadata = _load_responsibility_assignments(responsibility, graph)
    assert metadata["relation_graph_identity"] == (
        "global_gaussian_responsibility_to_explicit_valid_canonical_subset"
    )
    assert assignments[7]["primitive_ids"].tolist() == [0, 1]
    assert assignments[7]["pixel_ids"].tolist() == [0, 2]


def test_sharded_exact_authority_supplies_frame_contract_without_loading_hits(
    tmp_path,
) -> None:
    full_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    capability = tmp_path / "capability.pt"
    torch.save({"xyz": full_xyz}, capability)
    view_root = tmp_path / "authority.json.views"
    view_root.mkdir()
    torch.save({"large_hit_tensor_was_not_needed": True}, view_root / "view_00000.pt")
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema": "radio_gs.sparse_exact_marginal_responsibility_authority.v1",
                "schema_version": 1,
                "frame_indices": [7],
                "metadata": {
                    "selected_frame_indices": [7],
                    "feature_height": 1,
                    "feature_width": 3,
                    "post_compositor_alpha_threshold": 0.0,
                    "xyz_sha256": _sha256_tensor(full_xyz),
                },
                "views": [
                    {
                        "view_index": 0,
                        "frame_index": 7,
                        "relative_path": "authority.json.views/view_00000.pt",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = {
        "xyz": full_xyz[[0, 2]],
        "global_rows": torch.tensor([0, 2]),
        "metadata": {"capability_cache": str(capability)},
    }
    assignments, metadata = _load_responsibility_assignments(authority, graph)
    assert assignments == {7: {"declared_by_exact_marginal_authority": True}}
    assert metadata["alpha_threshold"] == 0.0
    assert metadata["responsibility_storage"] == "sharded_exact_marginal_authority_v1"
    assert metadata["relation_graph_identity"] == (
        "global_gaussian_responsibility_to_explicit_valid_canonical_subset"
    )


def test_graph_provenance_accepts_explicit_modern_capability_contract() -> None:
    _assert_query_free_graph_provenance({
        "capability_metadata": {
            "query_independent": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    })


def test_graph_provenance_rejects_undeclared_or_leaky_contract() -> None:
    import pytest

    with pytest.raises(ValueError, match="lacks an explicit"):
        _assert_query_free_graph_provenance({})
    with pytest.raises(ValueError, match="violates"):
        _assert_query_free_graph_provenance({
            "capability_metadata": {
                "query_independent": True,
                "benchmark_masks_opened": True,
                "text_queries_opened": False,
            },
        })


def test_compositing_membership_uses_only_explicit_global_rows() -> None:
    full = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    observed = torch.tensor([True, False, True])
    graph = {"xyz": torch.zeros(2, 3), "global_rows": torch.tensor([2, 0])}
    member, selected_observed = _select_relation_graph_rows(
        full, observed, graph=graph,
        relation_graph_identity="global_gaussian_responsibility_to_explicit_valid_canonical_subset",
    )
    torch.testing.assert_close(member, torch.tensor([[0.3, 0.1], [0.6, 0.4]]))
    assert selected_observed.tolist() == [True, True]


def test_mask_alignment_records_discrete_raster_resample() -> None:
    masks = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)
    values, resized = _align_masks_to_raster(masks, image_height=1, image_width=2)
    assert resized
    assert values.dtype == torch.bool
    assert values.shape == (1, 1, 2)


def test_multiple_mask_roots_are_deterministic_and_reject_duplicate_stems(tmp_path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    torch.save({}, first / "20.pt")
    torch.save({}, second / "3.pt")
    assert [path.name for path in _mask_cache_paths(f"{first} {second}")] == ["3.pt", "20.pt"]
    torch.save({}, second / "20.pt")
    with pytest.raises(ValueError, match="duplicate"):
        _mask_cache_paths(f"{first},{second}")
