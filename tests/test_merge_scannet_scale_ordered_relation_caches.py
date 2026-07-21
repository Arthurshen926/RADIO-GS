from pathlib import Path

import pytest
import torch

from radio_gs.scripts.merge_scannet_scale_ordered_relation_caches import merge_relation_caches


def _payload(*, frame: str, same: torch.Tensor, separate: torch.Tensor, dtype=torch.float32) -> dict:
    bins = torch.log(torch.tensor([0.1, 0.5, 2.0]))
    return {
        "schema_version": 2,
        "scene": "scene0000",
        "edge_rows": torch.tensor([0, 2]),
        "edge_index": torch.tensor([[0, 1], [1, 2]]),
        "features": torch.tensor([[0.1], [0.2]]),
        "same_votes": same.to(dtype),
        "separate_votes": separate.to(dtype),
        "observed_votes": (same + separate).to(dtype),
        "same_events": torch.tensor([1, 2]),
        "separate_events": torch.tensor([3, 4]),
        "scale_bin_edges_log": bins,
        "metadata": {
            "query_free": True, "labels_opened": False, "instances_opened": False, "text_opened": False,
            "scene_graph": "/tmp/graph.pt", "scene_graph_sha256": "graph",
            "membership_lifting": "raster_adjoint",
            "raster_lifting_semantics": "true_alpha_compositing_adjoint",
            "raster_responsibility_used": True,
            "responsibility_cache": "/tmp/responsibility.pt", "responsibility_metadata": {"xyz_sha256": "xyz"},
            "raster_adjoint_provenance": {"checkpoint": "/tmp/model.pt"},
            "mask_raster_alignment": "nearest_label_resample_to_frozen_mpr_raster",
            "inside_threshold": 0.8, "outside_threshold": 0.2,
            "minimum_primitives_per_mask": 3, "minimum_stability": 0.0,
            "scale_definition": "Q0.90_distance_to_coordinatewise_median_m",
            "scale_bins": 2, "scale_minimum_radius_m": 0.1, "scale_maximum_radius_m": 2.0,
            "vote_storage": "fp32_soft_same_and_separate_mass_no_overwrite",
            "mask_roots": [f"/tmp/{frame}"], "mask_frames": [f"{frame}.pt"],
            "skipped_mask_frames": [], "resized_mask_frames": [f"{frame}.pt"],
            "resized_mask_count": 1, "mask_schema_versions": [2],
        },
    }


def test_merge_adds_unquantized_votes_and_rebuilds_intervals(tmp_path: Path) -> None:
    first = _payload(
        frame="20", same=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        separate=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    )
    second = _payload(
        frame="3", same=torch.tensor([[0.0, 2.0], [1.0, 0.0]]),
        separate=torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
    )
    first_path, second_path = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save(first, first_path); torch.save(second, second_path)

    merged = merge_relation_caches([first_path, second_path])

    torch.testing.assert_close(merged["same_votes"], first["same_votes"] + second["same_votes"])
    torch.testing.assert_close(merged["separate_votes"], first["separate_votes"] + second["separate_votes"])
    assert merged["same_votes"].dtype == torch.float32
    assert merged["metadata"]["mask_frames"] == ["3.pt", "20.pt"]
    # The first edge has a small-scale lower bound and a larger same-mask
    # upper bound, so merging must reconstruct a consistent interval.
    assert bool(merged["has_lower"][0]) and bool(merged["has_upper"][0])
    assert bool(merged["interval_consistent"][0])


def test_merge_refuses_duplicate_masks_or_quantized_shards(tmp_path: Path) -> None:
    first = _payload(
        frame="0", same=torch.zeros(2, 2), separate=torch.zeros(2, 2),
    )
    duplicate = _payload(
        frame="0", same=torch.zeros(2, 2), separate=torch.zeros(2, 2),
    )
    first_path, duplicate_path = tmp_path / "first.pt", tmp_path / "duplicate.pt"
    torch.save(first, first_path); torch.save(duplicate, duplicate_path)
    with pytest.raises(ValueError, match="appear in more than one"):
        merge_relation_caches([first_path, duplicate_path])

    quantized = _payload(
        frame="1", same=torch.zeros(2, 2), separate=torch.zeros(2, 2), dtype=torch.float16,
    )
    quantized["metadata"]["vote_storage"] = "fp16_soft_same_and_separate_mass_no_overwrite"
    quantized_path = tmp_path / "quantized.pt"; torch.save(quantized, quantized_path)
    with pytest.raises(ValueError, match="quantized"):
        merge_relation_caches([first_path, quantized_path])
