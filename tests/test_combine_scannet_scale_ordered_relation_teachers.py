import torch

from radio_gs.scripts.combine_scannet_scale_ordered_relation_teachers import combine


def _payload(*, same: float, separate: float, teacher: str, same_only: bool) -> dict:
    metadata = {
        "query_free": True,
        "labels_opened": False,
        "instances_opened": False,
        "text_opened": False,
        "membership_lifting": "raster_adjoint",
        "raster_lifting_semantics": "true_alpha_compositing_adjoint",
        "vote_storage": "fp32_soft_same_and_separate_mass_no_overwrite",
        "scene_graph": "/tmp/graph.pt",
        "scene_graph_sha256": "graph",
        "responsibility_cache": "/tmp/mpr.pt",
        "responsibility_metadata": {"frozen": True},
        "inside_threshold": 0.8,
        "outside_threshold": 0.2,
        "minimum_primitives_per_mask": 3,
        "minimum_stability": 0.0,
        "scale_definition": "Q0.90_distance_to_coordinatewise_median_m",
        "scale_bins": 1,
        "scale_minimum_radius_m": 0.1,
        "scale_maximum_radius_m": 1.0,
        "mask_frames": [],
        "skipped_mask_frames": [],
        "teacher": teacher,
    }
    if same_only:
        metadata["confirmed_track_exterior"] = "not_used_same_only_positive_constraints"
    return {
        "schema_version": 2,
        "scene": "scene",
        "edge_rows": torch.tensor([0]),
        "edge_index": torch.tensor([[0], [1]]),
        "features": torch.zeros(1, 5),
        "same_votes": torch.tensor([[same]], dtype=torch.float32),
        "separate_votes": torch.tensor([[separate]], dtype=torch.float32),
        "observed_votes": torch.tensor([[same + separate]], dtype=torch.float32),
        "same_events": torch.tensor([int(same)]),
        "separate_events": torch.tensor([int(separate)]),
        "scale_bin_edges_log": torch.log(torch.tensor([0.1, 1.0])),
        "metadata": metadata,
    }


def test_confirmed_teacher_adds_only_same_vote_before_intervals(tmp_path) -> None:
    base = _payload(
        same=1.0,
        separate=2.0,
        teacher="official_sam3_multimask_scale_ordered_regions",
        same_only=False,
    )
    confirmed = _payload(
        same=3.0,
        separate=0.0,
        teacher="official_sam3_mpr_confirmed_source_target_track_same_only",
        same_only=True,
    )
    base_path, confirmed_path = tmp_path / "base.pt", tmp_path / "confirmed.pt"
    torch.save(base, base_path)
    torch.save(confirmed, confirmed_path)
    result = combine(base, confirmed, base_path=base_path, confirmed_path=confirmed_path)
    assert result["same_votes"].item() == 4.0
    assert result["separate_votes"].item() == 2.0
    assert result["same_events"].item() == 4
    assert result["separate_events"].item() == 2
    assert result["metadata"]["teacher"] == "official_sam3_multimask_plus_mpr_confirmed_tracks"
