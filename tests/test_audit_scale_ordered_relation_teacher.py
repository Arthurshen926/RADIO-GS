import torch

from radio_gs.scripts.audit_scale_ordered_relation_teacher import summarize_cache


def test_teacher_audit_separates_coverage_and_interval_consistency(tmp_path) -> None:
    path = tmp_path / "scene.pt"
    torch.save({
        "schema_version": 2, "scene": "sceneX",
        "same_mass": torch.tensor([1.0, 0.0, 1.0]),
        "separate_mass": torch.tensor([0.0, 1.0, 1.0]),
        "constraint_entropy": torch.tensor([0.0, 0.0, 0.69]),
        "has_lower": torch.tensor([False, True, True]),
        "has_upper": torch.tensor([True, False, True]),
        "interval_consistent": torch.tensor([True, True, False]),
        "metadata": {
            "teacher": "official_sam3_multimask_scale_ordered_regions",
            "labels_opened": False, "instances_opened": False, "text_opened": False,
            "membership_lifting": "raster_responsibility", "raster_responsibility_used": True,
            "mask_frames": ["000000.pt"], "mask_schema_versions": [2],
        },
    }, path)
    row = summarize_cache(path, require_raster_responsibility=True)
    assert row["constrained_edge_fraction"] == 1.0
    assert row["both_bounds_edges"] == 1
    assert row["interval_consistent_fraction_among_both"] == 0.0
