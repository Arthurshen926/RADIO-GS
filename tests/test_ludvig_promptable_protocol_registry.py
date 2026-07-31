from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "paper" / "artifacts" / "promptable_nvs_protocol_registry.yaml"


def _registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def test_ludvig_official_protocol_is_fail_closed_against_strict_unseen() -> None:
    protocol = _registry()["protocols"]["ludvig_official_online_multiview_v1"]

    assert protocol["common"]["target_rgb_visible_during_gaussian_splatting_training"] is True
    assert protocol["common"]["target_rgb_visible_during_uplifting"] is True
    assert protocol["common"]["target_view_2d_foundation_model_calls"] is True
    assert protocol["common"]["target_mask_use"] == "scoring_only"
    assert protocol["exact_match"]["nvos_strict_unseen_v1"] is False
    assert protocol["exact_match"]["spin_nerf_full_reference_mask_10scene_v1"] is False


def test_ludvig_source_calibration_and_aggregation_are_explicit() -> None:
    protocol = _registry()["protocols"]["ludvig_official_online_multiview_v1"]

    assert protocol["nvos"]["prompt"]["official_negative_scribble_used_by_released_code"] is False
    assert protocol["nvos"]["evaluation"]["aggregation"] == "equal_weight_macro_over_8_tasks"
    calibration = protocol["spin_nerf"]["source_mask_calibration"]
    assert calibration["reference_ground_truth_mask_used"] is True
    assert calibration["per_scene_threshold_selected_by_reference_iou"] is True
    assert calibration["per_scene_sam_candidate_selected_by_reference_iou"] is True
    assert calibration["non_reference_target_masks_used"] is False
    assert (
        protocol["spin_nerf"]["evaluation"]["aggregation"]
        == "frame_mean_then_equal_weight_macro_over_10_scenes"
    )


def test_ludvig_spin_9scene_context_is_recomputed_not_full_mean() -> None:
    registry = _registry()
    diagnostic = registry["protocols"][
        "ludvig_spin_nerf_9scene_without_fork_diagnostic_v1"
    ]
    sam = next(
        row
        for row in registry["reported_context"]
        if row["method_id"] == "marrie_et_al_iccv_2025_ludvig_sam"
    )
    scene_values = sam["published_per_scene_iou"]["spin_nerf"]
    available = diagnostic["scenes"]
    recomputed = sum(scene_values[scene] for scene in available) / len(available)

    assert diagnostic["missing_scenes"] == ["fork"]
    assert diagnostic["eligible_for_spin_nerf_10scene_table"] is False
    assert diagnostic["eligible_for_strict_unseen_claim"] is False
    assert recomputed == pytest.approx(
        diagnostic["paper_reported_same_scene_macro"]["ludvig_sam"]
    )
    assert recomputed != pytest.approx(sam["values"]["spin_nerf"]["miou"])


def test_ludvig_context_rows_reference_incompatible_protocol() -> None:
    rows = {
        row["method"]: row
        for row in _registry()["reported_context"]
        if row["method"] in {"LUDVIG-SAM", "LUDVIG-SAM2"}
    }

    assert set(rows) == {"LUDVIG-SAM", "LUDVIG-SAM2"}
    for row in rows.values():
        assert row["protocol_id"] == "ludvig_official_online_multiview_v1"
        assert row["strict_unseen_exact_match"] is False
        assert row["exact_local_evaluator"] is False


def test_ludvig_fern_hybrid_pilot_is_machine_readable_and_ineligible() -> None:
    pilot = _registry()["protocols"][
        "ludvig_nvos_fern_strict_geometry_hybrid_diagnostic_v1"
    ]

    assert pilot["result"]["selected_foreground_iou_percent"] == pytest.approx(
        84.01535329822478
    )
    assert pilot["result"]["delta_local_minus_paper_points"] == pytest.approx(
        -1.484646701775219
    )
    assert pilot["runtime"]["gpu_wall_time_seconds"] == pytest.approx(
        94.28004455566406
    )
    assert pilot["visibility"][
        "target_rgb_visible_during_gaussian_splatting_training"
    ] is False
    assert pilot["visibility"]["target_rgb_visible_during_uplifting"] is True
    assert pilot["eligibility"]["exact_paper_protocol_comparison"] is False
    assert pilot["eligibility"]["strict_unseen_exact_match"] is False
    assert pilot["eligibility"]["full_nvos_cohort_report"] is False


def test_ludvig_exact_all_view_dry_run_is_pinned_and_hardware_blocked() -> None:
    registry = _registry()
    official = registry["protocols"]["ludvig_official_online_multiview_v1"]
    preflight = registry["protocols"][
        "ludvig_nvos_fern_released_all_view_exact_preflight_v1"
    ]

    provenance = official["common"]["gaussian_splatting"]["source_provenance"]
    assert provenance["ludvig_vendored_as_gitlink"] is False
    assert (
        provenance["official_commit"]
        == "f7a116fb1397d9842239127d39dc212f93171f70"
    )
    assert provenance["byte_identical_train_entrypoint"] is True
    assert preflight["status"] == "blocked_hardware_after_validated_cpu_dry_run"
    assert preflight["training"]["registered_views"] == 20
    assert preflight["training"]["held_out_views"] == 0
    assert preflight["training"]["defaults"]["feature_lr"] == pytest.approx(0.0025)
    assert preflight["source"]["algorithm_source_patch"] is None
    assert preflight["blocker"]["type"] == "gpu_hardware_lost"
    assert preflight["blocker"]["gpu_process_queued"] is False
    assert preflight["eligibility"]["exact_paper_protocol_comparison"] is False


def test_ludvig_spin_fern_preflight_reuses_exact_geometry_fail_closed() -> None:
    preflight = _registry()["protocols"][
        "ludvig_spin_fern_checkpoint_reuse_preflight_v1"
    ]

    assert preflight["checkpoint_reuse"][
        "can_reuse_exact_nvos_fern_all_view_30k"
    ] is True
    assert preflight["checkpoint_reuse"]["additional_training_required"] is False
    assert preflight["source_equivalence"]["raw_sparse_byte_identical"] is True
    assert preflight["source_equivalence"]["raw_rgb_byte_identical"] is True
    assert preflight["mapping"]["first"] == {
        "mask": "image000.png",
        "camera": "IMG_4026",
        "role": "reference",
    }
    assert preflight["mapping"]["scored_target_frames"] == 19
    assert preflight["calibration"]["sam_candidates"] == 3
    assert preflight["calibration"][
        "threshold_and_candidate_selected_on_reference"
    ] is True
    assert preflight["eligibility"]["local_9scene_macro"] is False
    assert preflight["eligibility"]["paper_10scene_row"] is False
