import hashlib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "paper" / "artifacts" / "promptable_nvs_protocol_registry.yaml"


def _registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def test_ludvig_official_protocol_is_fail_closed_against_strict_unseen() -> None:
    protocol = _registry()["protocols"]["ludvig_official_online_multiview_v1"]

    assert protocol["common"]["foundation_model"]["checkpoint_sha256"] == (
        "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
    )
    assert (
        protocol["common"]["foundation_model"]["arbitrary_checkpoint_allowed"]
        is False
    )
    local = protocol["local_reproduction"]
    assert hashlib.sha256(
        (ROOT / local["launcher"]).read_bytes()
    ).hexdigest() == local["launcher_sha256"]
    assert hashlib.sha256(
        (ROOT / local["aggregator"]).read_bytes()
    ).hexdigest() == local["aggregator_sha256"]
    assert hashlib.sha256(
        (ROOT / local["reproduction_patch"]).read_bytes()
    ).hexdigest() == local["reproduction_patch_sha256"]
    assert local["reproduction_patch_sha256"] in local[
        "accepted_historical_patch_sha256"
    ]
    assert hashlib.sha256(
        (ROOT / local["all_view_trainer"]).read_bytes()
    ).hexdigest() == local["all_view_trainer_sha256"]
    assert hashlib.sha256(
        (ROOT / local["spin_llff_room_trainer"]).read_bytes()
    ).hexdigest() == local["spin_llff_room_trainer_sha256"]
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
    assert diagnostic["exact_completed_scenes"] == [
        "fern",
        "fortress",
        "horns",
        "leaves",
        "lego",
        "orchids",
        "pinecone",
        "room",
        "truck",
    ]
    assert diagnostic["exact_pending_available_scenes"] == []
    completed = diagnostic["completed_nine_scene_diagnostic"]
    assert completed[
        "local_scene_macro_iou_percent"
    ] == pytest.approx(93.7200449592385)
    assert completed["paper_same_scene_macro_iou_percent"] == pytest.approx(
        94.57777777777778
    )
    assert completed["delta_local_minus_paper_points"] == pytest.approx(
        -0.8577328185392759
    )
    assert completed[
        "eligible_for_matching_released_online_multiview_diagnostic"
    ] is True
    assert completed["eligible_for_9scene_macro"] is True
    assert completed["pinecone"]["run_values_iou_percent"] == pytest.approx(
        [81.40145559710756, 86.28110177736835, 86.13687631589627]
    )
    assert completed["pinecone"]["local_mean_iou_percent"] == pytest.approx(
        84.60647789679072
    )
    assert completed["pinecone"]["paper_iou_percent"] == pytest.approx(88.8)
    assert diagnostic["eligible_for_spin_nerf_10scene_table"] is False
    assert diagnostic["eligible_for_strict_unseen_claim"] is False
    assert diagnostic["artifacts"]["exact_available_cohort_summary_sha256"] == (
        "ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17"
    )
    assert diagnostic["artifacts"]["training_manifest_sha256_by_scene"] == {
        "fern": "8b7ebe7ff6d946c4ae03b1943835a4d73ee94b0e53d663a5933de03ea947909e",
        "fortress": "e01d76a52f0e0df0176e48301f9753ec610b475114f0449ab4cb634c6938aab3",
        "horns": "77154d2f5e761ba34a605677be8143214246ff8fbbf310b3400b55e4332488bc",
        "leaves": "2a610c0c356ae05f9a7d712c4352e032b137fa0794df3d701ec550759b67ba69",
        "lego": "145f2e8e3b10c36f085ffb6c389ec6e91f744310f88345e7da5add058593b5a4",
        "orchids": "8fc085e1c6ad817772898fd5e24f120b72ec78d3c0efa793a7ccbd761935d3d7",
        "pinecone": "2d1b5d8b14765f03dd0b84a3ed64f41d7dffcf2d776835a2d6270cf328eb816b",
        "room": "275c72ced3689cbca6b51cca8cc7830130fa6a1ac2d4eed3d471a346a7345dd4",
        "truck": "858ac2c6b6438ef609ac7bc82c701067a0117dc470159179326ed26929de2cd6",
    }
    assert diagnostic["artifacts"]["point_cloud_sha256_by_scene"] == {
        "fern": "4eaac89e70e00e4776e1bd9505b7560c483edd5df8ee1958c84b91091e49134b",
        "fortress": "723bfaa66c2e2da9c5e76662dc317d86fafe0dfed5c109b5e9fd6c11e8626c45",
        "horns": "1f3084daba5e9a70263152f73f704815f452a2f37e81f56cf3a3eb96c3e49803",
        "leaves": "da74e7c81debf212d0b3c6f7278a62259dfcc045f97401cb9370b1e62d340378",
        "lego": "544437bb194995502d3a8cfe8a74935944adb6b90b22de2a3d243656d9cc2183",
        "orchids": "e04a38f5aa22aaa716ec1c741222e1d1c94fb3cf0305aa4cc06a50a823ae5ef5",
        "pinecone": "27d5670cd642542cdba671a7a5718ae463b1097c437d2f4e232999090aef451e",
        "room": "c12a46a46ab3550905cfe48152916bd75834f5f36485ec4cf1cae03eaef294fd",
        "truck": "aa5b6d166277cde9d49632c9ffa427c392a4e52b7c68719c3d6ca312bedc3492",
    }
    assert diagnostic["artifacts"]["pinecone_run_manifest_sha256_by_seed"] == [
        "c6d17f651758e81804bed570cedf1eaa76f3e787c83cc2c136d24e8f9333c24c",
        "573029cfa39c63ac65b5c615f8d89a17e36fc68eccc7d86d4c6fecc0e38a2573",
        "de0d7bb7ce100e7d21b6554c0ae96c7dcc76c32c78cc279a82a8e8dd60130761",
    ]
    assert diagnostic["artifacts"][
        "pinecone_protocol_result_sha256_by_seed"
    ] == [
        "f51efdd892bfd936edeb36a1ffe0b4e6f718ff45d4121feb80482717c71cdee0",
        "0e432ceec545a61d6cb12d88b80d47127fd9f1ed64a182b38e973ff4c500978a",
        "88757c60385af7d2c5881b9ce72f92accc55db3cb9b8e153d5b7912f60d21c8e",
    ]
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


def test_ludvig_hybrid_cohort_pending_count_is_explicit_and_ineligible() -> None:
    cohort = _registry()["protocols"][
        "ludvig_nvos_hybrid_8scene_3seed_cohort_v1"
    ]

    assert cohort["status"] == "partially_complete_optional_nonpaper_diagnostic"
    assert cohort["expected_runs"] == 24
    assert cohort["complete_runs"] == 1
    assert cohort["pending_runs"] == 23
    assert cohort["completed_scene_seeds"] == [
        {"scene": "fern", "seed": 0, "attempt": "pinhole_stage_retry_1"}
    ]
    assert cohort["eligibility"]["exact_paper_protocol_comparison"] is False
    assert cohort["eligibility"]["strict_unseen_exact_match"] is False


def test_ludvig_exact_all_view_fern_is_completed_and_pinned() -> None:
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
    assert preflight["status"] == (
        "completed_checkpoint_and_exact_fern_three_seed_evaluation"
    )
    assert preflight["training"]["registered_views"] == 20
    assert preflight["training"]["held_out_views"] == 0
    assert preflight["training"]["defaults"]["feature_lr"] == pytest.approx(0.0025)
    assert preflight["source"]["algorithm_source_patch"] is None
    assert "blocker" not in preflight
    assert preflight["artifacts"]["training_manifest_sha256"] == (
        "8b7ebe7ff6d946c4ae03b1943835a4d73ee94b0e53d663a5933de03ea947909e"
    )
    assert preflight["artifacts"]["point_cloud_sha256"] == (
        "4eaac89e70e00e4776e1bd9505b7560c483edd5df8ee1958c84b91091e49134b"
    )
    assert preflight["completed_evaluation"]["local_mean_iou_percent"] == (
        pytest.approx(84.5072835444924)
    )
    assert preflight["eligibility"][
        "exact_per_scene_three_seed_paper_protocol_check"
    ] is True
    assert preflight["eligibility"]["exact_paper_protocol_comparison"] is False


def test_ludvig_exact_nvos_full_eight_task_result_is_complete_and_pinned() -> None:
    protocols = _registry()["protocols"]
    trex = protocols["ludvig_nvos_trex_released_all_view_exact_3seed_v1"]
    full = protocols["ludvig_nvos_released_all_view_full8_exact_3seed_v1"]

    assert trex["result"]["run_values_iou_percent"] == pytest.approx(
        [87.52962314809301, 87.67112176569724, 87.78877993071195]
    )
    assert trex["result"]["local_mean_iou_percent"] == pytest.approx(
        87.6631749481674
    )
    assert trex["result"]["fixed_threshold_parameter"] == 75
    assert trex["result"]["oracle_values_aggregated"] is False
    assert trex["failure_diagnosis"]["excluded_from_all_metrics"] is True
    assert trex["provenance"]["training_manifest_sha256"] == (
        "e8341f2ed5f77a48189790b3d61bdab0f1bea7efb45e41134403996bd4e10035"
    )
    assert trex["provenance"]["point_cloud_sha256"] == (
        "444c364891ff637a6f82b85fc1236f04a89f3ce7bd8b18bcf4806f88b5e0a7b0"
    )

    assert full["status"] == (
        "completed_exact_full_eight_task_three_seed_paper_protocol"
    )
    assert full["complete_runs"] == 24
    assert full["missing_tasks"] == []
    assert full["unique_geometry_checkpoints"] == 7
    assert full["result"]["local_task_macro_iou_percent"] == pytest.approx(
        91.25768502741802
    )
    assert full["result"][
        "delta_local_minus_paper_same_task_points"
    ] == pytest.approx(-0.07981497258198544)
    assert all(full["protocol_match_to_paper"].values())
    assert full["eligibility"]["exact_paper_protocol_comparison"] is True
    assert full["eligibility"]["strict_paper_protocol_table"] is True
    assert full["eligibility"]["strict_unseen_exact_match"] is False
    assert full["artifacts"]["summary_sha256"] == (
        "65e1f8e5c1f17083f66e5b7d4f6f03687f6806394c78a5cfce25546ca42e3546"
    )


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
    assert preflight["status"] == (
        "completed_reuse_and_exact_fern_three_seed_evaluation"
    )
    assert preflight["completed_evaluation"]["local_mean_iou_percent"] == (
        pytest.approx(97.04459720786971)
    )
    assert preflight["eligibility"]["single_scene_diagnostic"] is True
    assert preflight["eligibility"]["local_9scene_macro"] is False
    assert preflight["eligibility"]["paper_10scene_row"] is False


def test_ludvig_spin_room_resolution_safe_result_is_pinned_and_ineligible() -> None:
    room = _registry()["protocols"][
        "ludvig_spin_room_released_all_view_exact_3seed_v1"
    ]

    assert room["status"] == "completed_exact_per_scene_three_seed_check"
    assert room["result"]["run_values_iou_percent"] == pytest.approx(
        [97.12692191657342, 97.09182876192173, 97.34374774015015]
    )
    assert room["result"]["local_mean_iou_percent"] == pytest.approx(
        97.18749947288177
    )
    assert room["result"]["delta_local_minus_paper_points"] == pytest.approx(
        0.6874994728817683
    )
    assert room["failure_diagnosis"]["excluded_from_all_metrics"] is True
    assert room["resolution_safe_staging"]["center_crop_box"] == [
        0.5,
        0.0,
        4004.5,
        3003.0,
    ]
    assert room["resolution_safe_staging"]["source_dataset_modified"] is False
    assert room["provenance"]["training_manifest_sha256"] == (
        "275c72ced3689cbca6b51cca8cc7830130fa6a1ac2d4eed3d471a346a7345dd4"
    )
    assert room["provenance"]["point_cloud_sha256"] == (
        "c12a46a46ab3550905cfe48152916bd75834f5f36485ec4cf1cae03eaef294fd"
    )
    assert room["artifacts"]["summary_sha256"] == (
        "ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17"
    )
    assert room["eligibility"][
        "exact_per_scene_three_seed_paper_protocol_check"
    ] is True
    assert room["eligibility"][
        "released_online_multiview_protocol_match"
    ] is True
    assert room["eligibility"]["matching_nine_scene_available_cohort"] is True
    assert room["eligibility"]["local_9scene_macro"] is True
    assert room["eligibility"]["paper_10scene_row"] is False
    assert room["eligibility"]["strict_unseen_exact_match"] is False


def test_ludvig_spin_truck_exact_result_and_artifacts_are_pinned() -> None:
    truck = _registry()["protocols"][
        "ludvig_spin_truck_released_all_view_exact_3seed_v1"
    ]

    assert truck["status"] == "completed_exact_per_scene_three_seed_check"
    assert truck["result"]["run_values_iou_percent"] == pytest.approx(
        [96.7499675983893, 96.57112436400695, 96.81178718348382]
    )
    assert truck["result"]["local_mean_iou_percent"] == pytest.approx(
        96.71095971529336
    )
    assert truck["result"]["delta_local_minus_paper_points"] == pytest.approx(
        1.8109597152933503
    )
    assert truck["result"]["selected_threshold_parameter_per_seed"] == [
        87,
        87,
        87,
    ]
    assert truck["result"]["selected_sam_candidate_per_seed"] == [2, 2, 2]
    assert truck["result"]["reference_frames"] == 1
    assert truck["result"]["scored_target_frames"] == 64
    assert truck["provenance"]["registered_training_views"] == 251
    assert truck["provenance"]["held_out_training_views"] == 0
    assert truck["provenance"]["training_manifest_sha256"] == (
        "858ac2c6b6438ef609ac7bc82c701067a0117dc470159179326ed26929de2cd6"
    )
    assert truck["provenance"]["point_cloud_sha256"] == (
        "aa5b6d166277cde9d49632c9ffa427c392a4e52b7c68719c3d6ca312bedc3492"
    )
    assert truck["artifacts"]["summary_sha256"] == (
        "ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17"
    )
    assert truck["eligibility"][
        "exact_per_scene_three_seed_paper_protocol_check"
    ] is True
    assert truck["eligibility"][
        "released_online_multiview_protocol_match"
    ] is True
    assert truck["eligibility"]["matching_nine_scene_available_cohort"] is True
    assert truck["eligibility"]["local_9scene_macro"] is True
    assert truck["eligibility"]["paper_10scene_row"] is False
    assert truck["eligibility"]["strict_unseen_exact_match"] is False
