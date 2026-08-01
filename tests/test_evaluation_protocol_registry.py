from copy import deepcopy
from pathlib import Path

import pytest

from radio_gs.scripts.validate_evaluation_protocol_registry import (
    RegistryError,
    load_and_validate,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def _valid_registry() -> dict:
    return {
        "schema_version": 1,
        "reporting_policy": {
            "oracle_metrics_are_diagnostic_only": True,
            "incomplete_cohorts_are_strictly_comparable": False,
        },
        "evaluations": {
            "example": {
                "benchmark_family": "example",
                "task": "segmentation",
                "method": "method",
                "completion": "complete",
                "evidence_class": "exact_reproduction",
                "cohort": {"complete": True, "items": 4},
                "protocol": {
                    "aggregation": "scene macro",
                    "metric_domain": "pixels",
                    "calibration": "fixed before test",
                },
                "comparability": {
                    "paper_comparison": "strict",
                    "strict_table_eligible": True,
                    "reasons": [],
                    "protocol_match_to_paper": {
                        "cohort": True,
                        "prompt_or_query": True,
                        "target_visibility": True,
                        "metric_domain": True,
                        "aggregation": True,
                        "calibration": True,
                        "implementation": True,
                    },
                },
                "reported_metrics": [
                    {
                        "name": "miou_percent",
                        "role": "primary",
                        "local": 51.0,
                        "paper": 50.0,
                        "delta_points": 1.0,
                    }
                ],
            }
        },
    }


def test_valid_exact_registry_row_passes():
    validate_registry(_valid_registry())


def test_incomplete_cohort_cannot_be_strict():
    payload = _valid_registry()
    payload["evaluations"]["example"]["cohort"]["complete"] = False
    with pytest.raises(RegistryError, match="complete cohort"):
        validate_registry(payload)


def test_protocol_mismatch_cannot_be_strict():
    payload = _valid_registry()
    payload["evaluations"]["example"]["comparability"][
        "protocol_match_to_paper"
    ]["aggregation"] = False
    with pytest.raises(RegistryError, match="aggregation"):
        validate_registry(payload)


def test_oracle_must_never_select_reported_metric():
    payload = _valid_registry()
    payload["evaluations"]["example"]["oracle_diagnostics"] = {
        "diagnostic_only": True,
        "used_for_reported_metric": True,
        "used_for_model_or_threshold_selection": False,
    }
    with pytest.raises(RegistryError, match="used_for_reported_metric"):
        validate_registry(payload)


def test_diagnostic_row_is_allowed_when_fail_closed():
    payload = _valid_registry()
    row = payload["evaluations"]["example"]
    row["evidence_class"] = "diagnostic"
    row["comparability"]["paper_comparison"] = "diagnostic_only"
    row["comparability"]["strict_table_eligible"] = False
    row["comparability"]["protocol_match_to_paper"]["target_visibility"] = False
    row["comparability"]["reasons"] = ["target RGB is visible at query time"]
    row["oracle_diagnostics"] = {
        "diagnostic_only": True,
        "used_for_reported_metric": False,
        "used_for_model_or_threshold_selection": False,
        "best_iou_percent": 52.0,
    }
    validate_registry(payload)


def test_forbidden_comparison_cannot_hide_paper_delta():
    payload = _valid_registry()
    row = payload["evaluations"]["example"]
    row["evidence_class"] = "diagnostic"
    row["comparability"]["paper_comparison"] = "forbidden"
    row["comparability"]["strict_table_eligible"] = False
    row["comparability"]["reasons"] = ["the benchmark has no matching paper task"]
    with pytest.raises(RegistryError, match="cannot contain paper deltas"):
        validate_registry(payload)


def test_metric_delta_is_verified():
    payload = deepcopy(_valid_registry())
    payload["evaluations"]["example"]["reported_metrics"][0]["delta_points"] = 0.5
    with pytest.raises(RegistryError, match="does not equal"):
        validate_registry(payload)


def test_checked_in_cross_benchmark_registry_is_fail_closed():
    payload = load_and_validate(
        ROOT / "paper" / "artifacts" / "evaluation_protocol_registry_20260731.yaml"
    )
    rows = payload["evaluations"]

    lerf = rows["lerf2d_langsplatv2_exact_camera_20260731"]
    assert lerf["cohort"]["labelled_camera_roles"] == {"train": 15, "test": 7}
    assert lerf["cohort"]["labelled_query_roles"] == {"train": 134, "test": 74}
    assert lerf["evidence_class"] == "compatibility_reproduction"
    assert lerf["comparability"]["paper_comparison"] == "diagnostic_only"
    assert lerf["comparability"]["strict_table_eligible"] is False
    assert lerf["comparability"]["protocol_match_to_paper"]["cohort"] is False
    assert (
        lerf["comparability"]["protocol_match_to_paper"]["aggregation"] is False
    )
    assert "row-specific numerical match" in lerf["protocol"]["aggregation"]
    assert "aggregation-heterogeneous" in lerf["protocol"]["aggregation"]
    assert (
        lerf["diagnosis"]["dominant_remaining_gap_class"]
        == "checkpoint_training_and_feature_provenance"
    )
    assert (
        lerf["diagnosis"][
            "known_remaining_evaluator_protocol_error_explains_locacc_gap"
        ]
        is False
    )
    assert lerf["artifacts"]["final_gap_receipt_sha256"] == (
        "0bda52d7f8598c731817d2a2e69e62a043e5ab2f16de91840ccecbd11c4543d2"
    )
    lerf_metrics = {
        metric["name"]: metric for metric in lerf["reported_metrics"]
    }
    assert lerf_metrics["localization_hits"]["value"] == 166
    assert lerf_metrics["nearest_paper_scene_row_integer_hit_reconstruction"][
        "value"
    ] == 175
    assert lerf_metrics["nearest_paper_hit_deficit"]["value"] == 9

    drsplat = rows["lerf3d_drsplat_compatibility_20260519"]
    assert drsplat["completion"] == "complete"
    assert drsplat["evidence_class"] == "compatibility_reproduction"
    assert drsplat["strict_checkpoint_reproduction"] is False
    assert drsplat["cohort"]["complete"] is True
    assert drsplat["cohort"]["queries"] == 208
    assert drsplat["cohort"]["missing_predictions"] == 0
    assert drsplat["comparability"]["paper_comparison"] == "diagnostic_only"
    assert drsplat["comparability"]["strict_table_eligible"] is False
    assert (
        drsplat["protocol"]["aggregation"]
        == "paper deltas use an unweighted equal macro over four scenes; "
        "208-query micro is stored separately as a secondary diagnostic"
    )
    assert "feature_level=3" in drsplat["protocol"]["calibration"]
    assert drsplat["diagnosis"]["dominant_corrected_protocol_factor"] == (
        "feature_level_scale"
    )
    assert drsplat["diagnosis"]["scale_is_material"] is True
    assert (
        drsplat["diagnosis"]["scale_is_sufficient_to_reproduce_paper"]
        is False
    )
    assert drsplat["diagnosis"][
        "scene_delta_points_miou_accuracy_at_iou_0p25"
    ]["figurines"] == pytest.approx([23.7510031651306, 33.92857142857143])
    assert drsplat["diagnosis"][
        "scene_delta_points_miou_accuracy_at_iou_0p25"
    ]["teatime"] == pytest.approx([26.480341730352364, 45.76271186440678])
    assert drsplat["diagnosis"][
        "scene_delta_points_miou_accuracy_at_iou_0p25"
    ]["ramen"] == pytest.approx([0.35767141728785006, 1.4084507042253502])
    assert drsplat["diagnosis"][
        "scene_delta_points_miou_accuracy_at_iou_0p25"
    ]["waldo_kitchen"] == pytest.approx(
        [9.068395421143164, 18.181818181818183]
    )
    assert drsplat["diagnosis"][
        "scene_delta_points_accuracy_at_iou_0p5"
    ]["ramen"] == pytest.approx(-4.225352112676058)
    drsplat_metrics = {
        metric["name"]: metric for metric in drsplat["reported_metrics"]
    }
    assert drsplat_metrics["miou_percent"]["local"] == pytest.approx(
        32.53169394351046
    )
    assert drsplat_metrics["miou_percent"]["delta_points"] == pytest.approx(
        -10.758306056489538
    )
    assert drsplat_metrics[
        "accuracy_at_iou_0p25_percent"
    ]["local"] == pytest.approx(50.43195420597545)
    assert drsplat_metrics[
        "accuracy_at_iou_0p25_percent"
    ]["delta_points"] == pytest.approx(-13.868045794024546)
    assert drsplat_metrics[
        "l1_miou_scene_macro_percent"
    ]["value"] == pytest.approx(17.617341010031962)
    assert drsplat_metrics[
        "l1_to_l3_miou_scene_macro_delta_points"
    ]["value"] == pytest.approx(14.914352933478498)
    assert drsplat_metrics[
        "l1_to_l3_accuracy_at_iou_0p25_scene_macro_delta_points"
    ]["value"] == pytest.approx(24.82038804475543)
    assert drsplat_metrics[
        "accuracy_at_iou_0p5_scene_macro_percent"
    ]["value"] == pytest.approx(30.833983097351442)
    assert drsplat_metrics["miou_query_micro_percent"]["value"] == pytest.approx(
        32.50508397390278
    )
    assert drsplat_metrics[
        "accuracy_at_iou_0p25_query_micro_percent"
    ]["value"] == pytest.approx(51.442307692307686)
    assert drsplat["artifacts"]["result_sha256"] == (
        "39c849e45ee4fcca53fd5977ddfc37d15917ef954e47fe654d012c349cc39aca"
    )
    assert drsplat["artifacts"]["legacy_l1_result_sha256"] == (
        "9196611f5cdf24edfbd803e74f7a6164338eb4825e849b93ca77ffd15385c8cd"
    )

    scannet_p0 = rows[
        "scannet_ovs_vala_gaussian_vs_mesh_p0_2scene_20260801"
    ]
    assert scannet_p0["completion"] == "complete"
    assert scannet_p0["cohort"]["scenes"] == [
        "scene0000_00",
        "scene0400_00",
    ]
    assert scannet_p0["cohort"]["full_vala8_complete"] is False
    assert scannet_p0["comparability"]["paper_comparison"] == "forbidden"
    assert scannet_p0["comparability"]["strict_table_eligible"] is False
    p0_metrics = {
        metric["name"]: metric["value"]
        for metric in scannet_p0["reported_metrics"]
    }
    assert p0_metrics[
        "split10_gaussian_domain_two_scene_miou_percent"
    ] == pytest.approx(16.3306876202596)
    assert p0_metrics[
        "split10_mesh_domain_two_scene_miou_percent"
    ] == pytest.approx(14.457832793231376)
    assert p0_metrics[
        "split10_gaussian_minus_mesh_miou_points"
    ] == pytest.approx(1.8728548270282225)
    assert p0_metrics[
        "split10_gaussian_minus_mesh_macc_points"
    ] == pytest.approx(0.8605546524176111)
    assert scannet_p0["artifacts"]["result_sha256"] == (
        "3ea3e2a554e1fe639e04c45f7aafd505ea86dfe95b526a63c487192733ea7c04"
    )

    easy_primary = rows["agile3d_easy3d_agile_release_full312_20260731"]
    easy_sensitivity = rows[
        "agile3d_easy3d_released_code_full312_sensitivity_20260731"
    ]
    for row in (easy_primary, easy_sensitivity):
        assert row["completion"] == "complete"
        assert row["cohort"]["complete"] is True
        assert row["cohort"]["scenes"] == 312
        assert row["cohort"]["evaluated_objects"] == 10357
        assert row["cohort"]["failed_objects"] == 0
        assert row["comparability"]["paper_comparison"] == "diagnostic_only"
        assert row["comparability"]["strict_table_eligible"] is False
    assert easy_primary["protocol"]["interaction_contract"] == "agile3d_release"
    assert (
        easy_sensitivity["protocol"]["interaction_contract"]
        == "easy3d_released_code"
    )
    easy_primary_metrics = {
        metric["name"]: metric for metric in easy_primary["reported_metrics"]
    }
    easy_sensitivity_metrics = {
        metric["name"]: metric for metric in easy_sensitivity["reported_metrics"]
    }
    assert easy_primary_metrics[
        "iou_at_10_query_micro_percent"
    ]["local"] == pytest.approx(
        83.03839027824663
    )
    assert easy_sensitivity_metrics[
        "iou_at_10_query_micro_percent"
    ]["local"] == pytest.approx(82.13209216046787)
    assert (
        easy_primary["artifacts"]["result_sha256"]
        == "c771f29400e912565ee2ea5a754d0fd80a7fafc1eb91e38b6db1953cdcbbc09d"
    )
    assert (
        easy_sensitivity["artifacts"]["result_sha256"]
        == "c8c0c6820cf88166e77002a20ddf68af44c7be2e58a2e36cd26b147bdbd2f5a1"
    )

    nvos_fern = rows[
        "nvos_ludvig_fern_released_all_view_3seed_exact_20260731"
    ]
    spin_fern = rows[
        "spin_nerf_ludvig_fern_released_all_view_3seed_exact_20260731"
    ]
    assert nvos_fern["reported_metrics"][0]["local"] == pytest.approx(
        84.5072835444924
    )
    assert spin_fern["reported_metrics"][0]["local"] == pytest.approx(
        97.04459720786971
    )
    assert nvos_fern["cohort"]["full_benchmark_complete"] is True
    assert nvos_fern["cohort"]["missing_tasks"] == []
    assert spin_fern["cohort"]["full_benchmark_complete"] is False
    for row in (nvos_fern, spin_fern):
        assert row["cohort"]["complete"] is True
        assert row["comparability"]["paper_comparison"] == "diagnostic_only"
        assert row["comparability"]["strict_table_eligible"] is False
        assert row["artifacts"]["training_manifest_sha256"] == (
            "8b7ebe7ff6d946c4ae03b1943835a4d73ee94b0e53d663a5933de03ea947909e"
        )
        assert row["artifacts"]["point_cloud_sha256"] == (
            "4eaac89e70e00e4776e1bd9505b7560c483edd5df8ee1958c84b91091e49134b"
        )
        assert row["artifacts"][
            "cross_benchmark_same_training_manifest_and_ply"
        ] is True
    completed_spin_scene_ids = [
        "spin_nerf_ludvig_fern_released_all_view_3seed_exact_20260731",
        "spin_nerf_ludvig_fortress_released_all_view_3seed_exact_20260731",
        "spin_nerf_ludvig_horns_released_all_view_3seed_exact_20260731",
        "spin_nerf_ludvig_leaves_released_all_view_3seed_exact_20260731",
        "spin_nerf_ludvig_orchids_released_all_view_3seed_exact_20260731",
        "spin_nerf_ludvig_room_released_all_view_3seed_exact_20260731",
    ]
    expected_available_scenes = [
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
    for row_id in completed_spin_scene_ids:
        row = rows[row_id]
        assert row["cohort"]["full_benchmark_completed_scenes"] == (
            expected_available_scenes
        )
        assert row["cohort"]["missing_available_scenes"] == []
        assert row["cohort"]["unavailable_scenes"] == ["fork"]
        assert row["artifacts"][
            "latest_available_cohort_summary_sha256"
        ] == "ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17"

    spin_room = rows[
        "spin_nerf_ludvig_room_released_all_view_3seed_exact_20260731"
    ]
    assert spin_room["completion"] == "complete"
    assert spin_room["cohort"]["complete"] is True
    assert spin_room["reported_metrics"][0]["local"] == pytest.approx(
        97.18749947288177
    )
    assert spin_room["reported_metrics"][0]["delta_points"] == pytest.approx(
        0.6874994728817683
    )
    assert spin_room["artifacts"]["training_manifest_sha256"] == (
        "275c72ced3689cbca6b51cca8cc7830130fa6a1ac2d4eed3d471a346a7345dd4"
    )
    assert spin_room["artifacts"]["point_cloud_sha256"] == (
        "c12a46a46ab3550905cfe48152916bd75834f5f36485ec4cf1cae03eaef294fd"
    )
    assert spin_room["artifacts"]["fractional_center_crop_box"] == [
        0.5,
        0.0,
        4004.5,
        3003.0,
    ]
    assert spin_room["failure_diagnosis"]["excluded_from_all_metrics"] is True
    assert spin_room["artifacts"]["latest_available_cohort_summary_sha256"] == (
        "ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17"
    )
    assert spin_room["current_available_cohort_context"][
        "local_9scene_complete"
    ] is True
    assert spin_room["current_available_cohort_context"][
        "full_10scene_complete"
    ] is False

    trex = rows["nvos_ludvig_trex_released_all_view_3seed_exact_20260731"]
    assert trex["completion"] == "complete"
    assert trex["cohort"]["full_benchmark_complete"] is True
    assert trex["reported_metrics"][0]["local"] == pytest.approx(
        87.6631749481674
    )
    assert trex["reported_metrics"][0]["delta_points"] == pytest.approx(
        -0.336825051832605
    )
    assert trex["artifacts"]["training_manifest_sha256"] == (
        "e8341f2ed5f77a48189790b3d61bdab0f1bea7efb45e41134403996bd4e10035"
    )
    assert trex["artifacts"]["point_cloud_sha256"] == (
        "444c364891ff637a6f82b85fc1236f04a89f3ce7bd8b18bcf4806f88b5e0a7b0"
    )
    assert trex["artifacts"]["failed_v1_run_excluded_from_metrics"].endswith(
        "exact_f7a_allview_trex_seed0_v1/run_manifest.json"
    )

    nvos_full = rows[
        "nvos_ludvig_released_all_view_full8_3seed_exact_20260731"
    ]
    assert nvos_full["completion"] == "complete"
    assert nvos_full["cohort"]["complete"] is True
    assert nvos_full["cohort"]["full_benchmark_complete"] is True
    assert nvos_full["cohort"]["completed_runs"] == 24
    assert nvos_full["cohort"]["unique_geometry_checkpoints"] == 7
    assert nvos_full["comparability"]["paper_comparison"] == "strict"
    assert nvos_full["comparability"]["strict_table_eligible"] is True
    assert all(
        nvos_full["comparability"]["protocol_match_to_paper"].values()
    )
    full_metrics = {
        metric["name"]: metric for metric in nvos_full["reported_metrics"]
    }
    assert full_metrics[
        "full_eight_task_macro_iou_percent_vs_paper_headline"
    ]["local"] == pytest.approx(91.25768502741802)
    assert full_metrics[
        "full_eight_task_macro_iou_percent_vs_recomputed_paper_scene_macro"
    ]["delta_points"] == pytest.approx(-0.07981497258198544)
    assert nvos_full["protocol"]["strict_unseen_exact_match"] is False
    assert nvos_full["artifacts"]["summary_sha256"] == (
        "65e1f8e5c1f17083f66e5b7d4f6f03687f6806394c78a5cfce25546ca42e3546"
    )

    spin_partial = rows["spin_nerf_ludvig_9scene_missing_fork_20260731"]
    assert spin_partial["completion"] == "partial"
    assert spin_partial["cohort"]["complete"] is False
    assert spin_partial["cohort"]["locally_available_cohort_complete"] is True
    assert spin_partial["cohort"]["full_10scene_benchmark_complete"] is False
    assert spin_partial["cohort"]["exact_completed_scenes"] == [
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
    assert spin_partial["cohort"]["exact_pending_available_scenes"] == []
    assert spin_partial["comparability"]["paper_comparison"] == "diagnostic_only"
    assert spin_partial["comparability"]["strict_table_eligible"] is False
    spin_nine = spin_partial["completed_nine_scene_diagnostic"]
    assert spin_nine[
        "local_scene_macro_iou_percent"
    ] == pytest.approx(93.7200449592385)
    assert spin_nine["paper_same_scene_macro_iou_percent"] == pytest.approx(
        94.57777777777778
    )
    assert spin_nine["delta_local_minus_paper_points"] == pytest.approx(
        -0.8577328185392759
    )
    assert spin_nine["eligible_for_matching_9scene_diagnostic"] is True
    assert spin_nine["eligible_for_full_10scene_table"] is False
    assert spin_nine["pinecone"]["run_values_iou_percent"] == pytest.approx(
        [81.40145559710756, 86.28110177736835, 86.13687631589627]
    )
    assert spin_nine["pinecone"]["local_mean_iou_percent"] == pytest.approx(
        84.60647789679072
    )
    assert spin_partial["artifacts"][
        "exact_available_cohort_summary_sha256"
    ] == "ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17"
    assert spin_partial["artifacts"]["pinecone_training_manifest_sha256"] == (
        "2d1b5d8b14765f03dd0b84a3ed64f41d7dffcf2d776835a2d6270cf328eb816b"
    )
    assert spin_partial["artifacts"]["pinecone_point_cloud_sha256"] == (
        "27d5670cd642542cdba671a7a5718ae463b1097c437d2f4e232999090aef451e"
    )
    assert spin_partial["artifacts"][
        "pinecone_run_manifest_sha256_by_seed"
    ] == [
        "c6d17f651758e81804bed570cedf1eaa76f3e787c83cc2c136d24e8f9333c24c",
        "573029cfa39c63ac65b5c615f8d89a17e36fc68eccc7d86d4c6fecc0e38a2573",
        "de0d7bb7ce100e7d21b6554c0ae96c7dcc76c32c78cc279a82a8e8dd60130761",
    ]
    assert spin_partial["artifacts"][
        "pinecone_protocol_result_sha256_by_seed"
    ] == [
        "f51efdd892bfd936edeb36a1ffe0b4e6f718ff45d4121feb80482717c71cdee0",
        "0e432ceec545a61d6cb12d88b80d47127fd9f1ed64a182b38e973ff4c500978a",
        "88757c60385af7d2c5881b9ce72f92accc55db3cb9b8e153d5b7912f60d21c8e",
    ]
    assert spin_partial["artifacts"][
        "pinecone_failed_seed0_v1_excluded_from_metrics"
    ] is True
    assert spin_partial["artifacts"][
        "pinecone_failed_seed0_v1_manifest_sha256"
    ] == "55903ae10bdf37c61a514979facf9a517d1a15088bdea340251251989ce0c678"
    assert (
        rows["pfpr_ludvig_style_partial6_20260731"]["comparability"][
            "paper_comparison"
        ]
        == "forbidden"
    )
    for row in rows.values():
        if row["comparability"]["paper_comparison"] != "strict":
            assert row["comparability"]["strict_table_eligible"] is False
