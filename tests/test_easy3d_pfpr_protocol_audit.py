import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from radio_gs.benchmarks.agile3d_scannet40.evaluate_easy3d import (
    NearestComplementIndex,
    aggregate_trajectory_rows,
    existing_scene_shard_matches,
    quantize_easy3d_scene,
    reference_cohort_audit,
)
from radio_gs.benchmarks.agile3d_scannet40.protocol import Agile3DObject
from radio_gs.benchmarks.agile3d_scannet40.summarize_easy3d_pilot import (
    COMMON_PROVENANCE_KEYS,
    summarize_reports,
)
from radio_gs.benchmarks.scannet_pfpr.audit_ludvig_style_uplift import (
    EXPECTED_METHOD,
    EXPECTED_QUERY_DESCRIPTOR,
    EXPECTED_READOUT,
    prediction_set_sha256,
    validate_style_cache,
)


def test_easy3d_quantization_is_sorted_and_last_input_row_wins() -> None:
    xyz = np.asarray(
        [
            [0.11, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.12, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    colors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([3, 4, 9], dtype=np.int32)
    scene = quantize_easy3d_scene(
        xyz, colors, labels, voxel_size=0.05, max_scene_size_m=40.0
    )

    assert scene.coordinates.tolist() == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert scene.voxel_labels.tolist() == [4, 9]
    assert scene.inverse_map.tolist() == [1, 0, 1]
    np.testing.assert_allclose(scene.features[1, 3:], [-1.0, -1.0, 1.0])
    assert scene.quantization_diagnostics["duplicate_voxel_count"] == 1
    assert (
        scene.quantization_diagnostics[
            "first_vs_last_label_difference_count"
        ]
        == 1
    )
    assert (
        scene.quantization_diagnostics[
            "mean_rgb_vs_last_l2_mean_duplicate_voxels"
        ]
        > 0
    )


def test_reused_kdtree_error_center_matches_brute_force() -> None:
    coordinates = np.asarray(
        [[x, y, 0.0] for x in range(7) for y in range(4)],
        dtype=np.float32,
    )
    error = np.zeros(len(coordinates), dtype=bool)
    error[[5, 6, 9, 10, 13, 14, 17, 18]] = True
    index = NearestComplementIndex(coordinates)
    point_index, radius = index.error_center(error)

    error_indices = np.flatnonzero(error)
    complement_indices = np.flatnonzero(~error)
    expected_distances = cdist(
        coordinates[error_indices], coordinates[complement_indices]
    ).min(axis=1)
    expected_local = int(np.argmax(expected_distances))
    assert point_index == int(error_indices[expected_local])
    assert radius == expected_distances[expected_local]


def test_next_click_uses_false_positive_on_exact_radius_tie() -> None:
    coordinates = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        dtype=np.float32,
    )
    prediction = np.asarray([True, False, False, False])
    target = np.asarray([False, False, False, True])

    assert NearestComplementIndex(coordinates).next_click(
        prediction, target, valid=None
    ) == (0, False)


def test_easy3d_aggregation_caps_noc_at_declared_maximum() -> None:
    rows = [
        {"trajectory": {"1": 0.4, "2": 0.7, "3": 0.9}},
        {"trajectory": {"1": 0.6, "2": 0.6, "3": 0.6}},
    ]
    metrics = aggregate_trajectory_rows(rows, max_clicks=3)

    assert metrics["IoU@1"] == 0.5
    assert metrics["IoU@2"] == 0.6499999999999999
    assert metrics["IoU@3"] == 0.75
    assert metrics["NoC@50"] == 1.5
    assert metrics["NoC@80"] == 3.0


def test_agile_reference_audit_exposes_silent_key_intersection(
    tmp_path: Path,
) -> None:
    objects = [
        Agile3DObject("scene0001_00", 1, "chair"),
        Agile3DObject("scene0001_00", 2, "table"),
    ]
    csv_path = tmp_path / "official.csv"
    csv_path.write_text(
        "\n".join(
            f"{row} 0001_00 {object_id} {click} {iou}"
            for click in range(0, 16)
            for row, object_id, iou in (
                (0, 0, 0.9),
                (1, 1, 0.8),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    audit = reference_cohort_audit(objects, csv_path)

    assert audit["release_object_count"] == 2
    assert audit["csv_object_count"] == 2
    assert audit["legacy_matched_object_count"] == 1
    assert audit["release_objects_silently_unmatched"] == 1
    assert audit["csv_objects_not_in_release_list"] == 1
    assert audit["bundled_csv_metrics_legacy_match"]["IoU@10"] == 0.8


def test_existing_easy3d_shard_resumes_only_exact_provenance(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "scene.json"
    expected = {
        "evaluator_schema_version": "v3",
        "interaction_contract": "agile3d_release",
        "object_keys": ["scene0001_00_obj_1"],
    }
    shard.write_text(
        json.dumps({**expected, "rows": [{"key": expected["object_keys"][0]}]}),
        encoding="utf-8",
    )
    assert existing_scene_shard_matches(shard, expected)

    incompatible = {**expected, "interaction_contract": "easy3d_released_code"}
    try:
        existing_scene_shard_matches(shard, incompatible)
    except ValueError as exc:
        assert "incompatible provenance" in str(exc)
    else:
        raise AssertionError("Easy3D shard accepted an incompatible contract")
    try:
        existing_scene_shard_matches(shard, expected, no_resume=True)
    except FileExistsError as exc:
        assert "overwrite is disabled" in str(exc)
    else:
        raise AssertionError("--no-resume silently authorized shard overwrite")


def test_easy3d_pilot_summary_freezes_source_grounded_contract_before_gaps(
    tmp_path: Path,
) -> None:
    common = {
        key: (
            False
            if key == "formal_selection"
            else True
            if key == "amp_bfloat16"
            else 10
            if key == "max_clicks"
            else 4
            if key == "object_batch_size"
            else 1
            if key in {"scene_count", "object_count"}
            else 0.05
            if key == "voxel_size_m"
            else 40.0
            if key == "max_scene_size_m"
            else {}
            if key == "runtime"
            else f"same-{key}"
        )
        for key in COMMON_PROVENANCE_KEYS
    }
    reports = {}
    paths = {}
    for contract, offset in (
        ("agile3d_release", -0.020),
        ("easy3d_released_code", -0.001),
    ):
        path = tmp_path / f"{contract}.json"
        path.write_text(contract, encoding="utf-8")
        paths[contract] = path
        reports[contract] = {
            "status": "declared_pilot_or_partial",
            "provenance": {
                **common,
                "interaction_contract": contract,
            },
            "cohorts": {
                "complete_release_selection": {
                    "object_count": 1,
                    "evaluated_object_count": 1,
                    "failed_object_count": 0,
                    "metrics_query_micro": {
                        "IoU@1": 0.682 + offset,
                        "IoU@2": 0.746 + offset,
                        "IoU@3": 0.773 + offset,
                        "IoU@5": 0.796 + offset,
                        "IoU@10": 0.817 + offset,
                    },
                }
            },
            "object_failures": [],
            "rows": [{"key": "scene0001_00_obj_1", "scene_id": "scene0001_00"}],
        }

    summary = summarize_reports(
        reports,
        report_paths=paths,
        elapsed_seconds={
            "agile3d_release": 1.0,
            "easy3d_released_code": 2.0,
        },
    )

    assert summary["primary_contract"] == "agile3d_release"
    assert summary["paper_gap_used_for_contract_selection"] is False
    assert "source-grounded" in summary["primary_contract_basis"]
    assert summary["pilot_object_count"] == 1
    assert (
        summary["agile3d_release_advantage_percentage_points"]["IoU@10"]
        < -1.8
    )


def _write_pfpr_style_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    benchmark = tmp_path / "benchmark"
    predictions = tmp_path / "predictions"
    benchmark.mkdir()
    predictions.mkdir()
    method_queries = [
        {
            "query_id": "scene0001_00_pfpr_000",
            "scene_id": "scene0001_00",
            "available_method_inputs": ["scene_id", "crop_rgb"],
        }
    ]
    (benchmark / "manifest.method.json").write_text(
        json.dumps(
            {
                "benchmark_version": "scannet-pfpr-small-v2",
                "queries": method_queries,
            }
        ),
        encoding="utf-8",
    )
    (benchmark / "manifest.public.json").write_text(
        json.dumps(
            {
                "benchmark_version": "scannet-pfpr-small-v2",
                "queries": [
                    {
                        "query_id": "scene0001_00_pfpr_000",
                        "scene_id": "scene0001_00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (benchmark / "manifest.evaluator.json").write_text(
        json.dumps(
            {
                "benchmark_version": "scannet-pfpr-small-v2",
                "queries": [
                    {
                        "query_id": "scene0001_00_pfpr_000",
                        "scene_id": "scene0001_00",
                        "anchor_world_xyz": [0.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    np.save(predictions / "scene0001_00_pfpr_000.npy", np.asarray([0.1, 0.2]))
    report_path = predictions / "prediction_report.json"
    report_path.write_text(
        json.dumps(
            {
                "benchmark": "scannet-pfpr-small-v2",
                "method": EXPECTED_METHOD,
                "protocol": {
                    "method_manifest_only": True,
                    "evaluator_anchor_opened": False,
                    "query_descriptor": EXPECTED_QUERY_DESCRIPTOR,
                    "crop_spatial_adapter": None,
                    "crop_context_adapter": None,
                    "primitive_score": "cosine",
                    "feature_calibration": "none",
                    "candidate_readout": EXPECTED_READOUT,
                    "support_gate_required": False,
                    "requires_official_extracted_capability_teachers": True,
                },
                "scene_reports": [{"scene_id": "scene0001_00"}],
            }
        ),
        encoding="utf-8",
    )
    return benchmark, predictions, report_path


def test_pfpr_cache_is_validated_only_as_ludvig_style(tmp_path: Path) -> None:
    benchmark, predictions, report_path = _write_pfpr_style_fixture(tmp_path)

    validation = validate_style_cache(benchmark, predictions, report_path)

    assert validation["query_count"] == 1
    assert validation["scene_names"] == ("scene0001_00",)
    assert validation["prediction_set_sha256"] == prediction_set_sha256(
        predictions
    )


def test_pfpr_style_audit_rejects_private_anchor_leak(tmp_path: Path) -> None:
    benchmark, predictions, report_path = _write_pfpr_style_fixture(tmp_path)
    method_path = benchmark / "manifest.method.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["queries"][0]["anchor_world_xyz"] = [0.0, 0.0, 0.0]
    method_path.write_text(json.dumps(method), encoding="utf-8")

    try:
        validate_style_cache(benchmark, predictions, report_path)
    except ValueError as exc:
        assert "leaks evaluator-private" in str(exc)
    else:
        raise AssertionError("PFPR audit accepted a private-anchor leak")
