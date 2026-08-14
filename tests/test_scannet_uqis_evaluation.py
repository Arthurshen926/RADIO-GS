from __future__ import annotations

import hashlib
import json
import numpy as np
from PIL import Image
import pytest

from radio_gs.benchmarks.scannet_uqis.metrics import evaluate_query_probabilities
from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION,
    UQISProtocolConfig,
    freeze_release,
)
from radio_gs.benchmarks.scannet_uqis.evaluate_predictions import (
    MODALITIES,
    evaluate,
    evaluate_release,
    evaluate_predictions,
    seal_prediction_batch,
)


def test_query_metrics_use_tie_aware_ap_and_frozen_threshold() -> None:
    probabilities = np.asarray([0.9, 0.4, 0.8, 0.1, 0.6], dtype=np.float64)
    instance_ids = np.asarray([1, 1, 2, 2, 0], dtype=np.int32)
    xyz = np.asarray(
        [[0, 0, 0], [0, 2, 0], [10, 0, 0], [10, 2, 0], [5, 0, 0]],
        dtype=np.float64,
    )

    metrics = evaluate_query_probabilities(
        probabilities,
        target_instance_id=1,
        same_class_distractor_instance_ids=[2],
        mesh_instance_ids=instance_ids,
        mesh_xyz=xyz,
    )

    assert metrics["average_precision"] == pytest.approx(0.75)
    assert metrics["oracle_iou"] == pytest.approx(0.5)
    assert metrics["fixed_iou_0.5"] == pytest.approx(0.25)
    assert metrics["acc_at_iou_0.25"] == 1.0
    assert metrics["acc_at_iou_0.50"] == 0.0
    assert metrics["selected_purity"] == pytest.approx(1.0 / 3.0)
    assert metrics["positive_coverage"] == pytest.approx(0.5)
    assert metrics["same_class_distractor_iou"] == pytest.approx(0.25)
    assert metrics["centroid_error_m"] == pytest.approx(np.sqrt(26.0))


def test_average_precision_does_not_break_score_ties_by_vertex_order() -> None:
    xyz = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    instance_ids = np.asarray([1, 2], dtype=np.int32)

    metrics = evaluate_query_probabilities(
        np.asarray([0.5, 0.5]),
        target_instance_id=1,
        same_class_distractor_instance_ids=[2],
        mesh_instance_ids=instance_ids,
        mesh_xyz=xyz,
    )

    assert metrics["average_precision"] == pytest.approx(0.5)


def _evaluation_fixture():
    target_specs = [
        ("target-1", "scene-1", 1, [2]),
        ("target-2", "scene-2", 3, [4]),
        ("target-3", "scene-2", 4, [3]),
    ]
    targets = []
    predictions = {}
    instance_ids = {
        "scene-1": np.asarray([1, 1, 2, 2], dtype=np.int32),
        "scene-2": np.asarray([3, 3, 4, 4, 0, 0], dtype=np.int32),
    }
    xyz = {
        scene: np.stack(
            [
                np.arange(len(values), dtype=np.float64),
                np.zeros(len(values)),
                np.zeros(len(values)),
            ],
            axis=1,
        )
        for scene, values in instance_ids.items()
    }
    for target_id, scene_id, instance_id, distractors in target_specs:
        queries = {}
        for modality in MODALITIES:
            query_id = f"{target_id}-{modality}"
            queries[modality] = query_id
            perfect = instance_ids[scene_id] == instance_id
            prediction = np.where(perfect, 0.9, 0.1).astype(np.float64)
            if modality == "text" and scene_id == "scene-2":
                prediction.fill(0.0)
            predictions[query_id] = prediction
        targets.append(
            {
                "target_id": target_id,
                "scene_id": scene_id,
                "instance_id": instance_id,
                "nyu40_class_id": 5,
                "same_class_distractor_instance_ids": distractors,
                "queries": queries,
            }
        )
    evaluator = {"benchmark_version": BENCHMARK_VERSION, "targets": targets}
    scenes = {
        "benchmark_version": BENCHMARK_VERSION,
        "scene_domains": [{"scene_id": "scene-1"}, {"scene_id": "scene-2"}],
    }
    return evaluator, scenes, predictions, xyz, instance_ids


def test_evaluator_reports_modality_scene_macro_and_uq_mean() -> None:
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()

    report = evaluate_predictions(
        evaluator,
        scenes,
        predictions,
        xyz,
        instance_ids,
        bootstrap_samples=32,
        bootstrap_seed=7,
    )

    assert report["query_count"] == 12
    assert report["target_count"] == 3
    assert report["scene_count"] == 2
    assert report["modalities"]["text"]["query_micro"]["fixed_iou_0.5"] == pytest.approx(1 / 3)
    assert report["modalities"]["text"]["scene_macro"]["fixed_iou_0.5"] == pytest.approx(0.5)
    for modality in ("image", "point_2d", "point_3d"):
        assert report["modalities"][modality]["scene_macro"]["fixed_iou_0.5"] == 1.0
    assert report["uq_mean"]["value"] == pytest.approx(0.875)
    assert report["uq_mean"]["scene_clustered_ci"]["estimate"] == pytest.approx(0.875)
    assert report["modalities"]["text"]["scene_clustered_ci"]["fixed_iou_0.5"][
        "bootstrap_unit"
    ] == "scene"
    assert "per_query" not in report


def test_evaluator_reports_core_rank_mask_and_separate_relational_text() -> None:
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()
    evaluator["targets"][0]["evaluation_tier"] = "unified_core"
    evaluator["targets"][1]["evaluation_tier"] = "unified_core"
    evaluator["targets"][2]["evaluation_tier"] = "relational_text_challenge"
    # Keep the common-modality cohort perfect while making only the relational
    # text query fail.  It must not lower UQ-Rank/UQ-Mask.
    for target in evaluator["targets"][:2]:
        scene_id = target["scene_id"]
        perfect = instance_ids[scene_id] == target["instance_id"]
        for query_id in target["queries"].values():
            predictions[query_id] = np.where(perfect, 0.9, 0.1).astype(np.float64)

    report = evaluate_predictions(
        evaluator,
        scenes,
        predictions,
        xyz,
        instance_ids,
        bootstrap_samples=32,
        bootstrap_seed=7,
    )

    assert report["core_target_count"] == 2
    assert report["core_modalities"]["text"]["query_count"] == 2
    assert report["core_modalities"]["image"]["scene_macro"]["average_precision"] == pytest.approx(1.0)
    assert report["relational_text_target_count"] == 1
    assert report["uq_rank"]["metric"] == "average_precision"
    assert report["uq_rank"]["value"] == pytest.approx(1.0)
    assert report["uq_mask"]["metric"] == "fixed_iou_0.5"
    assert report["uq_mask"]["value"] == pytest.approx(1.0)
    assert report["relational_text_challenge"]["query_count"] == 1
    assert report["relational_text_challenge"]["scene_macro"]["average_precision"] < 1.0
    assert "per_query" not in report


def test_evaluator_rejects_partial_or_unknown_tier_metadata() -> None:
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()
    evaluator["targets"][0]["evaluation_tier"] = "unified_core"
    with pytest.raises(ValueError, match="evaluation_tier"):
        evaluate_predictions(
            evaluator, scenes, predictions, xyz, instance_ids, bootstrap_samples=8
        )
    for target in evaluator["targets"]:
        target["evaluation_tier"] = "not-a-tier"
    with pytest.raises(ValueError, match="evaluation_tier"):
        evaluate_predictions(
            evaluator, scenes, predictions, xyz, instance_ids, bootstrap_samples=8
        )


def test_single_modality_comparator_has_no_uq_mean_or_pairing_key() -> None:
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()
    image_predictions = {
        query_id: value
        for query_id, value in predictions.items()
        if query_id.endswith("-image")
    }

    report = evaluate_predictions(
        evaluator,
        scenes,
        image_predictions,
        xyz,
        instance_ids,
        bootstrap_samples=8,
        modalities=("image",),
    )

    assert report["row_scope"] == "modality_comparator"
    assert report["evaluated_modalities"] == ["image"]
    assert report["uq_mean"] is None
    assert set(report["modalities"]) == {"image"}
    assert "per_query" not in report


@pytest.mark.parametrize("failure", ["shape", "nan", "range", "complex"])
def test_evaluator_rejects_invalid_probability_vectors(failure: str) -> None:
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()
    query_id = "target-1-text"
    if failure == "shape":
        predictions[query_id] = np.zeros(3, dtype=np.float64)
    elif failure == "nan":
        predictions[query_id][0] = np.nan
    elif failure == "range":
        predictions[query_id][0] = 1.01
    else:
        predictions[query_id] = predictions[query_id].astype(np.complex128)

    with pytest.raises(ValueError):
        evaluate_predictions(
            evaluator,
            scenes,
            predictions,
            xyz,
            instance_ids,
            bootstrap_samples=8,
        )


@pytest.mark.parametrize("failure", ["missing", "unexpected"])
def test_evaluator_requires_the_exact_prediction_inventory(failure: str) -> None:
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()
    if failure == "missing":
        predictions.pop("target-1-text")
    else:
        predictions["not-in-manifest"] = np.zeros(4, dtype=np.float64)

    with pytest.raises(ValueError, match="prediction set is incomplete"):
        evaluate_predictions(
            evaluator,
            scenes,
            predictions,
            xyz,
            instance_ids,
            bootstrap_samples=8,
        )


def test_evaluator_requires_every_modality_for_every_target() -> None:
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()
    evaluator["targets"][0]["queries"].pop("point_3d")

    with pytest.raises(ValueError, match="queries must contain exactly"):
        evaluate_predictions(
            evaluator,
            scenes,
            predictions,
            xyz,
            instance_ids,
            bootstrap_samples=8,
        )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_file_fixture(tmp_path):
    evaluator, scenes, predictions, xyz, instance_ids = _evaluation_fixture()
    private = tmp_path / "private"
    prediction_dir = tmp_path / "predictions"
    private.mkdir()
    prediction_dir.mkdir()
    private_scene_domains = []
    for scene_id in ("scene-1", "scene-2"):
        xyz_path = private / f"{scene_id}.xyz.npy"
        ids_path = private / f"{scene_id}.ids.npy"
        np.save(xyz_path, xyz[scene_id])
        np.save(ids_path, instance_ids[scene_id])
        private_scene_domains.append(
            {
                "scene_id": scene_id,
                "mesh_xyz_path": xyz_path.name,
                "mesh_xyz_sha256": _sha256(xyz_path),
                "mesh_instance_ids_path": ids_path.name,
                "mesh_instance_ids_sha256": _sha256(ids_path),
            }
        )
    evaluator["scene_domains"] = private_scene_domains
    scenes["scene_domains"] = [
        {
            key: value
            for key, value in record.items()
            if not key.startswith("mesh_instance_ids")
        }
        for record in private_scene_domains
    ]
    evaluator_path = private / "manifest.evaluator.json"
    scene_path = private / "manifest.scenes.json"
    evaluator_path.write_text(json.dumps(evaluator), encoding="utf-8")
    scene_path.write_text(json.dumps(scenes), encoding="utf-8")
    for query_id, probability in predictions.items():
        np.save(prediction_dir / f"{query_id}.npy", probability)
    return evaluator_path, scene_path, prediction_dir


def test_file_evaluator_verifies_mesh_bindings_and_writes_report(tmp_path) -> None:
    evaluator_path, scene_path, prediction_dir = _write_file_fixture(tmp_path)

    output = tmp_path / "report.json"
    report = evaluate(
        evaluator_path,
        scene_path,
        prediction_dir,
        output,
        bootstrap_samples=8,
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["query_count"] == 12
    assert report["uq_mean"]["value"] == pytest.approx(0.875)


@pytest.mark.parametrize("failure", ["mesh_hash", "unexpected_prediction"])
def test_file_evaluator_fails_closed_on_changed_inventory(tmp_path, failure: str) -> None:
    evaluator_path, scene_path, prediction_dir = _write_file_fixture(tmp_path)
    if failure == "mesh_hash":
        payload = json.loads(evaluator_path.read_text(encoding="utf-8"))
        payload["scene_domains"][0]["mesh_instance_ids_sha256"] = "0" * 64
        evaluator_path.write_text(json.dumps(payload), encoding="utf-8")
        match = "SHA-256 mismatch"
    else:
        np.save(prediction_dir / "unexpected.npy", np.zeros(4, dtype=np.float64))
        match = "prediction file inventory is incomplete"

    with pytest.raises(ValueError, match=match):
        evaluate(
            evaluator_path,
            scene_path,
            prediction_dir,
            bootstrap_samples=8,
        )


def test_file_evaluator_reads_freeze_release_manifests_without_translation(tmp_path) -> None:
    mesh_path = tmp_path / "mesh_xyz.npy"
    instance_path = tmp_path / "mesh_instances.npy"
    crop_path = tmp_path / "crop.png"
    mesh_xyz = np.asarray([[0, 0, 1], [0.1, 0, 1], [1, 0, 1]], dtype=np.float32)
    mesh_ids = np.asarray([1, 1, 2], dtype=np.int32)
    np.save(mesh_path, mesh_xyz)
    np.save(instance_path, mesh_ids)
    Image.new("RGB", (224, 224), (20, 30, 40)).save(crop_path)
    scene = {
        "scene_id": "scene0000_00",
        "mesh_xyz_path": str(mesh_path),
        "mesh_instance_ids_path": str(instance_path),
        "query_frame_ids": ["000020"],
        "withheld_frame_ids": ["000000", "000020", "000040"],
        "field_frame_ids": ["000100"],
        "max_query_frames": 3,
    }
    target = {
        "scene_id": "scene0000_00",
        "instance_id": 1,
        "nyu40_class_id": 5,
        "mesh_vertex_count": 501,
        "size_bucket": "medium",
        "same_class_distractor_instance_ids": [2],
        "query_frame_id": "000020",
        "expression": "the red chair",
        "expression_annotation_id": "a1",
        "expression_source": "nr3d",
        "expression_view_independent": True,
        "crop_rgb_path": str(crop_path),
        "camera_to_world": np.eye(4).tolist(),
        "camera_intrinsics": [[100.0, 0, 5], [0, 100.0, 5], [0, 0, 1]],
        "raster_size": [10, 10],
        "positive_pixel_uv": [5, 5],
        "click_depth_m": 1.0,
        "point_world_xyz": [0.0, 0.0, 1.0],
        "projection_pixels": 1000,
        "projection_fraction": 0.01,
        "projection_purity": 0.95,
        "field_surface_coverage": 0.75,
        "field_visibility_count": 5,
    }
    release = tmp_path / "release"
    freeze_release(
        [scene],
        [target],
        release,
        split_role="pilot",
        query_id_salt=b"0123456789abcdef",
        config=UQISProtocolConfig(
            min_targets_per_scene=1,
            min_same_class_targets_per_scene=1,
            min_semantic_categories_per_scene=1,
        ),
        allow_incomplete_pilot=True,
    )
    evaluator_payload = json.loads(
        (release / "target_manifest.evaluator.json").read_text(encoding="utf-8")
    )
    predictions = tmp_path / "release-predictions"
    predictions.mkdir()
    for query_id in evaluator_payload["targets"][0]["queries"].values():
        np.save(
            predictions / f"{query_id}.npy",
            np.asarray([0.9, 0.9, 0.1], dtype=np.float32),
        )

    report = evaluate(
        release / "target_manifest.evaluator.json",
        release / "scene_manifest.json",
        predictions,
        bootstrap_samples=8,
    )

    assert report["query_count"] == 4
    assert report["uq_mean"]["value"] == 1.0

    method_run = predictions / "run_manifest.json"
    method_run.write_text(
        json.dumps(
            {
                "schema_version": "synthetic_uqis_method_v1",
                "status": "complete",
                "benchmark_version": BENCHMARK_VERSION,
                "result_eligible": False,
                "formal_benchmark_row_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    sealed_path = tmp_path / "sealed.json"
    sealed = seal_prediction_batch(
        release,
        predictions,
        method_run,
        sealed_path,
        row_scope="universal_complete",
    )
    assert sealed["status"] == "sealed_before_private_evaluation"
    formal_method = json.loads(method_run.read_text(encoding="utf-8"))
    formal_method["formal_benchmark_row_eligible"] = True
    formal_method_path = predictions / "formal_run_manifest.json"
    formal_method_path.write_text(json.dumps(formal_method), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a bound field inventory"):
        seal_prediction_batch(
            release,
            predictions,
            formal_method_path,
            tmp_path / "formal-sealed.json",
            row_scope="universal_complete",
        )
    with pytest.raises(ValueError, match="disabled.*pilots"):
        evaluate_release(
            release,
            predictions,
            sealed_path,
            tmp_path / "sealed-report.json",
        )


def test_seal_requires_float32_and_evaluation_rejects_path_alias(
    tmp_path,
) -> None:
    mesh_path = tmp_path / "mesh_xyz.npy"
    instance_path = tmp_path / "mesh_instances.npy"
    crop_path = tmp_path / "crop.png"
    np.save(mesh_path, np.asarray([[0, 0, 1], [1, 0, 1]], dtype=np.float32))
    np.save(instance_path, np.asarray([1, 2], dtype=np.int32))
    Image.new("RGB", (224, 224), (1, 2, 3)).save(crop_path)
    release = tmp_path / "release"
    freeze_release(
        [
            {
                "scene_id": "scene0000_00",
                "mesh_xyz_path": str(mesh_path),
                "mesh_instance_ids_path": str(instance_path),
                "query_frame_ids": ["1"],
                "withheld_frame_ids": ["1"],
                "field_frame_ids": ["2"],
                "max_query_frames": 3,
            }
        ],
        [
            {
                "scene_id": "scene0000_00",
                "instance_id": 1,
                "nyu40_class_id": 5,
                "mesh_vertex_count": 501,
                "size_bucket": "medium",
                "same_class_distractor_instance_ids": [2],
                "query_frame_id": "1",
                "expression": "the chair",
                "expression_annotation_id": "a",
                "expression_source": "nr3d",
                "expression_view_independent": True,
                "crop_rgb_path": str(crop_path),
                "camera_to_world": np.eye(4).tolist(),
                "camera_intrinsics": [
                    [100.0, 0.0, 5.0],
                    [0.0, 100.0, 5.0],
                    [0.0, 0.0, 1.0],
                ],
                "raster_size": [10, 10],
                "positive_pixel_uv": [5, 5],
                "click_depth_m": 1.0,
                "point_world_xyz": [0.0, 0.0, 1.0],
                "projection_pixels": 1000,
                "projection_fraction": 0.01,
                "projection_purity": 0.95,
                "field_surface_coverage": 0.75,
                "field_visibility_count": 5,
            }
        ],
        release,
        split_role="pilot",
        query_id_salt=b"0123456789abcdef",
        config=UQISProtocolConfig(
            min_targets_per_scene=1,
            min_same_class_targets_per_scene=1,
            min_semantic_categories_per_scene=1,
        ),
        allow_incomplete_pilot=True,
    )
    evaluator = json.loads(
        (release / "target_manifest.evaluator.json").read_text(encoding="utf-8")
    )
    query_ids = sorted(evaluator["targets"][0]["queries"].values())
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for query_id in query_ids:
        np.save(
            predictions / f"{query_id}.npy",
            np.asarray([0.9, 0.1], dtype=np.float64),
        )
    method = predictions / "run_manifest.json"
    method.write_text(
        json.dumps(
            {
                "schema_version": "synthetic_uqis_method_v1",
                "status": "complete",
                "benchmark_version": BENCHMARK_VERSION,
                "result_eligible": False,
                "formal_benchmark_row_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="float32"):
        seal_prediction_batch(
            release,
            predictions,
            method,
            tmp_path / "bad-seal.json",
            row_scope="universal_complete",
        )

    for query_id in query_ids:
        np.save(
            predictions / f"{query_id}.npy",
            np.asarray([0.9, 0.1], dtype=np.float32),
        )
    seal_path = tmp_path / "seal.json"
    seal = seal_prediction_batch(
        release,
        predictions,
        method,
        seal_path,
        row_scope="universal_complete",
    )
    seal["predictions"][1]["relative_path"] = seal["predictions"][0][
        "relative_path"
    ]
    seal["predictions"][1]["bytes"] = seal["predictions"][0]["bytes"]
    seal["predictions"][1]["sha256"] = seal["predictions"][0]["sha256"]
    seal["predictions"][1]["dtype"] = seal["predictions"][0]["dtype"]
    seal["predictions"][1]["shape"] = seal["predictions"][0]["shape"]
    from radio_gs.benchmarks.scannet_uqis.protocol import canonical_json_sha256

    seal["prediction_inventory_sha256"] = canonical_json_sha256(
        seal["predictions"]
    )
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    original_release = json.loads((release / "release.json").read_text())
    original_release["formal_benchmark_eligible"] = True
    (release / "release.json").write_text(json.dumps(original_release))
    seal["release_json_sha256"] = _sha256(release / "release.json")
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="path does not match query identity"):
        evaluate_release(release, predictions, seal_path)
