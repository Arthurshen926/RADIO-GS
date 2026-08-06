from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radio_gs.evaluation.promptable_segmentation import compute_protocol_hash
from radio_gs.scripts.score_nvos_sealed_prediction_batch import _score_frame
from radio_gs.scripts.score_registered_evidence_external_prediction import (
    _score_external_frame,
    score_verified_prediction,
    verify_prediction_barrier,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    sha256_file,
    write_frozen_json,
)


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _dummy(path: Path, value: bytes) -> dict[str, str]:
    path.write_bytes(value)
    return file_record(path)


def _protocol(benchmark: str) -> dict:
    return {
        "benchmark": benchmark,
        "dataset_version": "synthetic-frozen-v1",
        "task": "promptable_nvs_binary_segmentation",
        "prompt_type": (
            "foreground_background_scribbles"
            if benchmark == "NVOS"
            else "single_reference_binary_mask"
        ),
        "metrics": ["foreground_iou", "pixel_accuracy"],
        "aggregation": "per_frame_then_per_scene_then_dataset_scene_macro",
        "resize": "nearest",
        "prediction_representation": "continuous_margin",
        "threshold_comparison": "greater_or_equal",
        "empty_union_value": 1.0,
        "allow_reference_scoring": False,
        "threshold": {"mode": "fixed", "value": 0.0},
        "score_semantics": "cosine_similarity_foreground_minus_background",
        "score_temperature": "none",
    }


def _synthetic_bundle(tmp_path: Path, *, benchmark: str = "NVOS") -> dict:
    gt_values = {
        "target0": np.asarray([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),
        "target1": np.asarray([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
    }
    gt_records: dict[str, dict[str, str]] = {}
    for frame_id, value in gt_values.items():
        path = tmp_path / "gt" / f"{frame_id}.npy"
        _save_npy(path, value)
        gt_records[frame_id] = file_record(path)

    raw_manifest = {
        "schema_version": 1,
        "protocol": _protocol(benchmark),
        "scenes": [
            {
                "scene_id": "scene",
                "prompt_frame_ids": ["source"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target0", "target1"],
                "frames": [
                    {"frame_id": "source", "ground_truth": None},
                    *[
                        {
                            "frame_id": frame_id,
                            "ground_truth": record["path"],
                            "ground_truth_sha256": record["sha256"],
                        }
                        for frame_id, record in gt_records.items()
                    ],
                ],
            }
        ],
    }
    raw_manifest["protocol_hash"] = compute_protocol_hash(raw_manifest)
    manifest_path = tmp_path / "manifest.json"
    write_frozen_json(manifest_path, raw_manifest)
    manifest_record = file_record(manifest_path)

    upstream: dict[str, dict[str, str]] = {}
    for name in (
        "source_w",
        "source_report",
        "positive",
        "capability",
        "state",
        "v1",
        "v2",
        "config",
        "carrier",
        "camera_map",
        "v1_result",
        "primitive",
        "adapter_core",
        "materializer",
    ):
        upstream[name] = _dummy(tmp_path / name, name.encode("utf-8"))

    v2_source_result = {
        "promotion_gate_passed": True,
        "decision": "eligible_for_cross_scene_confirmation",
        "graph": "off",
        "connected_selection": "off",
    }
    v2_source_path = tmp_path / "v2_result.json"
    write_frozen_json(v2_source_path, v2_source_result)
    upstream["v2_result"] = file_record(v2_source_path)
    cross_scene_result = {
        "status": "cross_scene_clean_confirmation_complete",
        "scene_id": "scene0002_00",
        "fit_or_parameter_update": False,
        "candidate_eligibility": {
            "V2": {
                "eligible": True,
                "reason": (
                    "passed_original_whole_gate_and_independent_cross_scene_gate"
                ),
            },
            "V3": {"eligible": False},
            "V4": {"eligible": False},
        },
        "source_gate_states": {
            "V2": {
                "promotion_gate_passed": True,
                "decision": "eligible_for_cross_scene_confirmation",
            }
        },
        "source_access": {
            "scene0002_labels_enter_fit": False,
            "scene0002_target_rgb_opened": False,
            "per_scene_tuning": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    cross_result_path = tmp_path / "cross_scene_result.json"
    write_frozen_json(cross_result_path, cross_scene_result)
    upstream["cross_scene_confirmation_result"] = file_record(cross_result_path)
    cross_scene_receipt = {
        "schema": (
            "radio_gs.prompt_unary." "cross_scene_clean_confirmation_result_receipt.v1"
        ),
        "status": "scene0002_independent_confirmation_complete",
        "scene_id": "scene0002_00",
        "decision": {
            "V2": (
                "only existing candidate eligible to continue: original whole gate "
                "and independent scene0002 macro cross-scene gate both pass"
            )
        },
        "artifacts": {"result": upstream["cross_scene_confirmation_result"]},
        "source_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
            "per_scene_tuning": False,
        },
    }
    cross_receipt_path = tmp_path / "cross_scene_receipt.json"
    write_frozen_json(cross_receipt_path, cross_scene_receipt)
    upstream["cross_scene_confirmation_receipt"] = file_record(cross_receipt_path)
    authority = {
        "schema": "radio_gs.registered_evidence_external_execution_authority.v2",
        "schema_version": 1,
        "status": "authorized_after_clean_v2_promotion",
        "scene_id": "scene",
        "protocol_hash": raw_manifest["protocol_hash"],
        "prompt_mode": "full_mask",
        "source_frame_id": "source",
        "target_frame_ids": ["target0", "target1"],
        "records": {
            "manifest": manifest_record,
            "source_responsibility_cache": upstream["source_w"],
            "source_responsibility_report": upstream["source_report"],
            "source_positive_mask": upstream["positive"],
            "capability_bank": upstream["capability"],
            "factorized_primitive_state": upstream["state"],
            "v1_checkpoint": upstream["v1"],
            "v2_checkpoint": upstream["v2"],
            "carrier_config": upstream["config"],
            "carrier_checkpoint": upstream["carrier"],
            "camera_map": upstream["camera_map"],
        },
        "capability_source": "canonical_radio_field_official_frozen_capability_views",
        "promotion": {
            "v1_result": upstream["v1_result"],
            "v2_result": upstream["v2_result"],
            "cross_scene_confirmation_receipt": upstream[
                "cross_scene_confirmation_receipt"
            ],
            "cross_scene_confirmation_result": upstream[
                "cross_scene_confirmation_result"
            ],
            "v2_clean_gate_passed": True,
            "v2_cross_scene_gate_passed": True,
            "benchmark_execution_authorized": True,
        },
        "source_access": {
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
            "target_distribution_fit": False,
            "per_scene_parameter_fit": False,
        },
    }
    authority_path = tmp_path / "authority.json"
    write_frozen_json(authority_path, authority)

    score_values = {
        "target0": np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
        "target1": np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
    }
    frame_records = {}
    for frame_id, calibrated in score_values.items():
        frame_root = tmp_path / "scores" / frame_id
        raw = np.clip(calibrated * 0.9 + 0.05, 0, 1).astype(np.float32)
        support = np.ones(calibrated.shape, dtype=np.uint8)
        raw_path = frame_root / "raw.npy"
        calibrated_path = frame_root / "calibrated.npy"
        support_path = frame_root / "support.npy"
        _save_npy(raw_path, raw)
        _save_npy(calibrated_path, calibrated)
        _save_npy(support_path, support)
        frame_records[frame_id] = {
            "raw_v1_probability": file_record(raw_path),
            "supported": file_record(support_path),
            "calibrated_v2_probability": file_record(calibrated_path),
            "shape": list(calibrated.shape),
            "supported_pixels": int(support.sum()),
            "strict_domain_audit": {"count": int(support.sum())},
        }
    receipt = {
        "schema": "radio_gs.registered_evidence_external_prediction.v1",
        "schema_version": 1,
        "artifact_type": "registered_evidence_v2_pre_metric_prediction_receipt",
        "status": "sealed_before_target_ground_truth_open",
        "scene_id": "scene",
        "protocol_hash": raw_manifest["protocol_hash"],
        "execution_authority": file_record(authority_path),
        "primitive_unary": upstream["primitive"],
        "frames": frame_records,
        "frame_count": 2,
        "method_contract": {
            "primitive": "RegisteredEvidenceToUnaryV1 graph-off full global rows",
            "renderer": "frozen alpha-normalized scalar Gaussian compositor",
            "calibration": "GlobalPromptLogitCalibratorV2 after render on supported pixels only",
            "unsupported_policy": "exact zero before and after calibration",
            "graph": False,
            "connected_selection": False,
            "per_scene_parameters": False,
            "threshold_scan": False,
        },
        "carrier": {
            "config": upstream["config"],
            "checkpoint": upstream["carrier"],
            "camera_map": upstream["camera_map"],
        },
        "checkpoints": {"v1": upstream["v1"], "v2": upstream["v2"]},
        "implementation": {
            "core": upstream["adapter_core"],
            "materializer": upstream["materializer"],
        },
        "sealed_before_target_ground_truth_open": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    receipt["content_sha256"] = canonical_json_sha256(receipt)
    receipt_path = tmp_path / "receipt.json"
    write_frozen_json(receipt_path, receipt)
    return {
        "receipt": receipt_path,
        "manifest": manifest_path,
        "gt": gt_records,
        "frames": frame_records,
    }


@pytest.mark.parametrize("benchmark", ["NVOS", "SPIn-NeRF"])
def test_synthetic_hash_bound_receipt_scores_isomorphic_manifests(
    tmp_path: Path, benchmark: str
) -> None:
    bundle = _synthetic_bundle(tmp_path, benchmark=benchmark)
    verified = verify_prediction_barrier(
        prediction_receipt=bundle["receipt"],
        prediction_receipt_sha256=sha256_file(bundle["receipt"]),
        manifest=bundle["manifest"],
        manifest_sha256=sha256_file(bundle["manifest"]),
    )
    result = score_verified_prediction(verified)
    assert result["benchmark"] == benchmark
    assert result["scene_macro"]["frame_count"] == 2
    assert result["evaluation_protocol"] == {
        "input": "calibrated_v2_probability",
        "score_resize": "cv2.INTER_LINEAR",
        "threshold": 0.5,
        "comparison": "greater_or_equal",
        "empty_union_value": 1.0,
        "aggregation": "per_frame_then_scene_macro",
    }
    assert 0 <= result["scene_macro"]["foreground_iou"] <= 1
    assert 0 <= result["scene_macro"]["pixel_accuracy"] <= 1


def test_prediction_tamper_fails_before_missing_gt_is_opened(tmp_path: Path) -> None:
    bundle = _synthetic_bundle(tmp_path)
    Path(bundle["gt"]["target0"]["path"]).unlink()
    support_path = Path(bundle["frames"]["target0"]["supported"]["path"])
    _save_npy(support_path, np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="target support target0 SHA-256 differs"):
        verify_prediction_barrier(
            prediction_receipt=bundle["receipt"],
            prediction_receipt_sha256=sha256_file(bundle["receipt"]),
            manifest=bundle["manifest"],
            manifest_sha256=sha256_file(bundle["manifest"]),
        )


def test_receipt_target_frame_set_must_equal_manifest_before_gt(tmp_path: Path) -> None:
    bundle = _synthetic_bundle(tmp_path)
    receipt = json.loads(bundle["receipt"].read_text(encoding="utf-8"))
    receipt["frames"].pop("target1")
    receipt["frame_count"] = 1
    receipt.pop("content_sha256")
    receipt["content_sha256"] = canonical_json_sha256(receipt)
    replacement = tmp_path / "receipt_missing_frame.json"
    write_frozen_json(replacement, receipt)
    with pytest.raises(ValueError, match="target frame sets differ"):
        verify_prediction_barrier(
            prediction_receipt=replacement,
            prediction_receipt_sha256=sha256_file(replacement),
            manifest=bundle["manifest"],
            manifest_sha256=sha256_file(bundle["manifest"]),
        )


def test_new_frame_adapter_is_numerically_identical_to_frozen_score_frame() -> None:
    probability = np.asarray([[0.0, 0.5, 1.0], [0.49, 0.51, 0.2]], dtype=np.float32)
    ground_truth = np.asarray(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=bool,
    )
    assert _score_external_frame(probability, ground_truth) == _score_frame(
        probability, ground_truth, 0.5
    )
