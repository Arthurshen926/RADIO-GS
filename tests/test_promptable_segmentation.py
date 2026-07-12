import copy
import json

import numpy as np
import pytest

from radio_gs.evaluation.promptable_segmentation import (
    AGGREGATION,
    MissingPredictionError,
    ProtocolError,
    ProtocolHashMismatchError,
    compute_binary_metrics,
    compute_protocol_hash,
    evaluate_binary_scores,
    evaluate_manifest,
    resize_mask_nearest,
    validate_manifest,
)


def _protocol(**overrides):
    protocol = {
        "benchmark": "toy-nvos",
        "dataset_version": "v1",
        "task": "promptable_nvs_binary_segmentation",
        "prompt_type": "reference_mask",
        "metrics": ["foreground_iou", "pixel_accuracy"],
        "aggregation": AGGREGATION,
        "resize": "nearest",
        "prediction_representation": "continuous_margin",
        "threshold_comparison": "greater_or_equal",
        "empty_union_value": 1.0,
        "allow_reference_scoring": False,
        "threshold": {"mode": "fixed", "value": 0.5},
    }
    protocol.update(overrides)
    return protocol


def _save_mask(path, values):
    np.save(path, np.asarray(values))
    return str(path)


def _frame(frame_id, gt_path, prediction_path=None):
    value = {"frame_id": frame_id, "ground_truth": str(gt_path)}
    if prediction_path is not None:
        value["prediction"] = str(prediction_path)
    return value


def test_frame_scene_dataset_scene_macro_and_reference_exclusion(tmp_path):
    one = np.ones((2, 2), dtype=np.uint8)
    zero = np.zeros((2, 2), dtype=np.uint8)

    manifest = {
        "schema_version": 1,
        "protocol": _protocol(),
        "scenes": [
            {
                "scene_id": "scene_a",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "frames": [
                    # No reference prediction: its exclusion is observable.
                    _frame("ref", _save_mask(tmp_path / "a_ref.npy", one)),
                    _frame(
                        "target",
                        _save_mask(tmp_path / "a_gt.npy", one),
                        _save_mask(tmp_path / "a_pred.npy", one),
                    ),
                ],
            },
            {
                "scene_id": "scene_b",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target_1", "target_2"],
                "frames": [
                    _frame("ref", _save_mask(tmp_path / "b_ref.npy", one)),
                    _frame(
                        "target_1",
                        _save_mask(tmp_path / "b_gt1.npy", one),
                        _save_mask(tmp_path / "b_pred1.npy", zero),
                    ),
                    _frame(
                        "target_2",
                        _save_mask(tmp_path / "b_gt2.npy", one),
                        _save_mask(tmp_path / "b_pred2.npy", zero),
                    ),
                ],
            },
        ],
    }

    result = evaluate_manifest(manifest)

    assert result["dataset"]["num_scenes"] == 2
    assert result["dataset"]["num_frames"] == 3
    assert result["scenes"][0]["foreground_iou"] == pytest.approx(1.0)
    assert result["scenes"][1]["foreground_iou"] == pytest.approx(0.0)
    # Scene macro is (1 + 0) / 2, not flat-frame macro (1 + 0 + 0) / 3.
    assert result["dataset"]["foreground_iou"] == pytest.approx(0.5)
    assert result["dataset"]["pixel_accuracy"] == pytest.approx(0.5)


def test_resize_mask_nearest_replicates_pixels():
    source = np.array([[0, 1], [2, 3]], dtype=np.int16)
    resized = resize_mask_nearest(source, (4, 4))

    assert resized.dtype == source.dtype
    assert resized.tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]


def test_target_calibration_is_rejected_fail_closed(tmp_path):
    manifest = {
        "schema_version": 1,
        "protocol": _protocol(
            threshold={
                "mode": "calibrated",
                "source": "target",
                "scope": "dataset",
                "candidates": [0.25, 0.5],
            }
        ),
        "scenes": [],
    }

    with pytest.raises(ProtocolError, match="Target/test-set threshold calibration"):
        validate_manifest(manifest)


def test_missing_evaluation_prediction_is_fatal(tmp_path):
    gt = _save_mask(tmp_path / "gt.npy", np.ones((2, 2), dtype=np.uint8))
    manifest = {
        "schema_version": 1,
        "protocol": _protocol(),
        "scenes": [
            {
                "scene_id": "scene",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "frames": [_frame("ref", gt), _frame("target", gt)],
            }
        ],
    }

    with pytest.raises(MissingPredictionError, match="scene scene frame target"):
        evaluate_manifest(manifest)


def test_protocol_hash_is_stable_path_independent_and_verified(tmp_path):
    gt = _save_mask(tmp_path / "gt.npy", np.ones((2, 2), dtype=np.uint8))
    pred = _save_mask(tmp_path / "pred.npy", np.ones((2, 2), dtype=np.uint8))
    manifest = {
        "schema_version": 1,
        "protocol": _protocol(),
        "scenes": [
            {
                "scene_id": "scene",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "frames": [_frame("ref", gt), _frame("target", gt, pred)],
            }
        ],
    }

    original_hash = compute_protocol_hash(manifest)
    other_prediction = copy.deepcopy(manifest)
    other_prediction["scenes"][0]["frames"][1]["prediction"] = "another_method.npy"
    assert compute_protocol_hash(other_prediction) == original_hash

    changed_protocol = copy.deepcopy(manifest)
    changed_protocol["protocol"]["empty_union_value"] = 0.0
    assert compute_protocol_hash(changed_protocol) != original_hash

    stale_hash = copy.deepcopy(manifest)
    stale_hash["protocol_hash"] = "0" * 64
    with pytest.raises(ProtocolHashMismatchError, match="does not match"):
        validate_manifest(stale_hash)

    verified = copy.deepcopy(manifest)
    verified["protocol_hash"] = original_hash
    assert validate_manifest(verified)["protocol_hash"] == original_hash


def test_empty_empty_iou_defaults_to_one_and_is_configurable():
    empty = np.zeros((2, 3), dtype=bool)

    assert compute_binary_metrics(empty, empty)["foreground_iou"] == 1.0
    assert (
        compute_binary_metrics(empty, empty, empty_union_value=0.0)["foreground_iou"]
        == 0.0
    )


def test_prompt_evaluation_overlap_requires_explicit_protocol_change(tmp_path):
    gt = _save_mask(tmp_path / "gt.npy", np.ones((2, 2), dtype=np.uint8))
    pred = _save_mask(tmp_path / "pred.npy", np.ones((2, 2), dtype=np.uint8))
    manifest = {
        "schema_version": 1,
        "protocol": _protocol(),
        "scenes": [
            {
                "scene_id": "scene",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["ref"],
                "frames": [_frame("ref", gt, pred)],
            }
        ],
    }

    with pytest.raises(ProtocolError, match="prompt/evaluation frames overlap"):
        validate_manifest(manifest)


def test_json_manifest_calibrates_only_on_reference_frames(tmp_path):
    _save_mask(tmp_path / "ref_gt.npy", np.ones((1, 2), dtype=np.uint8))
    _save_mask(tmp_path / "ref_pred.npy", np.full((1, 2), 0.4, dtype=np.float32))
    _save_mask(tmp_path / "target_gt.npy", np.ones((1, 2), dtype=np.uint8))
    _save_mask(tmp_path / "target_pred.npy", np.full((1, 2), 0.4, dtype=np.float32))
    manifest = {
        "schema_version": 1,
        "protocol": _protocol(
            threshold={
                "mode": "calibrated",
                "source": "reference",
                "scope": "scene",
                "candidates": [0.3, 0.5],
            }
        ),
        "scenes": [
            {
                "scene_id": "scene",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "frames": [
                    _frame("ref", "ref_gt.npy", "ref_pred.npy"),
                    _frame("target", "target_gt.npy", "target_pred.npy"),
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(manifest_path)

    assert result["thresholds"]["by_scene"] == {"scene": 0.3}
    assert result["dataset"]["foreground_iou"] == 1.0
    assert result["thresholds"]["calibration"]["scene"]["source_frames"][0][
        "frame_id"
    ] == "ref"


def test_unscored_nvos_scribble_reference_does_not_require_full_gt(tmp_path):
    target_gt = _save_mask(tmp_path / "target_gt.npy", np.ones((2, 2), dtype=np.uint8))
    target_pred = _save_mask(tmp_path / "target_pred.npy", np.ones((2, 2), dtype=np.uint8))
    manifest = {
        "schema_version": 1,
        "protocol": _protocol(prompt_type="positive_negative_scribbles"),
        "scenes": [
            {
                "scene_id": "fern",
                "prompt_frame_ids": ["reference"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "frames": [
                    {"frame_id": "reference", "ground_truth": None},
                    _frame("target", target_gt, target_pred),
                ],
            }
        ],
    }

    validated = validate_manifest(manifest)
    assert validated["scenes"][0]["frames"]["reference"]["ground_truth"] is None
    assert evaluate_manifest(manifest)["dataset"]["foreground_iou"] == 1.0


def test_binary_mask_representation_scores_zero_as_background():
    prediction = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    target = prediction.copy()

    metrics = evaluate_binary_scores(
        prediction,
        target,
        threshold=0.5,
        prediction_representation="binary_mask",
        threshold_comparison="greater_or_equal",
    )

    assert metrics["foreground_iou"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0


def test_binary_mask_protocol_rejects_margin_threshold_metadata():
    manifest = {
        "schema_version": 1,
        "protocol": _protocol(
            prediction_representation="binary_mask",
            threshold={"mode": "fixed", "value": 0.0},
        ),
        "scenes": [],
    }
    with pytest.raises(ProtocolError, match="binary_mask predictions require"):
        validate_manifest(manifest)
