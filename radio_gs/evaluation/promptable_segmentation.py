"""Protocol-safe evaluation for promptable novel-view binary segmentation.

The evaluator in this module is deliberately independent from a model.  It
consumes precomputed score/mask files and a JSON-serializable protocol
manifest.  This makes the metric implementation reusable for NVOS,
SPIn-NeRF, and methods whose predictions were produced outside RADIO-GS.

Manifest schema (version 1)::

    {
      "schema_version": 1,
      "protocol": {
        "benchmark": "NVOS",
        "dataset_version": "official-2022",
        "task": "promptable_nvs_binary_segmentation",
        "prompt_type": "foreground_background_scribbles",
        "metrics": ["foreground_iou", "pixel_accuracy"],
        "aggregation": "per_frame_then_per_scene_then_dataset_scene_macro",
        "resize": "nearest",
        "prediction_representation": "continuous_margin",
        "threshold_comparison": "greater_or_equal",
        "empty_union_value": 1.0,
        "allow_reference_scoring": false,
        "threshold": {"mode": "fixed", "value": 0.5}
      },
      "scenes": [
        {
          "scene_id": "fern",
          "prompt_frame_ids": ["000"],
          "calibration_frame_ids": [],
          "evaluation_frame_ids": ["001"],
          "frames": [
            {"frame_id": "000", "ground_truth": "gt/000.png"},
            {
              "frame_id": "001",
              "ground_truth": "gt/001.png",
              "prediction": "pred/001.npy"
            }
          ]
        }
      ]
    }

``threshold.mode`` is either ``fixed`` or ``calibrated``.  A calibrated
threshold must declare ``source`` as ``reference`` or ``calibration`` and is
selected only from the corresponding non-evaluation frames.  Target/test
calibration and implicit fallback to target frames are rejected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
TASK_NAME = "promptable_nvs_binary_segmentation"
METRIC_NAMES = ("foreground_iou", "pixel_accuracy")
AGGREGATION = "per_frame_then_per_scene_then_dataset_scene_macro"
RESIZE_POLICY = "nearest"
PREDICTION_REPRESENTATIONS = (
    "continuous_margin",
    "probability",
    "binary_mask",
)
THRESHOLD_COMPARISON = "greater_or_equal"


class ProtocolError(ValueError):
    """Raised when a manifest could silently change the evaluation protocol."""


class ProtocolHashMismatchError(ProtocolError):
    """Raised when a declared protocol hash does not match the manifest."""


class MissingPredictionError(FileNotFoundError):
    """Raised when an evaluation/calibration prediction is not available."""


ManifestLike = Union[str, Path, Mapping[str, Any]]


def _ensure_2d(array: Any, *, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0]
    if values.ndim == 3 and values.shape[0] == 1:
        return values[0]
    raise ValueError(
        "%s must have shape [H,W], [H,W,1], or [1,H,W], got %s"
        % (name, tuple(values.shape))
    )


def resize_mask_nearest(mask: Any, target_shape: Sequence[int]) -> np.ndarray:
    """Resize a 2D mask/score map with nearest-neighbor interpolation only."""

    values = _ensure_2d(mask, name="mask")
    if len(target_shape) != 2:
        raise ValueError("target_shape must be (height, width)")
    height, width = int(target_shape[0]), int(target_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("target_shape dimensions must be positive")
    if values.shape == (height, width):
        return values.copy()

    original_dtype = values.dtype
    is_bool = np.issubdtype(original_dtype, np.bool_)
    if is_bool:
        image_values = values.astype(np.uint8)
    elif np.issubdtype(original_dtype, np.floating):
        # Pillow's F mode supports nearest-neighbor resizing and preserves
        # arbitrary score ranges (including logits).
        image_values = values.astype(np.float32)
    elif original_dtype.itemsize > 4:
        image_values = values.astype(np.int32)
    else:
        image_values = values

    resampling = getattr(Image, "Resampling", Image).NEAREST
    resized = np.asarray(
        Image.fromarray(image_values).resize((width, height), resample=resampling)
    )
    if is_bool:
        return resized.astype(bool)
    return resized.astype(original_dtype, copy=False)


def compute_binary_metrics(
    prediction: Any,
    ground_truth: Any,
    *,
    empty_union_value: float = 1.0,
) -> Dict[str, float]:
    """Compute foreground IoU and all-pixel accuracy for two binary masks."""

    pred = _ensure_2d(prediction, name="prediction").astype(bool, copy=False)
    target = _ensure_2d(ground_truth, name="ground_truth").astype(bool, copy=False)
    if pred.shape != target.shape:
        raise ValueError(
            "prediction/ground_truth shapes differ: %s versus %s"
            % (tuple(pred.shape), tuple(target.shape))
        )
    if pred.size == 0:
        raise ValueError("binary masks must contain at least one pixel")
    if not math.isfinite(float(empty_union_value)) or not 0.0 <= float(
        empty_union_value
    ) <= 1.0:
        raise ValueError("empty_union_value must be finite and within [0,1]")

    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    foreground_iou = (
        float(empty_union_value) if union == 0 else float(intersection) / float(union)
    )
    pixel_accuracy = float(np.equal(pred, target).mean())
    return {
        "foreground_iou": foreground_iou,
        "pixel_accuracy": pixel_accuracy,
    }


def evaluate_binary_scores(
    scores: Any,
    ground_truth: Any,
    *,
    threshold: float,
    empty_union_value: float = 1.0,
    resize: str = RESIZE_POLICY,
    prediction_representation: str = "continuous_margin",
    threshold_comparison: str = THRESHOLD_COMPARISON,
) -> Dict[str, float]:
    """Threshold a score map and evaluate it against a binary ground truth."""

    score_values = _ensure_2d(scores, name="scores")
    target = _ensure_2d(ground_truth, name="ground_truth").astype(bool, copy=False)
    if not bool(np.isfinite(score_values).all()):
        raise ValueError("prediction scores contain NaN or infinity")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if prediction_representation not in PREDICTION_REPRESENTATIONS:
        raise ProtocolError(
            f"prediction_representation must be one of {PREDICTION_REPRESENTATIONS}"
        )
    if threshold_comparison != THRESHOLD_COMPARISON:
        raise ProtocolError(
            f"threshold_comparison must be {THRESHOLD_COMPARISON!r}"
        )
    if score_values.shape != target.shape:
        if resize != RESIZE_POLICY:
            raise ProtocolError(
                "Only nearest-neighbor prediction-to-ground-truth resizing is allowed"
            )
        score_values = resize_mask_nearest(score_values, target.shape)
    if prediction_representation == "probability":
        if bool(((score_values < 0.0) | (score_values > 1.0)).any()):
            raise ValueError("Probability predictions must lie within [0,1]")
    if prediction_representation == "binary_mask":
        if not bool(np.logical_or(score_values == 0.0, score_values == 1.0).all()):
            raise ValueError("binary_mask predictions must contain only 0 and 1")
        prediction = score_values.astype(bool, copy=False)
    else:
        prediction = score_values >= float(threshold)
    return compute_binary_metrics(
        prediction,
        target,
        empty_union_value=empty_union_value,
    )


def load_json_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON object without mutating it during later validation."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProtocolError("manifest JSON must contain an object at the top level")
    return value


def _require_nonempty_string(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError("%s.%s must be a non-empty string" % (where, key))
    return value.strip()


def _normalize_frame_ids(value: Any, *, where: str, required: bool) -> List[str]:
    if value is None:
        if required:
            raise ProtocolError("%s must be explicitly declared" % where)
        return []
    if not isinstance(value, list):
        raise ProtocolError("%s must be a JSON list" % where)
    frame_ids: List[str] = []
    for raw in value:
        frame_id = str(raw)
        if not frame_id:
            raise ProtocolError("%s contains an empty frame id" % where)
        if frame_id in frame_ids:
            raise ProtocolError("%s contains duplicate frame id %r" % (where, frame_id))
        frame_ids.append(frame_id)
    if required and not frame_ids:
        raise ProtocolError("%s must contain at least one frame" % where)
    return frame_ids


def _normalize_threshold_policy(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProtocolError("protocol.threshold must be an object")
    mode = str(raw.get("mode", "")).lower()
    if mode == "fixed":
        if "value" not in raw:
            raise ProtocolError("A fixed threshold requires protocol.threshold.value")
        value = float(raw["value"])
        if not math.isfinite(value):
            raise ProtocolError("Fixed threshold value must be finite")
        forbidden = {"source", "source_frame_ids", "candidates", "scope"}.intersection(raw)
        if forbidden:
            raise ProtocolError(
                "Fixed threshold must not declare calibration fields: %s"
                % sorted(forbidden)
            )
        return {"mode": "fixed", "value": value}

    if mode != "calibrated":
        if mode in {"target", "test", "target_calibrated", "test_calibrated"}:
            raise ProtocolError("Target/test-set threshold calibration is forbidden")
        raise ProtocolError("threshold.mode must be 'fixed' or 'calibrated'")

    source = str(raw.get("source", "")).lower()
    if source in {"target", "test", "evaluation", "eval"}:
        raise ProtocolError("Target/test-set threshold calibration is forbidden")
    if source not in {"reference", "calibration"}:
        raise ProtocolError(
            "Calibrated threshold source must be 'reference' or 'calibration'"
        )
    scope = str(raw.get("scope", "dataset")).lower()
    if scope not in {"dataset", "scene"}:
        raise ProtocolError("Calibrated threshold scope must be 'dataset' or 'scene'")
    objective = str(raw.get("objective", "foreground_iou"))
    if objective != "foreground_iou":
        raise ProtocolError("Only foreground_iou threshold calibration is supported")
    tie_break = str(raw.get("tie_break", "lowest"))
    if tie_break != "lowest":
        raise ProtocolError("Only deterministic tie_break='lowest' is supported")
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ProtocolError("Calibrated threshold requires a non-empty candidates list")
    candidates = sorted({float(value) for value in candidates_raw})
    if not all(math.isfinite(value) for value in candidates):
        raise ProtocolError("Threshold candidates must all be finite")

    normalized: Dict[str, Any] = {
        "mode": "calibrated",
        "source": source,
        "scope": scope,
        "objective": objective,
        "tie_break": tie_break,
        "candidates": candidates,
    }
    if "source_frame_ids" in raw:
        source_ids = raw["source_frame_ids"]
        if not isinstance(source_ids, Mapping):
            raise ProtocolError(
                "threshold.source_frame_ids must map scene ids to frame-id lists"
            )
        normalized["source_frame_ids"] = {
            str(scene_id): _normalize_frame_ids(
                ids,
                where="threshold.source_frame_ids[%s]" % scene_id,
                required=True,
            )
            for scene_id, ids in source_ids.items()
        }
    return normalized


def _normalize_protocol(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProtocolError("manifest.protocol must be an object")
    protocol = copy.deepcopy(dict(raw))
    _require_nonempty_string(protocol, "benchmark", "protocol")
    _require_nonempty_string(protocol, "dataset_version", "protocol")
    _require_nonempty_string(protocol, "prompt_type", "protocol")

    if protocol.get("task") != TASK_NAME:
        raise ProtocolError("protocol.task must be %r" % TASK_NAME)
    if protocol.get("metrics") != list(METRIC_NAMES):
        raise ProtocolError(
            "protocol.metrics must be exactly %r" % list(METRIC_NAMES)
        )
    if protocol.get("aggregation") != AGGREGATION:
        raise ProtocolError("protocol.aggregation must be %r" % AGGREGATION)
    if protocol.get("resize") != RESIZE_POLICY:
        raise ProtocolError("protocol.resize must be 'nearest'")

    representation = str(protocol.get("prediction_representation", ""))
    if representation not in PREDICTION_REPRESENTATIONS:
        raise ProtocolError(
            "protocol.prediction_representation must be one of %r"
            % (PREDICTION_REPRESENTATIONS,)
        )
    protocol["prediction_representation"] = representation
    comparison = str(protocol.get("threshold_comparison", ""))
    if comparison != THRESHOLD_COMPARISON:
        raise ProtocolError(
            "protocol.threshold_comparison must be %r" % THRESHOLD_COMPARISON
        )
    protocol["threshold_comparison"] = comparison

    if "empty_union_value" not in protocol:
        raise ProtocolError("protocol.empty_union_value must be explicit")
    empty_union_value = float(protocol["empty_union_value"])
    if not math.isfinite(empty_union_value) or not 0.0 <= empty_union_value <= 1.0:
        raise ProtocolError("protocol.empty_union_value must be within [0,1]")
    protocol["empty_union_value"] = empty_union_value

    allow_reference_scoring = protocol.get("allow_reference_scoring", False)
    if not isinstance(allow_reference_scoring, bool):
        raise ProtocolError("protocol.allow_reference_scoring must be boolean")
    protocol["allow_reference_scoring"] = allow_reference_scoring
    if "exclude_reference_frames" in protocol:
        exclude = protocol["exclude_reference_frames"]
        if not isinstance(exclude, bool):
            raise ProtocolError("protocol.exclude_reference_frames must be boolean")
        if exclude == allow_reference_scoring:
            raise ProtocolError(
                "exclude_reference_frames must be the inverse of "
                "allow_reference_scoring"
            )
    protocol["exclude_reference_frames"] = not allow_reference_scoring
    protocol["threshold"] = _normalize_threshold_policy(protocol.get("threshold"))
    if representation == "binary_mask":
        if protocol["threshold"] != {"mode": "fixed", "value": 0.5}:
            raise ProtocolError(
                "binary_mask predictions require fixed threshold metadata 0.5; "
                "the binary values themselves are scored directly"
            )
    return protocol


def _normalize_frames(raw: Any, *, scene_id: str) -> Dict[str, Dict[str, Any]]:
    if isinstance(raw, Mapping):
        entries = []
        for frame_id, frame_value in raw.items():
            if not isinstance(frame_value, Mapping):
                raise ProtocolError(
                    "scene %s frame %s must be an object" % (scene_id, frame_id)
                )
            entry = copy.deepcopy(dict(frame_value))
            declared_id = entry.get("frame_id")
            if declared_id is not None and str(declared_id) != str(frame_id):
                raise ProtocolError(
                    "scene %s frame key/id mismatch for %s" % (scene_id, frame_id)
                )
            entry["frame_id"] = str(frame_id)
            entries.append(entry)
    elif isinstance(raw, list):
        entries = copy.deepcopy(raw)
    else:
        raise ProtocolError("scene %s frames must be a list or object" % scene_id)

    frames: Dict[str, Dict[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ProtocolError(
                "scene %s frames[%d] must be an object" % (scene_id, index)
            )
        entry = copy.deepcopy(dict(raw_entry))
        raw_frame_id = entry.get("frame_id")
        if raw_frame_id is None:
            raise ProtocolError(
                "scene[%s].frames[%d].frame_id must be explicitly declared"
                % (scene_id, index)
            )
        frame_id = _require_nonempty_string(
            {"frame_id": str(raw_frame_id)},
            "frame_id",
            "scene[%s].frames[%d]" % (scene_id, index),
        )
        if frame_id in frames:
            raise ProtocolError("scene %s has duplicate frame %s" % (scene_id, frame_id))
        entry["frame_id"] = frame_id
        ground_truth = entry.get("ground_truth")
        if ground_truth is not None:
            if not isinstance(ground_truth, (str, Path)):
                raise ProtocolError(
                    "scene %s frame %s ground_truth must be a path or null"
                    % (scene_id, frame_id)
                )
            entry["ground_truth"] = str(ground_truth)
        if "prediction" in entry:
            if not isinstance(entry["prediction"], (str, Path)):
                raise ProtocolError(
                    "scene %s frame %s prediction must be a path"
                    % (scene_id, frame_id)
                )
            entry["prediction"] = str(entry["prediction"])
        frames[frame_id] = entry
    if not frames:
        raise ProtocolError("scene %s must contain at least one frame" % scene_id)
    return frames


def _normalize_scenes(raw: Any, *, protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ProtocolError("manifest.scenes must be a non-empty list")
    scenes: List[Dict[str, Any]] = []
    seen_scene_ids = set()
    threshold = protocol["threshold"]
    for index, raw_scene in enumerate(raw):
        if not isinstance(raw_scene, Mapping):
            raise ProtocolError("scenes[%d] must be an object" % index)
        scene = copy.deepcopy(dict(raw_scene))
        scene_id = _require_nonempty_string(scene, "scene_id", "scenes[%d]" % index)
        if scene_id in seen_scene_ids:
            raise ProtocolError("duplicate scene_id %r" % scene_id)
        seen_scene_ids.add(scene_id)
        frames = _normalize_frames(scene.get("frames"), scene_id=scene_id)

        prompt_ids = _normalize_frame_ids(
            scene.get("prompt_frame_ids"),
            where="scene[%s].prompt_frame_ids" % scene_id,
            required=True,
        )
        calibration_ids = _normalize_frame_ids(
            scene.get("calibration_frame_ids", []),
            where="scene[%s].calibration_frame_ids" % scene_id,
            required=False,
        )
        evaluation_ids = _normalize_frame_ids(
            scene.get("evaluation_frame_ids"),
            where="scene[%s].evaluation_frame_ids" % scene_id,
            required=True,
        )
        referenced = set(prompt_ids) | set(calibration_ids) | set(evaluation_ids)
        unknown = referenced.difference(frames)
        if unknown:
            raise ProtocolError(
                "scene %s split lists reference unknown frames: %s"
                % (scene_id, sorted(unknown))
            )
        prompt_eval_overlap = set(prompt_ids).intersection(evaluation_ids)
        if prompt_eval_overlap and not protocol["allow_reference_scoring"]:
            raise ProtocolError(
                "scene %s prompt/evaluation frames overlap (%s); set "
                "allow_reference_scoring=true only for an intentionally different "
                "protocol"
                % (scene_id, sorted(prompt_eval_overlap))
            )
        calibration_eval_overlap = set(calibration_ids).intersection(evaluation_ids)
        if calibration_eval_overlap:
            raise ProtocolError(
                "scene %s calibration/evaluation frames overlap: %s"
                % (scene_id, sorted(calibration_eval_overlap))
            )

        if threshold["mode"] == "calibrated":
            allowed_ids = prompt_ids if threshold["source"] == "reference" else calibration_ids
            selected_ids = allowed_ids
            source_map = threshold.get("source_frame_ids")
            if source_map is not None:
                if scene_id not in source_map:
                    raise ProtocolError(
                        "threshold.source_frame_ids is missing scene %s" % scene_id
                    )
                selected_ids = source_map[scene_id]
                invalid = set(selected_ids).difference(allowed_ids)
                if invalid:
                    raise ProtocolError(
                        "scene %s threshold source frames are not declared %s frames: %s"
                        % (scene_id, threshold["source"], sorted(invalid))
                    )
            if not selected_ids:
                raise ProtocolError(
                    "scene %s has no non-evaluation %s frames for calibration"
                    % (scene_id, threshold["source"])
                )
            if set(selected_ids).intersection(evaluation_ids):
                # This remains forbidden even when reference scoring was
                # explicitly allowed: calibration and scoring data must not be
                # the same observations.
                raise ProtocolError("Target/test-set threshold calibration is forbidden")
            scene["threshold_source_frame_ids"] = list(selected_ids)

        required_ground_truth_ids = set(evaluation_ids)
        required_ground_truth_ids.update(scene.get("threshold_source_frame_ids", []))
        missing_ground_truth = sorted(
            frame_id
            for frame_id in required_ground_truth_ids
            if not frames[frame_id].get("ground_truth")
        )
        if missing_ground_truth:
            raise ProtocolError(
                "scene %s scoring/calibration frames require ground_truth paths: %s"
                % (scene_id, missing_ground_truth)
            )

        scene["scene_id"] = scene_id
        scene["prompt_frame_ids"] = prompt_ids
        scene["calibration_frame_ids"] = calibration_ids
        scene["evaluation_frame_ids"] = evaluation_ids
        scene["frames"] = frames
        scenes.append(scene)
    return scenes


def _protocol_hash_payload(normalized: Mapping[str, Any]) -> Dict[str, Any]:
    # All protocol-level fields are considered critical except prose and a
    # previously declared hash.  Method/prediction metadata belongs outside
    # `protocol` and therefore cannot accidentally change comparability.
    ignored_protocol_fields = {"description", "notes", "protocol_hash", "expected_hash"}
    protocol = {
        key: copy.deepcopy(value)
        for key, value in normalized["protocol"].items()
        if key not in ignored_protocol_fields
    }
    scene_payloads = []
    for scene in normalized["scenes"]:
        frame_payloads = []
        for frame_id in sorted(scene["frames"]):
            frame = scene["frames"][frame_id]
            frame_payload: Dict[str, Any] = {"frame_id": frame_id}
            # Paths vary across machines.  Stable content identifiers, when
            # supplied by a dataset adapter, are intentionally hashed.
            if "ground_truth_sha256" in frame:
                frame_payload["ground_truth_sha256"] = str(frame["ground_truth_sha256"])
            frame_payloads.append(frame_payload)
        scene_payload: Dict[str, Any] = {
            "scene_id": scene["scene_id"],
            "prompt_frame_ids": sorted(scene["prompt_frame_ids"]),
            "calibration_frame_ids": sorted(scene["calibration_frame_ids"]),
            "evaluation_frame_ids": sorted(scene["evaluation_frame_ids"]),
            "frames": frame_payloads,
        }
        if "threshold_source_frame_ids" in scene:
            scene_payload["threshold_source_frame_ids"] = sorted(
                scene["threshold_source_frame_ids"]
            )
        scene_payloads.append(scene_payload)
    scene_payloads.sort(key=lambda item: item["scene_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "scenes": scene_payloads,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ProtocolError("manifest must be a mapping")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("manifest.schema_version must equal %d" % SCHEMA_VERSION)
    normalized = copy.deepcopy(dict(manifest))
    normalized["protocol"] = _normalize_protocol(manifest.get("protocol"))
    normalized["scenes"] = _normalize_scenes(
        manifest.get("scenes"), protocol=normalized["protocol"]
    )
    return normalized


def compute_protocol_hash(manifest: Mapping[str, Any]) -> str:
    """Return the SHA-256 hash of protocol-critical, path-independent fields."""

    normalized = _normalize_manifest(manifest)
    payload = _protocol_hash_payload(normalized)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a manifest, including any declared hash."""

    normalized = _normalize_manifest(manifest)
    actual_hash = hashlib.sha256(
        _canonical_json(_protocol_hash_payload(normalized)).encode("utf-8")
    ).hexdigest()
    declared_hash = manifest.get("protocol_hash")
    if declared_hash is None and isinstance(manifest.get("protocol"), Mapping):
        declared_hash = manifest["protocol"].get("expected_hash")
    if declared_hash is not None and str(declared_hash) != actual_hash:
        raise ProtocolHashMismatchError(
            "Declared protocol hash %s does not match computed hash %s"
            % (declared_hash, actual_hash)
        )
    normalized["protocol_hash"] = actual_hash
    return normalized


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(str(path), allow_pickle=False))
    if suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as archive:
            names = list(archive.files)
            if "mask" in names:
                return np.asarray(archive["mask"])
            if len(names) != 1:
                raise ValueError(
                    "%s must contain exactly one array or an array named 'mask'" % path
                )
            return np.asarray(archive[names[0]])
    with Image.open(str(path)) as image:
        # Binary benchmark masks are scalar.  Converting palette/RGB images to
        # luminance prevents accidental channel broadcasting.
        return np.asarray(image.convert("L"))


def load_ground_truth_mask(path: Union[str, Path]) -> np.ndarray:
    """Load a binary ground-truth mask; every nonzero value is foreground."""

    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError("Ground-truth mask not found: %s" % mask_path)
    return _ensure_2d(_load_array(mask_path), name="ground_truth").astype(bool)


def load_prediction_scores(path: Union[str, Path]) -> np.ndarray:
    """Load precomputed scores/masks, normalizing integer image files to [0,1]."""

    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise MissingPredictionError("Prediction mask not found: %s" % prediction_path)
    values = _ensure_2d(_load_array(prediction_path), name="prediction")
    if prediction_path.suffix.lower() not in {".npy", ".npz"} and np.issubdtype(
        values.dtype, np.integer
    ):
        if values.size and int(values.max()) > 1:
            dtype_max = int(np.iinfo(values.dtype).max)
            values = values.astype(np.float32) / float(dtype_max)
    values = values.astype(np.float32, copy=False)
    if not bool(np.isfinite(values).all()):
        raise ValueError("Prediction scores contain NaN or infinity: %s" % prediction_path)
    return values


def _resolve_root(
    explicit: Optional[Union[str, Path]],
    manifest_value: Any,
    *,
    base_dir: Path,
) -> Path:
    raw = explicit if explicit is not None else manifest_value
    if raw is None:
        return base_dir
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


def _resolve_file(path: Union[str, Path], root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_prediction_overrides(
    source: Optional[ManifestLike],
    *,
    expected_protocol_hash: str,
    default_base_dir: Path,
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]], Path]:
    if source is None:
        return None, None, default_base_dir
    if isinstance(source, (str, Path)):
        path = Path(source)
        raw = load_json_manifest(path)
        base_dir = path.parent
        require_hash = True
    elif isinstance(source, Mapping):
        raw = copy.deepcopy(dict(source))
        base_dir = default_base_dir
        require_hash = False
    else:
        raise ProtocolError("prediction_manifest must be a path or mapping")

    declared_hash = raw.get("protocol_hash")
    if require_hash and declared_hash is None:
        raise ProtocolError("A JSON prediction manifest must declare protocol_hash")
    if declared_hash is not None and str(declared_hash) != expected_protocol_hash:
        raise ProtocolHashMismatchError(
            "Prediction protocol hash %s does not match evaluation protocol %s"
            % (declared_hash, expected_protocol_hash)
        )
    predictions = raw.get("predictions", raw)
    if not isinstance(predictions, Mapping):
        raise ProtocolError("prediction manifest 'predictions' must be an object")
    prediction_root = _resolve_root(None, raw.get("prediction_root"), base_dir=base_dir)
    prediction_hashes = raw.get("prediction_sha256")
    if prediction_hashes is not None and not isinstance(prediction_hashes, Mapping):
        raise ProtocolError("prediction_sha256 must be an object when declared")
    if str(raw.get("kind", "")).startswith("promptable_nvs_") and require_hash:
        if prediction_hashes is None:
            raise ProtocolError(
                "Protocol-bound promptable prediction manifests must declare "
                "prediction_sha256"
            )
    return predictions, prediction_hashes, prediction_root


def _prediction_override(
    predictions: Optional[Mapping[str, Any]], scene_id: str, frame_id: str
) -> Optional[str]:
    if predictions is None:
        return None
    nested = predictions.get(scene_id)
    if isinstance(nested, Mapping) and frame_id in nested:
        return str(nested[frame_id])
    slash_key = "%s/%s" % (scene_id, frame_id)
    if slash_key in predictions:
        return str(predictions[slash_key])
    colon_key = "%s:%s" % (scene_id, frame_id)
    if colon_key in predictions:
        return str(predictions[colon_key])
    return None


def _prediction_path(
    frame: Mapping[str, Any],
    *,
    scene_id: str,
    frame_id: str,
    overrides: Optional[Mapping[str, Any]],
    override_hashes: Optional[Mapping[str, Any]],
    inline_root: Path,
    override_root: Path,
) -> Path:
    override = _prediction_override(overrides, scene_id, frame_id)
    if override is not None:
        path = _resolve_file(override, override_root)
    elif "prediction" in frame:
        path = _resolve_file(frame["prediction"], inline_root)
    else:
        raise MissingPredictionError(
            "Missing prediction for scene %s frame %s" % (scene_id, frame_id)
        )
    if not path.is_file():
        raise MissingPredictionError(
            "Prediction mask not found for scene %s frame %s: %s"
            % (scene_id, frame_id, path)
        )
    expected_hash = _prediction_override(override_hashes, scene_id, frame_id)
    if expected_hash is not None:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise ProtocolHashMismatchError(
                "Prediction SHA-256 %s does not match declared %s for %s/%s"
                % (actual_hash, expected_hash, scene_id, frame_id)
            )
    return path


def _frame_inputs(
    scene: Mapping[str, Any],
    frame_id: str,
    *,
    overrides: Optional[Mapping[str, Any]],
    override_hashes: Optional[Mapping[str, Any]],
    ground_truth_root: Path,
    inline_prediction_root: Path,
    override_prediction_root: Path,
) -> Tuple[np.ndarray, np.ndarray, Path, Path]:
    frame = scene["frames"][frame_id]
    gt_path = _resolve_file(frame["ground_truth"], ground_truth_root)
    prediction_path = _prediction_path(
        frame,
        scene_id=scene["scene_id"],
        frame_id=frame_id,
        overrides=overrides,
        override_hashes=override_hashes,
        inline_root=inline_prediction_root,
        override_root=override_prediction_root,
    )
    target = load_ground_truth_mask(gt_path)
    scores = load_prediction_scores(prediction_path)
    if scores.shape != target.shape:
        scores = resize_mask_nearest(scores, target.shape)
    return scores, target, prediction_path, gt_path


def _choose_threshold(
    examples: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    candidates: Sequence[float],
    empty_union_value: float,
    prediction_representation: str,
    threshold_comparison: str,
) -> Tuple[float, List[Dict[str, float]]]:
    if not examples:
        raise ProtocolError("Threshold calibration has no permitted source frames")
    diagnostics = []
    for threshold in candidates:
        frame_ious = [
            evaluate_binary_scores(
                scores,
                target,
                threshold=threshold,
                empty_union_value=empty_union_value,
                prediction_representation=prediction_representation,
                threshold_comparison=threshold_comparison,
            )["foreground_iou"]
            for scores, target in examples
        ]
        diagnostics.append(
            {"threshold": float(threshold), "mean_foreground_iou": float(np.mean(frame_ious))}
        )
    # Candidates were sorted in validation, so max keeps the lowest threshold
    # when objective values tie.
    best = max(diagnostics, key=lambda item: item["mean_foreground_iou"])
    return float(best["threshold"]), diagnostics


def evaluate_manifest(
    manifest: ManifestLike,
    *,
    prediction_manifest: Optional[ManifestLike] = None,
    ground_truth_root: Optional[Union[str, Path]] = None,
    prediction_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluate precomputed binary predictions under a validated protocol.

    The result is JSON-serializable.  Per-frame metrics are averaged within
    each scene, and dataset metrics are the unweighted mean of scene metrics.
    Missing predictions are always fatal.
    """

    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        raw_manifest = load_json_manifest(manifest_path)
        base_dir = manifest_path.parent
    elif isinstance(manifest, Mapping):
        raw_manifest = copy.deepcopy(dict(manifest))
        base_dir = Path.cwd()
    else:
        raise ProtocolError("manifest must be a JSON path or mapping")

    normalized = validate_manifest(raw_manifest)
    protocol = normalized["protocol"]
    protocol_hash = normalized["protocol_hash"]
    gt_root = _resolve_root(
        ground_truth_root,
        raw_manifest.get("ground_truth_root"),
        base_dir=base_dir,
    )
    inline_pred_root = _resolve_root(
        prediction_root,
        raw_manifest.get("prediction_root"),
        base_dir=base_dir,
    )
    overrides, override_hashes, override_pred_root = _load_prediction_overrides(
        prediction_manifest,
        expected_protocol_hash=protocol_hash,
        default_base_dir=inline_pred_root,
    )

    threshold_policy = protocol["threshold"]
    threshold_by_scene: Dict[str, float] = {}
    calibration_diagnostics: Dict[str, Any] = {}
    if threshold_policy["mode"] == "fixed":
        for scene in normalized["scenes"]:
            threshold_by_scene[scene["scene_id"]] = float(threshold_policy["value"])
    elif threshold_policy["scope"] == "dataset":
        examples = []
        source_records = []
        for scene in normalized["scenes"]:
            for frame_id in scene["threshold_source_frame_ids"]:
                scores, target, pred_path, gt_path = _frame_inputs(
                    scene,
                    frame_id,
                    overrides=overrides,
                    override_hashes=override_hashes,
                    ground_truth_root=gt_root,
                    inline_prediction_root=inline_pred_root,
                    override_prediction_root=override_pred_root,
                )
                examples.append((scores, target))
                source_records.append(
                    {
                        "scene_id": scene["scene_id"],
                        "frame_id": frame_id,
                        "prediction": str(pred_path),
                        "ground_truth": str(gt_path),
                    }
                )
        threshold, candidate_metrics = _choose_threshold(
            examples,
            candidates=threshold_policy["candidates"],
            empty_union_value=protocol["empty_union_value"],
            prediction_representation=protocol["prediction_representation"],
            threshold_comparison=protocol["threshold_comparison"],
        )
        for scene in normalized["scenes"]:
            threshold_by_scene[scene["scene_id"]] = threshold
        calibration_diagnostics["dataset"] = {
            "threshold": threshold,
            "source_frames": source_records,
            "candidates": candidate_metrics,
        }
    else:
        for scene in normalized["scenes"]:
            examples = []
            source_records = []
            for frame_id in scene["threshold_source_frame_ids"]:
                scores, target, pred_path, gt_path = _frame_inputs(
                    scene,
                    frame_id,
                    overrides=overrides,
                    override_hashes=override_hashes,
                    ground_truth_root=gt_root,
                    inline_prediction_root=inline_pred_root,
                    override_prediction_root=override_pred_root,
                )
                examples.append((scores, target))
                source_records.append(
                    {
                        "frame_id": frame_id,
                        "prediction": str(pred_path),
                        "ground_truth": str(gt_path),
                    }
                )
            threshold, candidate_metrics = _choose_threshold(
                examples,
                candidates=threshold_policy["candidates"],
                empty_union_value=protocol["empty_union_value"],
                prediction_representation=protocol["prediction_representation"],
                threshold_comparison=protocol["threshold_comparison"],
            )
            threshold_by_scene[scene["scene_id"]] = threshold
            calibration_diagnostics[scene["scene_id"]] = {
                "threshold": threshold,
                "source_frames": source_records,
                "candidates": candidate_metrics,
            }

    scene_results = []
    for scene in normalized["scenes"]:
        scene_id = scene["scene_id"]
        threshold = threshold_by_scene[scene_id]
        frame_results = []
        for frame_id in scene["evaluation_frame_ids"]:
            scores, target, pred_path, gt_path = _frame_inputs(
                scene,
                frame_id,
                overrides=overrides,
                override_hashes=override_hashes,
                ground_truth_root=gt_root,
                inline_prediction_root=inline_pred_root,
                override_prediction_root=override_pred_root,
            )
            metrics = evaluate_binary_scores(
                scores,
                target,
                threshold=threshold,
                empty_union_value=protocol["empty_union_value"],
                resize=protocol["resize"],
                prediction_representation=protocol["prediction_representation"],
                threshold_comparison=protocol["threshold_comparison"],
            )
            frame_results.append(
                {
                    "frame_id": frame_id,
                    "threshold": threshold,
                    "prediction": str(pred_path),
                    "ground_truth": str(gt_path),
                    **metrics,
                }
            )
        scene_metrics = {
            name: float(np.mean([frame[name] for frame in frame_results]))
            for name in METRIC_NAMES
        }
        scene_results.append(
            {
                "scene_id": scene_id,
                "num_frames": len(frame_results),
                "threshold": threshold,
                "frames": frame_results,
                **scene_metrics,
            }
        )

    dataset_metrics = {
        name: float(np.mean([scene[name] for scene in scene_results]))
        for name in METRIC_NAMES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "thresholds": {
            "policy": threshold_policy,
            "by_scene": threshold_by_scene,
            "calibration": calibration_diagnostics,
        },
        "scenes": scene_results,
        "dataset": {
            "num_scenes": len(scene_results),
            "num_frames": sum(scene["num_frames"] for scene in scene_results),
            **dataset_metrics,
        },
    }


__all__ = [
    "AGGREGATION",
    "METRIC_NAMES",
    "MissingPredictionError",
    "ProtocolError",
    "ProtocolHashMismatchError",
    "RESIZE_POLICY",
    "SCHEMA_VERSION",
    "TASK_NAME",
    "compute_binary_metrics",
    "compute_protocol_hash",
    "evaluate_binary_scores",
    "evaluate_manifest",
    "load_ground_truth_mask",
    "load_json_manifest",
    "load_prediction_scores",
    "resize_mask_nearest",
    "validate_manifest",
]
