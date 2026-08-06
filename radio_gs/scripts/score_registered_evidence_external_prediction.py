#!/usr/bin/env python3
"""Score one hash-bound registered-evidence receipt after a strict GT barrier.

Phase one validates the complete prediction-side receipt, protocol/scene/frame
identity, and every target raw-score/calibrated-score/support array.  Ground
truth is first opened only after that phase returns successfully.  The metric
path delegates to the frozen NVOS frame scorer, whose semantics are
``cv2.INTER_LINEAR``, ``probability >= 0.5``, foreground IoU, and pixel
accuracy.  The same manifest topology supports NVOS and SPIn-NeRF.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np
from PIL import Image

from radio_gs.evaluation.promptable_segmentation import validate_manifest
from radio_gs.querying.registered_evidence_external_adapter import (
    ADAPTER_SCHEMA,
    validate_external_execution_authority,
    validate_external_promotion_documents,
)
from radio_gs.scripts.score_nvos_sealed_prediction_batch import (
    _score_frame as _frozen_score_frame,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    stable_descriptor_load,
    validate_file_record,
    write_frozen_json,
)


RECEIPT_TYPE = "registered_evidence_v2_pre_metric_prediction_receipt"
RESULT_TYPE = "registered_evidence_external_prediction_exact_score_v1"
SUPPORTED_BENCHMARKS = frozenset({"NVOS", "SPIn-NeRF"})


@dataclass(frozen=True)
class VerifiedTargetFrame:
    frame_id: str
    raw_probability: np.ndarray
    calibrated_probability: np.ndarray
    support: np.ndarray
    records: Mapping[str, object]
    ground_truth_path: Path
    ground_truth_sha256: str


@dataclass(frozen=True)
class VerifiedExternalPrediction:
    benchmark: str
    scene_id: str
    frame_ids: tuple[str, ...]
    protocol_hash: str
    manifest_path: Path
    manifest_sha256: str
    receipt_path: Path
    receipt_sha256: str
    execution_authority: Mapping[str, object]
    frames: Mapping[str, VerifiedTargetFrame]


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_npy_handle(handle: BinaryIO) -> np.ndarray:
    payload = io.BytesIO(handle.read())
    value = np.load(payload, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError("prediction artifact must contain one NPY array")
    return np.asarray(value)


def _load_record_array(record: object, *, label: str) -> tuple[np.ndarray, Path]:
    item = _mapping(record, label=f"{label} record")
    if set(item) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    value, _, path = stable_descriptor_load(
        str(item["path"]),
        _load_npy_handle,
        expected_sha256=str(item["sha256"]),
        label=label,
    )
    return value, path


def _validate_probability(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.size == 0
        or not bool(np.isfinite(array).all())
        or bool(((array < 0) | (array > 1)).any())
    ):
        raise ValueError(f"{label} must be finite nonempty float32 probability [H,W]")
    return array


def _validate_support(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.uint8
        or array.ndim != 2
        or array.size == 0
        or not bool(((array == 0) | (array == 1)).all())
    ):
        raise ValueError("target support must be nonempty binary uint8 [H,W]")
    return array.astype(bool, copy=False)


def _resolve_ground_truth_root(raw_manifest: Mapping[str, object], base: Path) -> Path:
    value = raw_manifest.get("ground_truth_root")
    if value is None:
        return base
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_ground_truth(path: object, *, root: Path) -> Path:
    value = Path(str(path)).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def _validate_method_contract(value: object) -> None:
    method = _mapping(value, label="external prediction method contract")
    expected = {
        "primitive": "RegisteredEvidenceToUnaryV1 graph-off full global rows",
        "renderer": "frozen alpha-normalized scalar Gaussian compositor",
        "calibration": "GlobalPromptLogitCalibratorV2 after render on supported pixels only",
        "unsupported_policy": "exact zero before and after calibration",
        "graph": False,
        "connected_selection": False,
        "per_scene_parameters": False,
        "threshold_scan": False,
    }
    if dict(method) != expected:
        raise ValueError("external prediction method contract differs")


def _validate_receipt_header(receipt: Mapping[str, object]) -> None:
    if (
        receipt.get("schema") != ADAPTER_SCHEMA
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != RECEIPT_TYPE
        or receipt.get("status") != "sealed_before_target_ground_truth_open"
        or receipt.get("sealed_before_target_ground_truth_open") is not True
        or any(
            receipt.get(key) is not False
            for key in (
                "target_rgb_opened",
                "target_mask_opened",
                "target_metric_computed",
            )
        )
    ):
        raise ValueError("external prediction receipt is not pre-GT")
    declared_content = str(receipt.get("content_sha256", ""))
    content = dict(receipt)
    content.pop("content_sha256", None)
    if canonical_json_sha256(content) != declared_content:
        raise ValueError("external prediction receipt content digest differs")
    _validate_method_contract(receipt.get("method_contract"))


def _validate_receipt_provenance(receipt: Mapping[str, object]) -> None:
    validate_file_record(receipt.get("primitive_unary"), label="primitive unary")
    for section_name, expected_keys in (
        ("carrier", ("config", "checkpoint", "camera_map")),
        ("checkpoints", ("v1", "v2")),
        ("implementation", ("core", "materializer")),
    ):
        section = _mapping(receipt.get(section_name), label=section_name)
        if set(section) != set(expected_keys):
            raise ValueError(f"receipt {section_name} records differ")
        for name in expected_keys:
            validate_file_record(section[name], label=f"receipt {section_name}.{name}")


def verify_prediction_barrier(
    *,
    prediction_receipt: str | Path,
    prediction_receipt_sha256: str,
    manifest: str | Path,
    manifest_sha256: str,
) -> VerifiedExternalPrediction:
    """Exhaust prediction/protocol validation without opening any GT file."""

    receipt, receipt_sha, receipt_path = load_json_object(
        prediction_receipt,
        expected_sha256=prediction_receipt_sha256,
        label="registered-evidence pre-GT receipt",
    )
    _validate_receipt_header(receipt)
    authority_record = _mapping(
        receipt.get("execution_authority"), label="receipt execution authority"
    )
    authority_path = validate_file_record(
        authority_record, label="external execution authority"
    )
    authority, _, _ = load_json_object(
        authority_path,
        expected_sha256=str(authority_record["sha256"]),
        label="external execution authority",
    )
    authority = validate_external_execution_authority(authority)
    promotion = _mapping(authority["promotion"], label="promotion records")
    promotion_paths = {
        name: validate_file_record(promotion[name], label=f"promotion {name}")
        for name in (
            "v1_result",
            "v2_result",
            "cross_scene_confirmation_receipt",
            "cross_scene_confirmation_result",
        )
    }
    v2_source_result, _, _ = load_json_object(
        promotion_paths["v2_result"],
        expected_sha256=str(promotion["v2_result"]["sha256"]),
        label="clean V2 source result",
    )
    cross_scene_receipt, _, _ = load_json_object(
        promotion_paths["cross_scene_confirmation_receipt"],
        expected_sha256=str(promotion["cross_scene_confirmation_receipt"]["sha256"]),
        label="cross-scene promotion receipt",
    )
    cross_scene_result, _, _ = load_json_object(
        promotion_paths["cross_scene_confirmation_result"],
        expected_sha256=str(promotion["cross_scene_confirmation_result"]["sha256"]),
        label="cross-scene promotion result",
    )
    validate_external_promotion_documents(
        v2_source_result=v2_source_result,
        cross_scene_receipt=cross_scene_receipt,
        cross_scene_result=cross_scene_result,
        expected_cross_scene_result_sha256=str(
            promotion["cross_scene_confirmation_result"]["sha256"]
        ),
    )

    raw_manifest, verified_manifest_sha, manifest_path = load_json_object(
        manifest,
        expected_sha256=manifest_sha256,
        label="frozen promptable manifest",
    )
    normalized = validate_manifest(raw_manifest)
    protocol = _mapping(normalized.get("protocol"), label="normalized protocol")
    benchmark = str(protocol.get("benchmark", ""))
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError("external scorer supports only NVOS and SPIn-NeRF manifests")
    if (
        protocol.get("task") != "promptable_nvs_binary_segmentation"
        or protocol.get("metrics") != ["foreground_iou", "pixel_accuracy"]
        or protocol.get("aggregation")
        != "per_frame_then_per_scene_then_dataset_scene_macro"
        or protocol.get("allow_reference_scoring") is not False
    ):
        raise ValueError("frozen promptable protocol semantics differ")
    protocol_hash = str(normalized["protocol_hash"])
    if (
        receipt.get("protocol_hash") != protocol_hash
        or authority.get("protocol_hash") != protocol_hash
    ):
        raise ValueError("receipt/authority/manifest protocol hashes differ")
    authority_manifest = _mapping(
        _mapping(authority["records"], label="authority records")["manifest"],
        label="authority manifest record",
    )
    if (
        Path(str(authority_manifest["path"])).expanduser().resolve() != manifest_path
        or str(authority_manifest["sha256"]) != verified_manifest_sha
    ):
        raise ValueError("execution authority binds a different manifest")

    scene_id = str(receipt.get("scene_id", ""))
    if scene_id != str(authority.get("scene_id", "")):
        raise ValueError("receipt and execution-authority scenes differ")
    matching_scenes = [
        scene for scene in normalized["scenes"] if str(scene["scene_id"]) == scene_id
    ]
    if len(matching_scenes) != 1:
        raise ValueError("manifest scene identity is ambiguous")
    scene = matching_scenes[0]
    frame_ids = tuple(str(value) for value in scene["evaluation_frame_ids"])
    authority_frames = tuple(str(value) for value in authority["target_frame_ids"])
    source_frame = str(authority["source_frame_id"])
    if authority_frames != frame_ids or tuple(
        str(value) for value in scene["prompt_frame_ids"]
    ) != (source_frame,):
        raise ValueError("manifest and execution-authority frame sets differ")
    receipt_frames = _mapping(receipt.get("frames"), label="receipt target frames")
    if set(receipt_frames) != set(frame_ids) or int(
        receipt.get("frame_count", -1)
    ) != len(frame_ids):
        raise ValueError("receipt and manifest target frame sets differ")

    _validate_receipt_provenance(receipt)
    ground_truth_root = _resolve_ground_truth_root(raw_manifest, manifest_path.parent)
    verified_frames: dict[str, VerifiedTargetFrame] = {}
    for frame_id in frame_ids:
        record = _mapping(
            receipt_frames[frame_id], label=f"target frame {frame_id} receipt"
        )
        required = {
            "raw_v1_probability",
            "supported",
            "calibrated_v2_probability",
            "shape",
            "supported_pixels",
            "strict_domain_audit",
        }
        if set(record) != required:
            raise ValueError(f"target frame {frame_id} receipt fields differ")
        raw, _ = _load_record_array(
            record["raw_v1_probability"], label=f"target raw score {frame_id}"
        )
        calibrated, _ = _load_record_array(
            record["calibrated_v2_probability"],
            label=f"target calibrated score {frame_id}",
        )
        support_value, _ = _load_record_array(
            record["supported"], label=f"target support {frame_id}"
        )
        raw = _validate_probability(raw, label=f"target raw score {frame_id}")
        calibrated = _validate_probability(
            calibrated, label=f"target calibrated score {frame_id}"
        )
        support = _validate_support(support_value)
        declared_shape = record["shape"]
        if (
            not isinstance(declared_shape, list)
            or declared_shape != list(raw.shape)
            or calibrated.shape != raw.shape
            or support.shape != raw.shape
        ):
            raise ValueError(f"target frame {frame_id} score/support shapes differ")
        supported_pixels = int(support.sum())
        audit = _mapping(
            record["strict_domain_audit"],
            label=f"target frame {frame_id} strict-domain audit",
        )
        if (
            int(record["supported_pixels"]) != supported_pixels
            or int(audit.get("count", -1)) != supported_pixels
            or supported_pixels <= 0
        ):
            raise ValueError(f"target frame {frame_id} support accounting differs")
        if bool(raw[~support].any()) or bool(calibrated[~support].any()):
            raise ValueError(f"target frame {frame_id} unsupported score is nonzero")
        frame_manifest = scene["frames"][frame_id]
        ground_truth = frame_manifest.get("ground_truth")
        ground_truth_sha = str(frame_manifest.get("ground_truth_sha256", ""))
        if not ground_truth or len(ground_truth_sha) != 64:
            raise ValueError(f"target frame {frame_id} lacks GT authority")
        verified_frames[frame_id] = VerifiedTargetFrame(
            frame_id=frame_id,
            raw_probability=raw,
            calibrated_probability=calibrated,
            support=support,
            records=dict(record),
            ground_truth_path=_resolve_ground_truth(
                ground_truth, root=ground_truth_root
            ),
            ground_truth_sha256=ground_truth_sha,
        )

    return VerifiedExternalPrediction(
        benchmark=benchmark,
        scene_id=scene_id,
        frame_ids=frame_ids,
        protocol_hash=protocol_hash,
        manifest_path=manifest_path,
        manifest_sha256=verified_manifest_sha,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        execution_authority=authority,
        frames=verified_frames,
    )


def _decode_ground_truth(handle: BinaryIO, *, suffix: str) -> np.ndarray:
    payload = io.BytesIO(handle.read())
    if suffix == ".npy":
        value = np.load(payload, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(payload, allow_pickle=False) as archive:
            names = list(archive.files)
            if "mask" in names:
                value = archive["mask"]
            elif len(names) == 1:
                value = archive[names[0]]
            else:
                raise ValueError("GT NPZ must contain one array or 'mask'")
    else:
        with Image.open(payload) as image:
            value = np.asarray(image.convert("L"))
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.size == 0:
        raise ValueError("ground truth must be a nonempty scalar 2-D mask")
    return array.astype(bool, copy=False)


def _load_ground_truth_after_barrier(frame: VerifiedTargetFrame) -> np.ndarray:
    value, _, _ = stable_descriptor_load(
        frame.ground_truth_path,
        lambda handle: _decode_ground_truth(
            handle, suffix=frame.ground_truth_path.suffix.lower()
        ),
        expected_sha256=frame.ground_truth_sha256,
        label=f"target ground truth {frame.frame_id}",
    )
    return value


def _score_external_frame(
    probability: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    """Delegate exactly to the frozen scorer at the fixed probability threshold."""

    return _frozen_score_frame(probability, ground_truth, 0.5)


def score_verified_prediction(
    verified: VerifiedExternalPrediction,
) -> dict[str, object]:
    """Open GT only after a complete :func:`verify_prediction_barrier`."""

    frame_metrics: list[dict[str, object]] = []
    for frame_id in verified.frame_ids:
        frame = verified.frames[frame_id]
        ground_truth = _load_ground_truth_after_barrier(frame)
        metrics = _score_external_frame(frame.calibrated_probability, ground_truth)
        frame_metrics.append(
            {
                "frame_id": frame_id,
                **metrics,
                "supported_fraction": float(frame.support.mean()),
                "score_records": dict(frame.records),
                "ground_truth": {
                    "path": str(frame.ground_truth_path),
                    "sha256": frame.ground_truth_sha256,
                },
            }
        )
    scene_iou = float(np.mean([row["foreground_iou"] for row in frame_metrics]))
    scene_accuracy = float(np.mean([row["pixel_accuracy"] for row in frame_metrics]))
    return {
        "schema_version": 1,
        "artifact_type": RESULT_TYPE,
        "benchmark": verified.benchmark,
        "scene_id": verified.scene_id,
        "protocol_hash": verified.protocol_hash,
        "manifest": {
            "path": str(verified.manifest_path),
            "sha256": verified.manifest_sha256,
        },
        "prediction_receipt": {
            "path": str(verified.receipt_path),
            "sha256": verified.receipt_sha256,
        },
        "implementation": {
            "scorer": file_record(Path(__file__).resolve()),
            "frozen_frame_scorer": file_record(
                Path(__file__).resolve().parent
                / "score_nvos_sealed_prediction_batch.py"
            ),
        },
        "frames": frame_metrics,
        "scene_macro": {
            "foreground_iou": scene_iou,
            "pixel_accuracy": scene_accuracy,
            "frame_count": len(frame_metrics),
        },
        "evaluation_protocol": {
            "input": "calibrated_v2_probability",
            "score_resize": "cv2.INTER_LINEAR",
            "threshold": 0.5,
            "comparison": "greater_or_equal",
            "empty_union_value": 1.0,
            "aggregation": "per_frame_then_scene_macro",
        },
        "safety": {
            "all_target_score_and_support_files_loaded_before_first_ground_truth_open": True,
            "protocol_and_frame_set_verified_before_first_ground_truth_open": True,
            "prediction_changed_after_receipt": False,
            "target_metrics_used_for_method_selection": False,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prediction-receipt", required=True)
    result.add_argument("--prediction-receipt-sha256", required=True)
    result.add_argument("--manifest", required=True)
    result.add_argument("--manifest-sha256", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite external score: {output}")
    verified = verify_prediction_barrier(
        prediction_receipt=args.prediction_receipt,
        prediction_receipt_sha256=args.prediction_receipt_sha256,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
    )
    payload = score_verified_prediction(verified)
    write_frozen_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "scene_id": verified.scene_id,
                **payload["scene_macro"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
