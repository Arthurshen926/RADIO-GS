#!/usr/bin/env python3
"""Fail-closed validation and exact replay of the frozen ScanNet paper8 result."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts import eval_ours_scannet_vala_gaussian_protocol as evaluator
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    volume_weighted_split_metrics,
)
from radio_gs.scripts.validate_scannet_canonical_mpr_v3_totality_freeze import (
    V2_EXPECTED_FREEZE_SHA256,
    validate_freeze,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHORITY = (
    ROOT
    / "paper/artifacts/scannet_canonical_mpr_v3_gaussian_semantic_result_authority_20260802.json"
)
EXPECTED_AUTHORITY_SHA256 = (
    "ac98a1f9a678b237006051fc57a83fdf0f7792adfbd1baccf4a56b80d8a2b13c"
)
ARTIFACT_TYPE = (
    "radio_gs_scannet_canonical_mpr_v3_gaussian_semantic_result_authority"
)
CONTRACT = "radio_gs.scannet_canonical_mpr_v3_gaussian_semantic_result_authority.v1"
EXPECTED_SCENES = [
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
]
SPLITS = ["19", "15", "10"]
FILE_RECORD_KEYS = {"path", "bytes", "sha256"}
SCENE_RECORD_KEYS = {
    "num_gaussians",
    "region_observed_count",
    "no_evidence_fallback_count",
    "semantic_score_cache",
    "semantic_score_cache_receipt",
    "pseudo_gt",
    "prediction",
}
RECEIPT_KEYS = {
    "canonical_field_source",
    "canonical_mainline",
    "canonical_method_freeze",
    "config_source",
    "geometry_checkpoint",
    "materializer_contract",
    "method_family",
    "mpr_source",
    "num_classes",
    "num_gaussians",
    "official_radio_source",
    "producer_source",
    "protocol_freeze",
    "protocol_freeze_id",
    "protocol_freeze_task",
    "protocol_registry_row",
    "query_source",
    "row_tensor_sha256",
    "scene_id",
    "semantic_score_cache",
    "status",
    "support_graph_source",
    "surface_region_readout_source",
}
RECEIPT_TRANSITIVE_ROLES = {
    "canonical_field_source",
    "canonical_mainline",
    "canonical_method_freeze",
    "geometry_checkpoint",
    "mpr_source",
    "official_radio_source",
    "producer_source",
    "protocol_freeze",
    "query_source",
    "support_graph_source",
    "surface_region_readout_source",
}
CACHE_KEYS = {
    "schema_version",
    "artifact_type",
    "metadata",
    "class_ids",
    "class_names",
    "query_ids",
    "xyz",
    "scale",
    "quaternion",
    "opacity",
    "valid",
    "region_observed",
    "semantic_scores",
}
CACHE_METADATA_KEYS = {
    "benchmark_images_opened",
    "benchmark_labels_opened",
    "benchmark_masks_opened",
    "benchmark_metrics_opened",
    "canonical_field_geometry_row_match",
    "canonical_field_source",
    "canonical_mainline",
    "canonical_mainline_name",
    "canonical_mainline_sha256",
    "canonical_method_freeze",
    "canonical_method_freeze_name",
    "canonical_method_freeze_sha256",
    "class_order_sha256",
    "compact_feature_key",
    "config_source",
    "diagnostic_only",
    "gaussian_query_position",
    "geometry_authority_activation_device",
    "geometry_checkpoint",
    "geometry_checkpoint_sha256",
    "knn_used",
    "logit_calibration",
    "logit_smoothing",
    "materializer_contract",
    "mesh_vertices_used",
    "method_family",
    "mpr_source",
    "no_evidence_fallback",
    "no_evidence_fallback_count",
    "official_radio_checkpoint_sha256",
    "official_radio_source",
    "prediction_domain",
    "producer_source",
    "producer_source_sha256",
    "protocol_contract",
    "protocol_freeze",
    "protocol_freeze_id",
    "protocol_freeze_sha256",
    "protocol_freeze_task",
    "protocol_registry_row",
    "query_class_order_sha256",
    "query_set_calibration",
    "query_source",
    "query_source_sha256",
    "query_text_sha256",
    "region_graph_geometry_row_match",
    "region_observed_count",
    "region_radii_m",
    "region_scale_aggregation",
    "row_order",
    "row_tensor_sha256",
    "scene_id",
    "score_formula",
    "semantic_readout",
    "semantic_source",
    "semantic_source_sha256",
    "spatial_transfer",
    "support_graph_source",
    "surface_region_readout_sha256",
    "surface_region_readout_source",
    "totality_contract",
    "totality_semantics",
}


class ResultAuthorityError(ValueError):
    """Raised when any final-result authority, input, output, or replay drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultAuthorityError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultAuthorityError(f"{label} must be a mapping")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultAuthorityError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultAuthorityError(f"{label} is not canonical UTF-8 JSON") from error
    return _mapping(payload, label)


def _sha256_bytes(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _read_regular_file(path: Path, *, label: str) -> bytes:
    _require(path.is_absolute(), f"{label} path must be absolute")
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or a symlink")
    before = path.stat()
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ResultAuthorityError(f"cannot read {label}: {path}") from error
    after = path.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while being read",
    )
    _require(len(encoded) == after.st_size, f"{label} byte size changed while being read")
    return encoded


def _load_authority(path: Path) -> tuple[Mapping[str, Any], bytes]:
    encoded = _read_regular_file(path, label="result authority")
    _require(
        _sha256_bytes(encoded) == EXPECTED_AUTHORITY_SHA256,
        "result authority SHA256 differs; create a new immutable version instead of mutating it",
    )
    return _decode_json(encoded, label="result authority"), encoded


def _read_file_record(value: Any, *, label: str) -> tuple[Path, bytes]:
    record = _mapping(value, label)
    _require(set(record) == FILE_RECORD_KEYS, f"{label} file-record schema differs")
    path = Path(str(record.get("path", "")))
    encoded = _read_regular_file(path, label=label)
    _require(len(encoded) == record.get("bytes"), f"{label} byte size drifted")
    _require(_sha256_bytes(encoded) == record.get("sha256"), f"{label} SHA256 drifted")
    return path, encoded


def _validate_source_record(
    value: Any,
    *,
    label: str,
    hash_cache: dict[Path, str],
) -> Path:
    record = _mapping(value, label)
    _require(set(record) == {"path", "sha256"}, f"{label} source-record schema differs")
    path = Path(str(record.get("path", "")))
    _require(path.is_absolute(), f"{label} source path must be absolute")
    _require(path.is_file() and not path.is_symlink(), f"{label} source is missing or a symlink")
    if path not in hash_cache:
        before = path.stat()
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
        except OSError as error:
            raise ResultAuthorityError(f"cannot hash {label} source: {path}") from error
        after = path.stat()
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label} source changed while being hashed",
        )
        hash_cache[path] = digest.hexdigest()
    _require(hash_cache[path] == record.get("sha256"), f"{label} source SHA256 drifted")
    return path


def _validate_payload_semantics(payload: Mapping[str, Any]) -> None:
    _require(
        set(payload)
        == {
            "schema_version",
            "artifact_type",
            "contract",
            "status",
            "frozen_at",
            "benchmark_binding",
            "method_binding",
            "derivation_binding",
            "totality_freeze",
            "evaluator_report",
            "metrics",
            "scenes",
        },
        "result authority top-level schema differs",
    )
    _require(payload.get("schema_version") == 1, "result authority schema version differs")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "result authority type differs")
    _require(payload.get("contract") == CONTRACT, "result authority contract differs")
    _require(payload.get("status") == "formal_result_frozen_exact", "result status differs")
    _require(payload.get("frozen_at") == "2026-08-02", "result freeze date differs")

    benchmark = _mapping(payload.get("benchmark_binding"), "benchmark binding")
    _require(
        dict(benchmark)
        == {
            "benchmark": "ScanNet OVS",
            "canonical_task_id": evaluator.EXTERNAL_PROTOCOL_FREEZE_TASK,
            "registry_row": evaluator.EXTERNAL_PROTOCOL_REGISTRY_ROW,
            "external_protocol_freeze_id": evaluator.EXTERNAL_PROTOCOL_FREEZE_ID,
            "cohort": "paper8",
            "scenes": EXPECTED_SCENES,
            "splits": SPLITS,
        },
        "paper8 benchmark binding differs",
    )
    method = _mapping(payload.get("method_binding"), "method binding")
    _require(
        dict(method)
        == {
            "method_family": evaluator.CURRENT_METHOD_FAMILY,
            "mainline_name": evaluator.CANONICAL_MAINLINE_NAME,
            "method_freeze_name": evaluator.CANONICAL_METHOD_FREEZE_NAME,
            "totality_contract": evaluator.CANONICAL_TOTALITY_CONTRACT,
            "materializer_contract": evaluator.CURRENT_MATERIALIZER_CONTRACT,
            "semantic_score_cache_contract": evaluator.PROTOCOL_CONTRACT,
            "prediction_domain": evaluator.PREDICTION_DOMAIN,
            "semantic_readout": evaluator.SEMANTIC_READOUT,
            "spatial_transfer": evaluator.SPATIAL_TRANSFER,
        },
        "canonical-mpr-v3 method binding differs",
    )
    derivation = _mapping(payload.get("derivation_binding"), "derivation binding")
    _require(
        dict(derivation)
        == {
            "pseudo_gt": "VALA anisotropic Mahalanobis-density vote",
            "metric_weights": "activated_opacity * activated_scale.prod()",
            "class_aggregation": "present classes within each scene",
            "scene_aggregation": "unweighted scene macro",
            "prediction_rule": (
                "restrict_the_single_exact_split19_score_bank_to_each_frozen_split_then_per_row_argmax"
            ),
            "required_scene_evidence": [
                "semantic_score_cache",
                "semantic_score_cache_receipt",
                "pseudo_gt",
                "prediction",
            ],
            "receipt_transitive_sources": [
                "config_source",
                "geometry_checkpoint",
                "canonical_field_source",
                "mpr_source",
                "support_graph_source",
            ],
            "exact_array_replay_required": True,
            "exact_metric_replay_required": True,
        },
        "exact result derivation binding differs",
    )
    for role in ("totality_freeze", "evaluator_report"):
        record = _mapping(payload.get(role), role)
        _require(set(record) == FILE_RECORD_KEYS, f"{role} file-record schema differs")
    metrics = _mapping(payload.get("metrics"), "frozen metrics")
    _require(list(metrics) == SPLITS, "frozen metric split order differs")
    for split, values in metrics.items():
        item = _mapping(values, f"frozen split {split} metrics")
        _require(set(item) == {"miou", "macc"}, f"split {split} metric schema differs")
        _require(
            all(isinstance(item[key], float) and np.isfinite(item[key]) for key in item),
            f"split {split} metrics are not finite floats",
        )
    scenes = _mapping(payload.get("scenes"), "scene evidence")
    _require(list(scenes) == EXPECTED_SCENES, "paper8 scene evidence/order differs")
    for scene, value in scenes.items():
        record = _mapping(value, f"{scene} evidence")
        _require(set(record) == SCENE_RECORD_KEYS, f"{scene} evidence schema differs")
        count = record.get("num_gaussians")
        observed = record.get("region_observed_count")
        fallback = record.get("no_evidence_fallback_count")
        _require(
            isinstance(count, int)
            and isinstance(observed, int)
            and isinstance(fallback, int)
            and count > 0
            and observed > 0
            and fallback >= 0
            and observed + fallback == count,
            f"{scene} totality counts differ",
        )
        for role in (
            "semantic_score_cache",
            "semantic_score_cache_receipt",
            "pseudo_gt",
            "prediction",
        ):
            child = _mapping(record.get(role), f"{scene} {role}")
            _require(set(child) == FILE_RECORD_KEYS, f"{scene} {role} file-record schema differs")


def _load_bound_receipt(
    encoded: bytes,
    *,
    scene: str,
    scene_evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    receipt = _decode_json(encoded, label=f"{scene} semantic score receipt")
    _require(set(receipt) == RECEIPT_KEYS, f"{scene} score receipt schema differs")
    expected_scalars = {
        "scene_id": scene,
        "status": "complete_immutable_gaussian_semantic_score_cache",
        "method_family": evaluator.CURRENT_METHOD_FAMILY,
        "materializer_contract": evaluator.CURRENT_MATERIALIZER_CONTRACT,
        "protocol_freeze_id": evaluator.EXTERNAL_PROTOCOL_FREEZE_ID,
        "protocol_freeze_task": evaluator.EXTERNAL_PROTOCOL_FREEZE_TASK,
        "protocol_registry_row": evaluator.EXTERNAL_PROTOCOL_REGISTRY_ROW,
        "num_classes": len(evaluator.PAPER_CLASS_IDS),
        "num_gaussians": scene_evidence["num_gaussians"],
    }
    for key, expected in expected_scalars.items():
        _require(receipt.get(key) == expected, f"{scene} receipt {key} differs")
    final_cache = _mapping(scene_evidence["semantic_score_cache"], f"{scene} cache")
    _require(
        receipt.get("semantic_score_cache")
        == {"path": final_cache["path"], "sha256": final_cache["sha256"]},
        f"{scene} receipt/final cache binding differs",
    )
    return receipt


def _validate_cache_and_receipt(
    *,
    scene: str,
    scene_evidence: Mapping[str, Any],
    cache_encoded: bytes,
    receipt: Mapping[str, Any],
    source_hash_cache: dict[Path, str],
) -> dict[str, Any]:
    try:
        payload = torch.load(BytesIO(cache_encoded), map_location="cpu", weights_only=False)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ResultAuthorityError(f"{scene} semantic score cache cannot be loaded: {error}") from error
    cache = _mapping(payload, f"{scene} semantic score cache")
    _require(set(cache) == CACHE_KEYS, f"{scene} semantic score cache schema differs")
    _require(cache.get("schema_version") == evaluator.SCHEMA_VERSION, f"{scene} cache schema version differs")
    _require(cache.get("artifact_type") == evaluator.ARTIFACT_TYPE, f"{scene} cache artifact type differs")
    metadata = _mapping(cache.get("metadata"), f"{scene} cache metadata")
    _require(set(metadata) == CACHE_METADATA_KEYS, f"{scene} cache metadata schema differs")
    expected_metadata = {
        "scene_id": scene,
        "protocol_contract": evaluator.PROTOCOL_CONTRACT,
        "prediction_domain": evaluator.PREDICTION_DOMAIN,
        "row_order": evaluator.ROW_ORDER,
        "semantic_readout": evaluator.SEMANTIC_READOUT,
        "spatial_transfer": evaluator.SPATIAL_TRANSFER,
        "mesh_vertices_used": False,
        "knn_used": False,
        "query_text_sha256": evaluator.QUERY_TEXT_SHA256,
        "class_order_sha256": evaluator.CLASS_ORDER_SHA256,
        "query_class_order_sha256": evaluator.QUERY_CLASS_ORDER_SHA256,
        "method_family": evaluator.CURRENT_METHOD_FAMILY,
        "materializer_contract": evaluator.CURRENT_MATERIALIZER_CONTRACT,
        "protocol_freeze_id": evaluator.EXTERNAL_PROTOCOL_FREEZE_ID,
        "protocol_freeze_task": evaluator.EXTERNAL_PROTOCOL_FREEZE_TASK,
        "protocol_registry_row": evaluator.EXTERNAL_PROTOCOL_REGISTRY_ROW,
        "protocol_freeze_sha256": evaluator.EXTERNAL_PROTOCOL_FREEZE_SHA256,
        "canonical_mainline_name": evaluator.CANONICAL_MAINLINE_NAME,
        "canonical_mainline_sha256": evaluator.CANONICAL_MAINLINE_SHA256,
        "canonical_method_freeze_name": evaluator.CANONICAL_METHOD_FREEZE_NAME,
        "canonical_method_freeze_sha256": evaluator.CANONICAL_METHOD_FREEZE_SHA256,
        "surface_region_readout_sha256": evaluator.CANONICAL_READOUT_SHA256,
        "official_radio_checkpoint_sha256": evaluator.OFFICIAL_RADIO_SHA256,
        "region_radii_m": evaluator.CANONICAL_REGION_RADII_M,
        "query_set_calibration": False,
        "logit_calibration": "none",
        "logit_smoothing": "none",
        "canonical_field_geometry_row_match": True,
        "region_graph_geometry_row_match": True,
        "region_scale_aggregation": "max_independent_cosine_over_0.20_0.40_0.70",
        "totality_contract": evaluator.CANONICAL_TOTALITY_CONTRACT,
        "totality_semantics": "graph_observed_surface_region_h128_else_exact_canonical_field_primitive",
        "no_evidence_fallback": "canonical_field_primitive_official_summary_head_independent_cosine",
        "diagnostic_only": False,
        "geometry_authority_activation_device": "cpu",
        "gaussian_query_position": "optimized_gaussian_center",
        "compact_feature_key": "features",
        "score_formula": (
            "l2_normalize(canonical_mpr_v3_surface_region_descriptor) @ "
            "l2_normalize(exact_split19_text_embedding).T"
        ),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_metrics_opened": False,
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"{scene} cache metadata.{key} differs")
    _require(
        metadata.get("region_observed_count") == scene_evidence["region_observed_count"]
        and metadata.get("no_evidence_fallback_count")
        == scene_evidence["no_evidence_fallback_count"],
        f"{scene} cache/final totality counts differ",
    )
    for role in RECEIPT_TRANSITIVE_ROLES:
        _require(receipt.get(role) == metadata.get(role), f"{scene} receipt/cache {role} binding differs")
    _require(receipt.get("config_source") == metadata.get("config_source"), f"{scene} receipt/cache config binding differs")
    _require(receipt.get("row_tensor_sha256") == metadata.get("row_tensor_sha256"), f"{scene} receipt/cache row hashes differ")
    for role in sorted(RECEIPT_TRANSITIVE_ROLES | {"config_source"}):
        _validate_source_record(
            receipt.get(role),
            label=f"{scene} {role}",
            hash_cache=source_hash_cache,
        )
    _require(
        metadata.get("geometry_checkpoint_sha256") == receipt["geometry_checkpoint"]["sha256"]
        and metadata.get("query_source_sha256") == receipt["query_source"]["sha256"]
        and metadata.get("semantic_source_sha256") == receipt["canonical_field_source"]["sha256"]
        and metadata.get("producer_source_sha256") == receipt["producer_source"]["sha256"],
        f"{scene} cache metadata/source digest binding differs",
    )

    count = int(scene_evidence["num_gaussians"])
    expected_shapes = {
        "xyz": (count, 3),
        "scale": (count, 3),
        "quaternion": (count, 4),
        "opacity": (count,),
        "valid": (count,),
        "region_observed": (count,),
        "semantic_scores": (count, len(evaluator.PAPER_CLASS_IDS)),
    }
    tensors: dict[str, torch.Tensor] = {}
    for key, shape in expected_shapes.items():
        value = cache.get(key)
        _require(isinstance(value, torch.Tensor), f"{scene} cache {key} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        _require(tuple(tensor.shape) == shape, f"{scene} cache {key} shape differs")
        if key in {"valid", "region_observed"}:
            _require(tensor.dtype == torch.bool, f"{scene} cache {key} dtype differs")
        else:
            _require(tensor.is_floating_point() and bool(torch.isfinite(tensor).all()), f"{scene} cache {key} contains invalid values")
        tensors[key] = tensor
    _require(bool(tensors["valid"].all()), f"{scene} semantic valid domain is not total")
    _require(int(tensors["region_observed"].sum()) == scene_evidence["region_observed_count"], f"{scene} region partition differs")
    _require(cache.get("class_ids") == list(evaluator.PAPER_CLASS_IDS), f"{scene} class-id order differs")
    _require(cache.get("class_names") == list(evaluator.PAPER_CLASS_NAMES), f"{scene} class-name order differs")
    _require(cache.get("query_ids") == list(evaluator.PAPER_CLASS_NAMES), f"{scene} query order differs")
    row_hashes = _mapping(metadata.get("row_tensor_sha256"), f"{scene} row hashes")
    _require(set(row_hashes) == set(expected_shapes), f"{scene} row-hash schema differs")
    for key, tensor in tensors.items():
        _require(row_hashes.get(key) == evaluator._tensor_sha256(tensor), f"{scene} cache {key} tensor SHA256 differs")
    return {**tensors, "metadata": dict(metadata)}


def _load_npz(encoded: bytes, *, label: str, expected_keys: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(BytesIO(encoded), allow_pickle=False) as archive:
            _require(set(archive.files) == expected_keys, f"{label} array schema differs")
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ResultAuthorityError):
            raise
        raise ResultAuthorityError(f"cannot load {label}: {error}") from error


def _exact_array(observed: np.ndarray, expected: np.ndarray, *, message: str) -> None:
    _require(
        observed.dtype == expected.dtype
        and observed.shape == expected.shape
        and np.array_equal(observed, expected),
        message,
    )


def _validate_report_protocol(report: Mapping[str, Any]) -> None:
    _require(
        set(report) == {"timestamp", "method", "protocol", "authority", "args", "macro", "scenes"},
        "evaluator report schema differs",
    )
    _require(report.get("method") == "RADIO-GS Ours explicit Gaussian semantic-score cache", "evaluator report method differs")
    protocol = _mapping(report.get("protocol"), "evaluator report protocol")
    expected_protocol = {
        "contract": evaluator.PROTOCOL_CONTRACT,
        "task": "ScanNet OVS text-query semantic segmentation",
        "prediction_domain": evaluator.PREDICTION_DOMAIN,
        "semantic_readout": evaluator.SEMANTIC_READOUT,
        "spatial_transfer": evaluator.SPATIAL_TRANSFER,
        "legacy_mesh_knn8": "forbidden",
        "pseudo_gt": "VALA anisotropic Mahalanobis-density vote",
        "metric_weights": "activated_opacity * activated_scale.prod()",
        "class_splits": SPLITS,
        "class_aggregation": "present classes within each scene",
        "scene_aggregation": "unweighted scene macro",
        "cohort": "paper8",
        "cohort_scenes": EXPECTED_SCENES,
        "cohort_status": "paper8_canonical_paper_facing",
        "cpu_only": True,
        "query_text_sha256": evaluator.QUERY_TEXT_SHA256,
        "class_order_sha256": evaluator.CLASS_ORDER_SHA256,
        "query_class_order_sha256": evaluator.QUERY_CLASS_ORDER_SHA256,
        "method_family": evaluator.CURRENT_METHOD_FAMILY,
        "materializer_contract": evaluator.CURRENT_MATERIALIZER_CONTRACT,
        "external_protocol_freeze_id": evaluator.EXTERNAL_PROTOCOL_FREEZE_ID,
        "external_protocol_freeze_task": evaluator.EXTERNAL_PROTOCOL_FREEZE_TASK,
        "external_protocol_registry_row": evaluator.EXTERNAL_PROTOCOL_REGISTRY_ROW,
        "external_protocol_freeze_sha256": evaluator.EXTERNAL_PROTOCOL_FREEZE_SHA256,
        "canonical_mainline_name": evaluator.CANONICAL_MAINLINE_NAME,
        "canonical_mainline_sha256": evaluator.CANONICAL_MAINLINE_SHA256,
        "canonical_method_freeze_name": evaluator.CANONICAL_METHOD_FREEZE_NAME,
        "canonical_method_freeze_sha256": evaluator.CANONICAL_METHOD_FREEZE_SHA256,
        "canonical_totality_contract": evaluator.CANONICAL_TOTALITY_CONTRACT,
    }
    _require(dict(protocol) == expected_protocol, "evaluator report frozen protocol differs")
    args = _mapping(report.get("args"), "evaluator report args")
    _require(
        args.get("cohort") == "paper8"
        and args.get("scenes") == "None"
        and args.get("candidate_k") == "1000"
        and args.get("fallback_k") == "1"
        and args.get("radius_factor") == "5.0"
        and args.get("force_pseudo_gt") == "False",
        "evaluator report protocol arguments differ",
    )


def validate_result_authority(path: str | Path = DEFAULT_AUTHORITY) -> dict[str, Any]:
    authority_path = Path(path).absolute()
    payload, encoded = _load_authority(authority_path)
    _validate_payload_semantics(payload)

    freeze_path, _ = _read_file_record(payload["totality_freeze"], label="totality freeze v2")
    _require(payload["totality_freeze"]["sha256"] == V2_EXPECTED_FREEZE_SHA256, "totality freeze v2 binding differs")
    try:
        freeze_result = validate_freeze(freeze_path)
    except ValueError as error:
        raise ResultAuthorityError(f"totality freeze validation failed: {error}") from error
    _require(freeze_result.get("artifact_version") == "v2", "formal result is not bound to totality v2")

    _, report_encoded = _read_file_record(payload["evaluator_report"], label="exact evaluator report")
    report = _decode_json(report_encoded, label="exact evaluator report")
    _validate_report_protocol(report)
    report_scenes = _mapping(report.get("scenes"), "evaluator report scenes")
    _require(list(report_scenes) == EXPECTED_SCENES, "evaluator report paper8 cohort differs")
    authority_scenes = _mapping(payload.get("scenes"), "authority scenes")

    replayed_scenes: dict[str, Mapping[str, Any]] = {}
    source_hash_cache: dict[Path, str] = {}
    total_rows = 0
    total_observed = 0
    total_fallback = 0
    for scene in EXPECTED_SCENES:
        evidence = _mapping(authority_scenes[scene], f"{scene} authority evidence")
        report_scene = _mapping(report_scenes[scene], f"{scene} report")
        _, cache_encoded = _read_file_record(evidence["semantic_score_cache"], label=f"{scene} semantic score cache")
        _, receipt_encoded = _read_file_record(evidence["semantic_score_cache_receipt"], label=f"{scene} semantic score receipt")
        _, pseudo_encoded = _read_file_record(evidence["pseudo_gt"], label=f"{scene} pseudo-GT")
        _, prediction_encoded = _read_file_record(evidence["prediction"], label=f"{scene} prediction")
        receipt = _load_bound_receipt(receipt_encoded, scene=scene, scene_evidence=evidence)
        cache = _validate_cache_and_receipt(
            scene=scene,
            scene_evidence=evidence,
            cache_encoded=cache_encoded,
            receipt=receipt,
            source_hash_cache=source_hash_cache,
        )

        pseudo = _load_npz(
            pseudo_encoded,
            label=f"{scene} pseudo-GT",
            expected_keys={"pseudo_labels", "settings_json", "stats_json"},
        )
        prediction = _load_npz(
            prediction_encoded,
            label=f"{scene} prediction",
            expected_keys={
                "xyz",
                "pseudo_labels",
                "significance",
                "pred_split_19",
                "pred_split_15",
                "pred_split_10",
            },
        )
        count = int(evidence["num_gaussians"])
        _require(pseudo["pseudo_labels"].dtype == np.int32 and pseudo["pseudo_labels"].shape == (count,), f"{scene} pseudo-GT row domain differs")
        _exact_array(prediction["pseudo_labels"], pseudo["pseudo_labels"], message=f"{scene} prediction/pseudo-GT rows differ")
        _exact_array(prediction["xyz"], cache["xyz"].numpy(), message=f"{scene} prediction/cache xyz rows differ")
        expected_significance = cache["opacity"].numpy() * cache["scale"].numpy().prod(axis=1)
        _exact_array(prediction["significance"], expected_significance, message=f"{scene} prediction significance differs")
        expected_predictions = evaluator.predict_frozen_splits(cache["semantic_scores"])
        for split in SPLITS:
            _exact_array(
                prediction[f"pred_split_{split}"],
                expected_predictions[split],
                message=f"{scene} split{split} prediction is not exact score-bank argmax",
            )
        try:
            pseudo_settings = json.loads(str(pseudo["settings_json"].item()))
            pseudo_stats = json.loads(str(pseudo["stats_json"].item()))
        except (ValueError, json.JSONDecodeError) as error:
            raise ResultAuthorityError(f"{scene} pseudo-GT metadata is invalid") from error
        _require(
            set(pseudo_settings)
            == {
                "radius_factor",
                "candidate_k",
                "fallback_k",
                "class_balance",
                "geometry_sha256",
                "label_cloud_sha256",
            }
            and pseudo_settings["radius_factor"] == 5.0
            and pseudo_settings["candidate_k"] == 1000
            and pseudo_settings["fallback_k"] == 1
            and pseudo_settings["class_balance"] is True,
            f"{scene} pseudo-GT settings differ",
        )
        _require(report_scene.get("pseudo_gt") == pseudo_stats, f"{scene} pseudo-GT stats/report differ")

        replayed_splits: dict[str, Any] = {}
        for split in SPLITS:
            replayed_splits[split] = volume_weighted_split_metrics(
                pseudo["pseudo_labels"],
                prediction[f"pred_split_{split}"],
                prediction["significance"],
                OPENGAUSSIAN_NYU40_CLASS_SPLITS[split],
            )
        _require(report_scene.get("splits") == replayed_splits, f"{scene} exact metric replay differs")
        _require(
            report_scene.get("num_gaussians") == count
            and report_scene.get("region_observed_count") == evidence["region_observed_count"]
            and report_scene.get("no_evidence_fallback_count") == evidence["no_evidence_fallback_count"],
            f"{scene} report row counts differ",
        )
        _require(
            report_scene.get("semantic_score_cache") == evidence["semantic_score_cache"]["path"]
            and report_scene.get("semantic_score_cache_sha256") == evidence["semantic_score_cache"]["sha256"]
            and report_scene.get("pseudo_gt_cache") == evidence["pseudo_gt"]["path"]
            and report_scene.get("pseudo_gt_cache_sha256") == evidence["pseudo_gt"]["sha256"]
            and report_scene.get("prediction_npz") == evidence["prediction"]["path"],
            f"{scene} report/final evidence binding differs",
        )
        _require(
            report_scene.get("geometry_checkpoint") == receipt["geometry_checkpoint"]["path"]
            and report_scene.get("geometry_checkpoint_sha256") == receipt["geometry_checkpoint"]["sha256"]
            and report_scene.get("semantic_source") == receipt["canonical_field_source"]["path"]
            and report_scene.get("semantic_source_sha256") == receipt["canonical_field_source"]["sha256"]
            and report_scene.get("query_source") == receipt["query_source"]["path"]
            and report_scene.get("query_source_sha256") == receipt["query_source"]["sha256"],
            f"{scene} report/receipt transitive source binding differs",
        )
        replayed_scenes[scene] = {"splits": replayed_splits}
        total_rows += count
        total_observed += int(evidence["region_observed_count"])
        total_fallback += int(evidence["no_evidence_fallback_count"])
        del cache

    replayed_macro = evaluator._scene_macro(replayed_scenes)
    _require(report.get("macro") == replayed_macro, "evaluator macro does not replay exactly")
    _require(payload.get("metrics") == replayed_macro, "frozen authority metrics do not replay exactly")
    return {
        "status": "validated",
        "contract": CONTRACT,
        "authority": str(authority_path),
        "authority_sha256": _sha256_bytes(encoded),
        "totality_freeze_sha256": V2_EXPECTED_FREEZE_SHA256,
        "evaluator_report_sha256": payload["evaluator_report"]["sha256"],
        "scene_count": len(EXPECTED_SCENES),
        "total_gaussian_rows": total_rows,
        "region_observed_rows": total_observed,
        "no_evidence_fallback_rows": total_fallback,
        "macro": replayed_macro,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    args = parser.parse_args(argv)
    print(json.dumps(validate_result_authority(args.authority), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
