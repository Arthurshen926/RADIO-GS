"""Strict element-domain evaluator boundary for a future formal LERF3D task.

This module intentionally does not read LERF polygon annotations or render a
3-D prediction back into an image.  A target authority is accepted only when
it binds one binary one-dimensional target to the persistent carrier elements
for every query in an external frozen cohort manifest.

The public benchmark release currently available to this repository does not
provide that authority.  Consequently the production cohort manifest is
classified as blocked.  The typed core remains useful for contract tests and
can score only after an independently validated official element-target
authority is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.object_memory import ElementQueryPosterior, SparseObjectAssignments
from radio_gs.v4.query import QueryPacket, QuerySelectionMode


COHORT_SCHEMA = "radio_gs.evaluation.lerf3d_element_cohort_manifest.v1"
TARGET_SCHEMA = "radio_gs.evaluation.lerf3d_element_target_authority.v1"
PREDICTION_SCHEMA = "radio_gs.evaluation.lerf3d_element_prediction_receipt.v1"
REPORT_SCHEMA = "radio_gs.evaluation.lerf3d_element_metric_report.v1"

READY_STATUS = "validated_official_element_target_contract"
BLOCKED_STATUS = "blocked_missing_official_element_target_contract"
TARGET_AUTHORITY_STATUS = "validated_official_element_targets"
TARGET_DOMAIN = "persistent_carrier_elements"
QUERY_UNIT_SEMANTICS = "official_3d_object_target"
AGGREGATION = "per_query_then_unweighted_scene_macro"
POSTERIOR_API = "SparseObjectAssignments.element_posterior"


class Lerf3DElementContractError(ValueError):
    """Raised instead of silently falling back to a rendered 2-D proxy."""


@dataclass(frozen=True, order=True)
class ElementQueryKey:
    scene_id: str
    query_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise Lerf3DElementContractError("scene_id must be a non-empty string")
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise Lerf3DElementContractError("query_id must be a non-empty string")

    @property
    def identifier(self) -> str:
        return f"{self.scene_id}:{self.query_id}"


@dataclass(frozen=True)
class CohortScene:
    scene_id: str
    expected_query_count: int
    query_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise Lerf3DElementContractError("cohort scene_id must be non-empty")
        if (
            isinstance(self.expected_query_count, bool)
            or not isinstance(self.expected_query_count, int)
            or self.expected_query_count <= 0
        ):
            raise Lerf3DElementContractError("expected_query_count must be positive")
        values = tuple(self.query_ids)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise Lerf3DElementContractError("query ids must be non-empty strings")
        if len(values) != len(set(values)) or values != tuple(sorted(values)):
            raise Lerf3DElementContractError("query ids must be unique and sorted")
        object.__setattr__(self, "query_ids", values)


@dataclass(frozen=True)
class ElementCohortManifest:
    path: Path
    sha256: str
    protocol_id: str
    status: str
    scenes: tuple[CohortScene, ...]
    expected_total_query_count: int
    target_domain: str
    query_unit_semantics: str
    iou_thresholds: tuple[float, float]
    aggregation: str
    target_authority_schema: str
    blocker: str | None = None

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes)

    @property
    def query_keys(self) -> tuple[ElementQueryKey, ...]:
        return tuple(
            ElementQueryKey(scene.scene_id, query_id)
            for scene in self.scenes
            for query_id in scene.query_ids
        )

    @property
    def ready(self) -> bool:
        return self.status == READY_STATUS

    def require_ready(self) -> None:
        if not self.ready:
            detail = self.blocker or "official element target contract is unavailable"
            raise Lerf3DElementContractError(
                f"formal LERF3D scoring is blocked: {detail}"
            )


@dataclass(frozen=True)
class CarrierBinding:
    sha256: str
    num_elements: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.sha256):
            raise Lerf3DElementContractError("carrier binding requires a SHA256 digest")
        if (
            isinstance(self.num_elements, bool)
            or not isinstance(self.num_elements, int)
            or self.num_elements <= 0
        ):
            raise Lerf3DElementContractError("carrier num_elements must be positive")


@dataclass(frozen=True)
class ElementTargetAuthority:
    cohort_manifest_sha256: str
    masks: Mapping[ElementQueryKey, np.ndarray]
    carrier_bindings: Mapping[str, CarrierBinding]
    association_authority: Mapping[str, Any]
    status: str = TARGET_AUTHORITY_STATUS


@dataclass(frozen=True)
class PredictionInformationPolicy:
    target_rgb_opened: bool = False
    target_membership_opened: bool = False
    target_association_opened: bool = False
    target_metrics_opened: bool = False

    def validate(self) -> None:
        values = (
            self.target_rgb_opened,
            self.target_membership_opened,
            self.target_association_opened,
            self.target_metrics_opened,
        )
        if any(not isinstance(value, bool) for value in values):
            raise Lerf3DElementContractError("information-policy fields must be booleans")
        if any(values):
            raise Lerf3DElementContractError(
                "prediction construction opened forbidden target information"
            )


@dataclass(frozen=True)
class ElementPredictionBatch:
    cohort_manifest_sha256: str
    masks: Mapping[ElementQueryKey, np.ndarray]
    carrier_bindings: Mapping[str, CarrierBinding]
    method_id: str
    selection_threshold: float
    query_selection_mode: QuerySelectionMode | str
    information_policy: PredictionInformationPolicy = field(
        default_factory=PredictionInformationPolicy
    )
    source_records: tuple[Mapping[str, str], ...] = ()
    posterior_api: str = POSTERIOR_API


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Lerf3DElementContractError(f"cannot read JSON contract: {path}") from error
    if not isinstance(payload, dict):
        raise Lerf3DElementContractError("contract root must be a JSON object")
    return payload


def _resolve_bound_path(receipt: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise Lerf3DElementContractError("bound archive path must be non-empty")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = receipt.parent / candidate
    return candidate.resolve(strict=True)


def _binary_element_mask(value: Any, *, expected: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (expected,):
        raise Lerf3DElementContractError(
            f"{label} must be a one-dimensional carrier-element mask [{expected}]"
        )
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.integer):
        raise Lerf3DElementContractError(f"{label} must be boolean or binary integer")
    if not bool(np.logical_or(array == 0, array == 1).all()):
        raise Lerf3DElementContractError(f"{label} contains values outside {{0,1}}")
    return np.ascontiguousarray(array.astype(bool))


def _mask_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask.astype(bool))
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_cohort_manifest(path: str | Path) -> ElementCohortManifest:
    source = Path(path).resolve(strict=True)
    payload = _read_json(source)
    if payload.get("schema") != COHORT_SCHEMA:
        raise Lerf3DElementContractError("element cohort manifest schema differs")
    status = payload.get("status")
    if status not in {READY_STATUS, BLOCKED_STATUS}:
        raise Lerf3DElementContractError("cohort manifest status is unsupported")
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise Lerf3DElementContractError("protocol_id must be non-empty")
    if payload.get("target_domain") != TARGET_DOMAIN:
        raise Lerf3DElementContractError("target domain is not persistent carrier elements")
    query_semantics = payload.get("query_unit_semantics")
    if not isinstance(query_semantics, str) or not query_semantics:
        raise Lerf3DElementContractError("query-unit semantics must be explicit")
    if payload.get("aggregation") != AGGREGATION:
        raise Lerf3DElementContractError("aggregation contract differs")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("iou") != "binary_element_iou":
        raise Lerf3DElementContractError("element IoU metric contract differs")
    raw_thresholds = metrics.get("strict_accuracy_thresholds")
    if not isinstance(raw_thresholds, list) or len(raw_thresholds) != 2:
        raise Lerf3DElementContractError("exactly two strict IoU thresholds are required")
    thresholds = tuple(float(value) for value in raw_thresholds)
    if (
        not all(math.isfinite(value) for value in thresholds)
        or thresholds[0] <= 0
        or thresholds[0] >= thresholds[1]
        or thresholds[1] >= 1
    ):
        raise Lerf3DElementContractError("strict IoU thresholds are invalid")
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise Lerf3DElementContractError("cohort must contain scenes")
    scenes: list[CohortScene] = []
    for row in raw_scenes:
        if not isinstance(row, dict):
            raise Lerf3DElementContractError("cohort scene entries must be objects")
        raw_query_ids = row.get("query_ids", [])
        if not isinstance(raw_query_ids, list):
            raise Lerf3DElementContractError("query_ids must be a list")
        scenes.append(
            CohortScene(
                scene_id=row.get("scene_id"),
                expected_query_count=row.get("expected_query_count"),
                query_ids=tuple(raw_query_ids),
            )
        )
    scene_ids = tuple(scene.scene_id for scene in scenes)
    if len(scene_ids) != len(set(scene_ids)):
        raise Lerf3DElementContractError("cohort scene ids are duplicated")
    expected_total = payload.get("expected_total_query_count")
    if (
        isinstance(expected_total, bool)
        or not isinstance(expected_total, int)
        or expected_total <= 0
        or expected_total != sum(scene.expected_query_count for scene in scenes)
    ):
        raise Lerf3DElementContractError("cohort total query count differs")
    if status == READY_STATUS:
        if query_semantics != QUERY_UNIT_SEMANTICS:
            raise Lerf3DElementContractError("ready cohort lacks official 3-D query semantics")
        for scene in scenes:
            if len(scene.query_ids) != scene.expected_query_count:
                raise Lerf3DElementContractError(
                    f"ready cohort query inventory is incomplete for {scene.scene_id}"
                )
    target_schema = payload.get("target_authority_schema")
    if target_schema != TARGET_SCHEMA:
        raise Lerf3DElementContractError("target authority schema binding differs")
    blocker = payload.get("blocker")
    if status == BLOCKED_STATUS and (not isinstance(blocker, str) or not blocker):
        raise Lerf3DElementContractError("blocked cohort must state its blocker")
    return ElementCohortManifest(
        path=source,
        sha256=sha256_file(source),
        protocol_id=protocol_id,
        status=status,
        scenes=tuple(scenes),
        expected_total_query_count=expected_total,
        target_domain=TARGET_DOMAIN,
        query_unit_semantics=query_semantics,
        iou_thresholds=(thresholds[0], thresholds[1]),
        aggregation=AGGREGATION,
        target_authority_schema=TARGET_SCHEMA,
        blocker=blocker if isinstance(blocker, str) else None,
    )


def compose_element_posterior(
    assignments: SparseObjectAssignments,
    query: QueryPacket,
    token_probability: torch.Tensor,
    *,
    null_probability: float | torch.Tensor | None = None,
) -> ElementQueryPosterior:
    """Use the sole v4 token-to-element posterior implementation."""

    if not isinstance(assignments, SparseObjectAssignments):
        raise TypeError("assignments must be SparseObjectAssignments")
    if not isinstance(query, QueryPacket):
        raise TypeError("query must be a v4 QueryPacket")
    return assignments.element_posterior(
        query,
        token_probability,
        null_probability=null_probability,
    )


def select_element_mask(
    assignments: SparseObjectAssignments,
    query: QueryPacket,
    token_probability: torch.Tensor,
    *,
    threshold: float,
    null_probability: float | torch.Tensor | None = None,
) -> np.ndarray:
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) < 1:
        raise Lerf3DElementContractError("selection threshold must lie in [0,1)")
    posterior = compose_element_posterior(
        assignments,
        query,
        token_probability,
        null_probability=null_probability,
    )
    return (posterior.foreground > float(threshold)).cpu().numpy()


def _carrier_bindings(
    value: Mapping[str, CarrierBinding | Mapping[str, Any]],
    scenes: Sequence[str],
) -> dict[str, CarrierBinding]:
    if set(value) != set(scenes):
        raise Lerf3DElementContractError("carrier-binding scene inventory differs")
    result: dict[str, CarrierBinding] = {}
    for scene_id in scenes:
        row = value[scene_id]
        if isinstance(row, CarrierBinding):
            binding = row
        elif isinstance(row, Mapping):
            binding = CarrierBinding(
                sha256=row.get("sha256"), num_elements=row.get("num_elements")
            )
        else:
            raise Lerf3DElementContractError("carrier binding must be an object")
        result[scene_id] = binding
    return result


def validate_prediction_batch(
    batch: ElementPredictionBatch,
    manifest: ElementCohortManifest,
) -> ElementPredictionBatch:
    manifest.require_ready()
    if batch.cohort_manifest_sha256 != manifest.sha256:
        raise Lerf3DElementContractError("prediction cohort-manifest digest differs")
    if not isinstance(batch.method_id, str) or not batch.method_id.strip():
        raise Lerf3DElementContractError("prediction method_id must be non-empty")
    if batch.posterior_api != POSTERIOR_API:
        raise Lerf3DElementContractError("prediction used a non-canonical posterior API")
    query_mode = QueryPacket(batch.query_selection_mode).selection_mode
    if query_mode is QuerySelectionMode.LOCAL_SEMANTIC:
        raise Lerf3DElementContractError(
            "object element predictions cannot claim local-semantic query mode"
        )
    threshold = float(batch.selection_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise Lerf3DElementContractError("prediction threshold must lie in [0,1)")
    batch.information_policy.validate()
    bindings = _carrier_bindings(batch.carrier_bindings, manifest.scene_ids)
    expected_keys = set(manifest.query_keys)
    if set(batch.masks) != expected_keys:
        missing = sorted(key.identifier for key in expected_keys - set(batch.masks))
        extra = sorted(key.identifier for key in set(batch.masks) - expected_keys)
        raise Lerf3DElementContractError(
            f"prediction query inventory differs; missing={missing[:3]}, extra={extra[:3]}"
        )
    masks = {
        key: _binary_element_mask(
            value,
            expected=bindings[key.scene_id].num_elements,
            label=f"prediction {key.identifier}",
        )
        for key, value in batch.masks.items()
    }
    return ElementPredictionBatch(
        cohort_manifest_sha256=batch.cohort_manifest_sha256,
        masks=masks,
        carrier_bindings=bindings,
        method_id=batch.method_id,
        selection_threshold=threshold,
        query_selection_mode=query_mode,
        information_policy=batch.information_policy,
        source_records=tuple(batch.source_records),
        posterior_api=POSTERIOR_API,
    )


def _validated_association_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Lerf3DElementContractError("target association authority is missing")
    required = {"authority_id", "release", "source_sha256", "official_element_domain"}
    if not required.issubset(value):
        raise Lerf3DElementContractError("target association authority is incomplete")
    if value.get("official_element_domain") is not True:
        raise Lerf3DElementContractError("target authority is not official element-domain data")
    if not all(isinstance(value.get(key), str) and value[key] for key in ("authority_id", "release")):
        raise Lerf3DElementContractError("target association provenance is incomplete")
    if not _is_sha256(value.get("source_sha256")):
        raise Lerf3DElementContractError("target association source digest is invalid")
    return dict(value)


def load_target_authority(
    receipt_path: str | Path,
    manifest: ElementCohortManifest,
) -> ElementTargetAuthority:
    """Load only 1-D carrier-element targets; no raster adapter is accepted."""

    manifest.require_ready()
    receipt = Path(receipt_path).resolve(strict=True)
    payload = _read_json(receipt)
    if payload.get("schema") != TARGET_SCHEMA:
        raise Lerf3DElementContractError("element target authority schema differs")
    if payload.get("status") != TARGET_AUTHORITY_STATUS:
        raise Lerf3DElementContractError("element target authority is not validated")
    if payload.get("target_domain") != TARGET_DOMAIN:
        raise Lerf3DElementContractError("target authority is not element-domain")
    if payload.get("cohort_manifest_sha256") != manifest.sha256:
        raise Lerf3DElementContractError("target cohort-manifest digest differs")
    forbidden_proxy_fields = {
        "polygon_annotations",
        "raster_masks",
        "rendered_predictions",
        "image_shapes",
        "projection_adapter",
    }
    if forbidden_proxy_fields.intersection(payload):
        raise Lerf3DElementContractError("2-D polygon/raster proxy fields are forbidden")
    association = _validated_association_authority(payload.get("association_authority"))
    archive_record = payload.get("archive")
    if not isinstance(archive_record, dict):
        raise Lerf3DElementContractError("target archive binding is missing")
    archive = _resolve_bound_path(receipt, archive_record.get("path"))
    if archive_record.get("sha256") != sha256_file(archive):
        raise Lerf3DElementContractError("target archive SHA256 differs")
    raw_bindings = payload.get("carrier_bindings")
    if not isinstance(raw_bindings, dict):
        raise Lerf3DElementContractError("target carrier bindings are missing")
    bindings = _carrier_bindings(raw_bindings, manifest.scene_ids)
    records = payload.get("targets")
    if not isinstance(records, list):
        raise Lerf3DElementContractError("target records must be a list")
    masks: dict[ElementQueryKey, np.ndarray] = {}
    with np.load(archive, allow_pickle=False) as arrays:
        expected_arrays: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise Lerf3DElementContractError("target records must be objects")
            key = ElementQueryKey(record.get("scene_id"), record.get("query_id"))
            if key in masks:
                raise Lerf3DElementContractError(f"duplicate target: {key.identifier}")
            array_key = record.get("array_key")
            if not isinstance(array_key, str) or not array_key:
                raise Lerf3DElementContractError("target array_key must be non-empty")
            expected_arrays.add(array_key)
            if array_key not in arrays:
                raise Lerf3DElementContractError(f"target array is absent: {array_key}")
            if record.get("shape") != [bindings[key.scene_id].num_elements]:
                raise Lerf3DElementContractError(
                    f"target record is not one-dimensional: {key.identifier}"
                )
            mask = _binary_element_mask(
                arrays[array_key],
                expected=bindings[key.scene_id].num_elements,
                label=f"target {key.identifier}",
            )
            if not bool(mask.any()):
                raise Lerf3DElementContractError(f"target is empty: {key.identifier}")
            if record.get("mask_sha256") != _mask_sha256(mask):
                raise Lerf3DElementContractError(f"target mask digest differs: {key.identifier}")
            masks[key] = mask
        if set(arrays.files) != expected_arrays:
            raise Lerf3DElementContractError("target archive array inventory differs")
    if set(masks) != set(manifest.query_keys):
        raise Lerf3DElementContractError("target query inventory differs from cohort")
    return ElementTargetAuthority(
        cohort_manifest_sha256=manifest.sha256,
        masks=masks,
        carrier_bindings=bindings,
        association_authority=association,
    )


def _element_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    if union <= 0:
        raise Lerf3DElementContractError("element target/prediction union is empty")
    return float(intersection / union)


def _summarize(values: Sequence[float], thresholds: tuple[float, float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.isfinite(array).all()):
        raise Lerf3DElementContractError("metric summary requires finite IoUs")
    return {
        "miou": float(array.mean()),
        "acc025": float((array > thresholds[0]).mean()),
        "acc050": float((array > thresholds[1]).mean()),
        "query_count": int(array.size),
    }


def score_element_predictions(
    batch: ElementPredictionBatch,
    targets: ElementTargetAuthority,
    manifest: ElementCohortManifest,
) -> dict[str, Any]:
    validated = validate_prediction_batch(batch, manifest)
    if targets.status != TARGET_AUTHORITY_STATUS:
        raise Lerf3DElementContractError("target authority status differs")
    if targets.cohort_manifest_sha256 != manifest.sha256:
        raise Lerf3DElementContractError("target cohort-manifest digest differs")
    bindings = _carrier_bindings(targets.carrier_bindings, manifest.scene_ids)
    if bindings != dict(validated.carrier_bindings):
        raise Lerf3DElementContractError("prediction and target carrier bindings differ")
    if set(targets.masks) != set(manifest.query_keys):
        raise Lerf3DElementContractError("target query inventory differs")
    _validated_association_authority(dict(targets.association_authority))
    per_scene_values: dict[str, list[float]] = {
        scene_id: [] for scene_id in manifest.scene_ids
    }
    per_query: list[dict[str, Any]] = []
    for key in manifest.query_keys:
        expected = bindings[key.scene_id].num_elements
        target = _binary_element_mask(
            targets.masks[key], expected=expected, label=f"target {key.identifier}"
        )
        if not bool(target.any()):
            raise Lerf3DElementContractError(f"target is empty: {key.identifier}")
        prediction = _binary_element_mask(
            validated.masks[key], expected=expected, label=f"prediction {key.identifier}"
        )
        iou = _element_iou(prediction, target)
        per_scene_values[key.scene_id].append(iou)
        per_query.append(
            {
                "scene_id": key.scene_id,
                "query_id": key.query_id,
                "iou": iou,
                "acc025": bool(iou > manifest.iou_thresholds[0]),
                "acc050": bool(iou > manifest.iou_thresholds[1]),
            }
        )
    per_scene = {
        scene_id: _summarize(per_scene_values[scene_id], manifest.iou_thresholds)
        for scene_id in manifest.scene_ids
    }
    scene_macro = {
        name: float(np.mean([per_scene[scene_id][name] for scene_id in manifest.scene_ids]))
        for name in ("miou", "acc025", "acc050")
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete_validated_element_domain_cohort",
        "protocol_id": manifest.protocol_id,
        "cohort_manifest_sha256": manifest.sha256,
        "target_domain": TARGET_DOMAIN,
        "uses_2d_polygon_proxy": False,
        "formal_lerf3d_eligible": True,
        "posterior_api": POSTERIOR_API,
        "query_selection_mode": validated.query_selection_mode.value,
        "selection_threshold": validated.selection_threshold,
        "strict_iou_thresholds": list(manifest.iou_thresholds),
        "aggregation": AGGREGATION,
        "per_scene": per_scene,
        "scene_equal_macro": scene_macro,
        "per_query": per_query,
    }


def seal_prediction_batch(
    batch: ElementPredictionBatch,
    manifest: ElementCohortManifest,
    *,
    archive_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    validated = validate_prediction_batch(batch, manifest)
    archive = Path(archive_path).resolve()
    receipt = Path(receipt_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for index, key in enumerate(manifest.query_keys):
        array_key = f"prediction_{index:06d}"
        mask = validated.masks[key]
        arrays[array_key] = mask.astype(np.uint8)
        records.append(
            {
                "scene_id": key.scene_id,
                "query_id": key.query_id,
                "array_key": array_key,
                "shape": list(mask.shape),
                "mask_sha256": _mask_sha256(mask),
            }
        )
    with archive.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    payload = {
        "schema": PREDICTION_SCHEMA,
        "sealed_before_target_access": True,
        "cohort_manifest_sha256": manifest.sha256,
        "archive": {"path": str(archive), "sha256": sha256_file(archive)},
        "method_id": validated.method_id,
        "posterior_api": validated.posterior_api,
        "selection_threshold": validated.selection_threshold,
        "query_selection_mode": validated.query_selection_mode.value,
        "information_policy": vars(validated.information_policy),
        "carrier_bindings": {
            scene_id: vars(binding)
            for scene_id, binding in validated.carrier_bindings.items()
        },
        "source_records": list(validated.source_records),
        "predictions": records,
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def load_prediction_batch(
    receipt_path: str | Path,
    manifest: ElementCohortManifest,
) -> ElementPredictionBatch:
    manifest.require_ready()
    receipt = Path(receipt_path).resolve(strict=True)
    payload = _read_json(receipt)
    if payload.get("schema") != PREDICTION_SCHEMA:
        raise Lerf3DElementContractError("element prediction receipt schema differs")
    if payload.get("sealed_before_target_access") is not True:
        raise Lerf3DElementContractError("predictions were not sealed before target access")
    if payload.get("cohort_manifest_sha256") != manifest.sha256:
        raise Lerf3DElementContractError("prediction cohort-manifest digest differs")
    archive_record = payload.get("archive")
    if not isinstance(archive_record, dict):
        raise Lerf3DElementContractError("prediction archive binding is missing")
    archive = _resolve_bound_path(receipt, archive_record.get("path"))
    if archive_record.get("sha256") != sha256_file(archive):
        raise Lerf3DElementContractError("prediction archive SHA256 differs")
    raw_bindings = payload.get("carrier_bindings")
    if not isinstance(raw_bindings, dict):
        raise Lerf3DElementContractError("prediction carrier bindings are missing")
    bindings = _carrier_bindings(raw_bindings, manifest.scene_ids)
    records = payload.get("predictions")
    if not isinstance(records, list):
        raise Lerf3DElementContractError("prediction records must be a list")
    masks: dict[ElementQueryKey, np.ndarray] = {}
    with np.load(archive, allow_pickle=False) as arrays:
        expected_arrays: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise Lerf3DElementContractError("prediction records must be objects")
            key = ElementQueryKey(record.get("scene_id"), record.get("query_id"))
            if key in masks:
                raise Lerf3DElementContractError(f"duplicate prediction: {key.identifier}")
            array_key = record.get("array_key")
            if not isinstance(array_key, str) or array_key not in arrays:
                raise Lerf3DElementContractError("prediction array is absent")
            expected_arrays.add(array_key)
            mask = _binary_element_mask(
                arrays[array_key],
                expected=bindings[key.scene_id].num_elements,
                label=f"prediction {key.identifier}",
            )
            if list(mask.shape) != record.get("shape"):
                raise Lerf3DElementContractError("prediction record shape differs")
            if record.get("mask_sha256") != _mask_sha256(mask):
                raise Lerf3DElementContractError("prediction mask digest differs")
            masks[key] = mask
        if set(arrays.files) != expected_arrays:
            raise Lerf3DElementContractError("prediction archive array inventory differs")
    policy_payload = payload.get("information_policy")
    if not isinstance(policy_payload, dict):
        raise Lerf3DElementContractError("prediction information policy is missing")
    try:
        policy = PredictionInformationPolicy(**policy_payload)
    except TypeError as error:
        raise Lerf3DElementContractError("prediction information policy differs") from error
    return validate_prediction_batch(
        ElementPredictionBatch(
            cohort_manifest_sha256=manifest.sha256,
            masks=masks,
            carrier_bindings=bindings,
            method_id=payload.get("method_id"),
            selection_threshold=payload.get("selection_threshold"),
            query_selection_mode=payload.get("query_selection_mode"),
            information_policy=policy,
            source_records=tuple(payload.get("source_records", [])),
            posterior_api=payload.get("posterior_api"),
        ),
        manifest,
    )


def _score_command(args: argparse.Namespace) -> None:
    manifest = load_cohort_manifest(args.cohort_manifest)
    predictions = load_prediction_batch(args.prediction_receipt, manifest)
    targets = load_target_authority(args.target_receipt, manifest)
    report = score_element_predictions(predictions, targets, manifest)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": report["status"], "output": str(destination)}, indent=2))


def _audit_manifest_command(args: argparse.Namespace) -> None:
    manifest = load_cohort_manifest(args.cohort_manifest)
    print(
        json.dumps(
            {
                "protocol_id": manifest.protocol_id,
                "status": manifest.status,
                "scene_count": len(manifest.scenes),
                "expected_total_query_count": manifest.expected_total_query_count,
                "formal_scoring_ready": manifest.ready,
                "blocker": manifest.blocker,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-manifest")
    audit.add_argument("--cohort-manifest", required=True)
    audit.set_defaults(handler=_audit_manifest_command)
    score = subparsers.add_parser("score")
    score.add_argument("--cohort-manifest", required=True)
    score.add_argument("--prediction-receipt", required=True)
    score.add_argument("--target-receipt", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(handler=_score_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
