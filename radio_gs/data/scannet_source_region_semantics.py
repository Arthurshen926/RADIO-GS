"""Official ScanNet source semantics aligned to canonical SurfaceRegion rows.

The adapter is deliberately narrower than a benchmark loader.  It accepts an
official ScanNet *training* scene and maps its mesh annotation to the already
frozen SurfaceRegion membership authority.  The mapping never consumes the
AGILE3D instance-id PLY or a model prediction:

``official mesh vertex -> official segment -> aggregation raw label -> NYU40``

Every canonical region retains a soft 41-way NYU40 distribution.  Mixed
regions therefore remain mixed instead of being silently converted to an
argmax pseudo-label.  Geometry coverage, semantic coverage, and purity are
separate channels so downstream likelihood training can abstain explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree


SCANNET_SOURCE_REGION_SEMANTIC_SCHEMA = (
    "radio_gs.scannet_source_region_semantic_sidecar.v1"
)
SCANNET_DEVELOPMENT_REGION_SEMANTIC_SCHEMA = (
    "radio_gs.scannet_development_region_semantic_sidecar.v1"
)
PREREGISTERED_SOURCE_FIT_SCENES = (
    "scene0001_00",
    "scene0002_00",
    "scene0005_00",
)
PREREGISTERED_DEVELOPMENT_SCENES = ("scene0003_00",)
NYU40_CLASS_COUNT_WITH_UNANNOTATED = 41
MESH_NEAREST_MAX_DISTANCE_METERS = 0.05
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    if tensor.ndim == 0:
        digest.update(tensor.numpy().tobytes(order="C"))
    else:
        for start in range(0, int(tensor.shape[0]), 4096):
            digest.update(tensor[start : start + 4096].numpy().tobytes(order="C"))
    return digest.hexdigest()


def _record(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve(strict=True)
    return {"path": str(source), "sha256": sha256_file(source)}


def source_region_semantic_contract() -> dict[str, Any]:
    return {
        "schema": SCANNET_SOURCE_REGION_SEMANTIC_SCHEMA,
        "schema_version": 1,
        "partition": "preregistered_scannet_source_fit_only",
        "allowed_source_fit_scenes": list(PREREGISTERED_SOURCE_FIT_SCENES),
        "held_out_development_scenes": list(PREREGISTERED_DEVELOPMENT_SCENES),
        "semantic_authority": (
            "official_mesh_vertex_to_official_segmentation_to_official_aggregation_"
            "raw_category_to_scannetv2_tsv_nyu40id"
        ),
        "geometry_alignment": {
            "method": "nearest_official_mesh_vertex_euclidean",
            "maximum_distance_meters": MESH_NEAREST_MAX_DISTANCE_METERS,
            "threshold_scope": "single_global_fixed_not_scene_tuned",
        },
        "region_aggregation": {
            "membership": "accepted_v2_region_rows_and_token_mask",
            "weight": "uniform_valid_region_member",
            "output": "soft_distribution_over_nyu40_ids_0_through_40",
            "mixed_region_policy": "retain_soft_distribution_never_argmax",
        },
        "forbidden": [
            "agile3d_instance_id",
            "pseudo_semantic_label",
            "development_label_during_fit",
            "test_scene_or_label",
            "benchmark_metric_or_prediction",
        ],
    }


def development_region_semantic_contract() -> dict[str, Any]:
    contract = source_region_semantic_contract()
    return {
        **contract,
        "schema": SCANNET_DEVELOPMENT_REGION_SEMANTIC_SCHEMA,
        "partition": "single_heldout_scannet_development_open_after_source_gates",
        "allowed_source_fit_scenes": [],
        "held_out_development_scenes": list(PREREGISTERED_DEVELOPMENT_SCENES),
    }


def load_scannet_raw_to_nyu40(path: str | Path) -> dict[str, int]:
    source = Path(path).expanduser().resolve(strict=True)
    mapping: dict[str, int] = {}
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"raw_category", "nyu40id"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("ScanNet label TSV lacks raw_category/nyu40id")
        for row in reader:
            raw = str(row["raw_category"]).strip()
            if not raw:
                continue
            nyu40 = int(row["nyu40id"])
            if nyu40 < 0 or nyu40 >= NYU40_CLASS_COUNT_WITH_UNANNOTATED:
                raise ValueError("ScanNet TSV contains an out-of-range NYU40 id")
            previous = mapping.get(raw)
            if previous is not None and previous != nyu40:
                raise ValueError(f"ScanNet TSV maps {raw!r} inconsistently")
            mapping[raw] = nyu40
    if not mapping:
        raise ValueError("ScanNet label TSV is empty")
    return mapping


def official_vertex_nyu40_labels(
    *,
    scene_id: str,
    vertex_count: int,
    segmentation: Mapping[str, Any],
    aggregation: Mapping[str, Any],
    raw_to_nyu40: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Return official per-mesh-vertex NYU40 ids without instance-id shortcuts."""

    if str(segmentation.get("sceneId")) != scene_id:
        raise ValueError("ScanNet segmentation scene id differs")
    aggregation_scene = str(aggregation.get("sceneId", ""))
    if aggregation_scene not in {scene_id, f"scannet.{scene_id}"}:
        raise ValueError("ScanNet aggregation scene id differs")
    segment_ids = np.asarray(segmentation.get("segIndices"), dtype=np.int64)
    if segment_ids.shape != (int(vertex_count),):
        raise ValueError("ScanNet segIndices do not align with official mesh vertices")
    segment_to_class: dict[int, int] = {}
    groups = aggregation.get("segGroups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ValueError("ScanNet aggregation lacks segGroups")
    unknown_raw_labels: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("ScanNet segGroup must be a mapping")
        raw_label = str(group.get("label", "")).strip()
        if raw_label not in raw_to_nyu40:
            unknown_raw_labels.add(raw_label)
            nyu40 = 0
        else:
            nyu40 = int(raw_to_nyu40[raw_label])
        segments = group.get("segments")
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            raise ValueError("ScanNet segGroup lacks a segment list")
        for raw_segment in segments:
            segment = int(raw_segment)
            previous = segment_to_class.get(segment)
            if previous is not None and previous != nyu40:
                raise ValueError("one official ScanNet segment has conflicting semantics")
            segment_to_class[segment] = nyu40
    vertex_labels = np.fromiter(
        (segment_to_class.get(int(segment), 0) for segment in segment_ids),
        dtype=np.int64,
        count=segment_ids.size,
    )
    return vertex_labels, {
        "mesh_vertex_count": int(vertex_count),
        "segmentation_segment_count": int(np.unique(segment_ids).size),
        "aggregation_group_count": int(len(groups)),
        "aggregation_assigned_segment_count": int(len(segment_to_class)),
        "unknown_raw_label_count": int(len(unknown_raw_labels)),
        "annotated_mesh_vertex_count": int((vertex_labels > 0).sum()),
    }


def _build_region_semantic_sidecar(
    *,
    scene_id: str,
    accepted_region_payload: Mapping[str, Any],
    factorized_field_payload: Mapping[str, Any],
    official_mesh_xyz: np.ndarray,
    official_vertex_labels: np.ndarray,
    lineage_paths: Mapping[str, str | Path],
    official_label_audit: Mapping[str, int] | None = None,
    schema: str,
    contract: Mapping[str, Any],
    partition: str,
    allowed_scenes: Sequence[str],
    required_lineage: set[str],
    source_access: Mapping[str, bool],
) -> dict[str, Any]:
    """Map official source semantics onto canonical regions with soft labels."""

    if scene_id not in allowed_scenes:
        raise PermissionError("scene is not in the preregistered semantic cohort")
    if accepted_region_payload.get("scene_id") != scene_id:
        raise ValueError("accepted region authority scene differs")
    region_rows = torch.as_tensor(accepted_region_payload.get("region_rows")).long()
    token_mask = torch.as_tensor(accepted_region_payload.get("token_mask"))
    if (
        region_rows.ndim != 2
        or token_mask.shape != region_rows.shape
        or token_mask.dtype != torch.bool
        or region_rows.shape[0] <= 0
    ):
        raise ValueError("accepted region membership authority differs")
    canonical_region_indices = torch.as_tensor(
        accepted_region_payload.get("canonical_region_indices")
    ).long()
    region_fingerprints = accepted_region_payload.get("region_fingerprints")
    region_count = int(region_rows.shape[0])
    if canonical_region_indices.shape != (region_count,) or not isinstance(
        region_fingerprints, Sequence
    ) or len(region_fingerprints) != region_count:
        raise ValueError("accepted canonical region row authority differs")

    xyz = torch.as_tensor(factorized_field_payload.get("xyz")).detach().cpu().float()
    factorized = factorized_field_payload.get("factorized_radio")
    if not isinstance(factorized, Mapping):
        raise ValueError("factorized field lacks canonical factorized RADIO")
    field_valid = torch.as_tensor(factorized.get("valid")).detach().cpu()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or field_valid.shape != (xyz.shape[0],):
        raise ValueError("factorized field geometry/valid axes differ")
    if field_valid.dtype != torch.bool or not bool(torch.isfinite(xyz).all()):
        raise ValueError("factorized field geometry/valid authority differs")
    accepted_geometry = accepted_region_payload.get("geometry_fingerprint")
    field_geometry = factorized_field_payload.get("geometry_fingerprint")
    if accepted_geometry != field_geometry:
        raise ValueError("accepted region and factorized field geometry differ")
    if bool((region_rows[token_mask] < 0).any()) or bool(
        (region_rows[token_mask] >= xyz.shape[0]).any()
    ):
        raise ValueError("accepted region membership contains an invalid primitive row")

    mesh_xyz = np.asarray(official_mesh_xyz, dtype=np.float32)
    vertex_labels = np.asarray(official_vertex_labels, dtype=np.int64)
    if mesh_xyz.ndim != 2 or mesh_xyz.shape[1] != 3 or not np.isfinite(mesh_xyz).all():
        raise ValueError("official ScanNet mesh xyz must be finite [V,3]")
    if vertex_labels.shape != (mesh_xyz.shape[0],):
        raise ValueError("official ScanNet mesh labels do not align")
    if np.any((vertex_labels < 0) | (vertex_labels >= 41)):
        raise ValueError("official ScanNet mesh contains invalid NYU40 ids")

    membership_valid = token_mask & field_valid[region_rows.clamp(0, xyz.shape[0] - 1)]
    flat_region = (
        torch.arange(region_count)[:, None]
        .expand_as(region_rows)[membership_valid]
        .numpy()
    )
    flat_primitive = region_rows[membership_valid].numpy()
    if flat_primitive.size == 0:
        raise ValueError("accepted region membership has no valid field rows")
    unique_primitive, inverse = np.unique(flat_primitive, return_inverse=True)
    tree = cKDTree(mesh_xyz)
    nearest_distance_unique, nearest_vertex_unique = tree.query(
        xyz[torch.from_numpy(unique_primitive)].numpy(), k=1, workers=-1
    )
    nearest_distance = nearest_distance_unique[inverse].astype(np.float32, copy=False)
    nearest_vertex = nearest_vertex_unique[inverse]
    within = nearest_distance <= MESH_NEAREST_MAX_DISTANCE_METERS
    member_label = vertex_labels[nearest_vertex]
    annotated = within & (member_label > 0)

    member_count = np.bincount(flat_region, minlength=region_count).astype(np.int64)
    within_count = np.bincount(flat_region[within], minlength=region_count).astype(np.int64)
    annotated_count = np.bincount(
        flat_region[annotated], minlength=region_count
    ).astype(np.int64)
    counts = np.zeros((region_count, 41), dtype=np.float32)
    np.add.at(counts, (flat_region[annotated], member_label[annotated]), 1.0)
    distribution = np.divide(
        counts,
        annotated_count[:, None],
        out=np.zeros_like(counts),
        where=annotated_count[:, None] > 0,
    )
    geometry_coverage = np.divide(
        within_count,
        member_count,
        out=np.zeros(region_count, dtype=np.float32),
        where=member_count > 0,
    ).astype(np.float32)
    semantic_coverage = np.divide(
        annotated_count,
        member_count,
        out=np.zeros(region_count, dtype=np.float32),
        where=member_count > 0,
    ).astype(np.float32)
    purity = distribution.max(axis=1).astype(np.float32)
    valid = annotated_count > 0
    mean_distance = np.zeros(region_count, dtype=np.float32)
    p95_distance = np.zeros(region_count, dtype=np.float32)
    for region in range(region_count):
        values = nearest_distance[flat_region == region]
        if values.size:
            mean_distance[region] = float(values.mean())
            p95_distance[region] = float(np.quantile(values, 0.95))

    global_geometry_coverage = float(within.sum() / max(1, within.size))
    valid_region_fraction = float(valid.mean())
    if global_geometry_coverage < 0.25:
        raise ValueError("source mesh/field alignment has less than 25% fixed-radius coverage")
    if valid_region_fraction < 0.25:
        raise ValueError("source region semantic authority covers less than 25% of regions")

    tensors = {
        "canonical_region_indices": canonical_region_indices.cpu().contiguous(),
        "nyu40_class_distribution": torch.from_numpy(distribution).contiguous(),
        "valid": torch.from_numpy(valid).contiguous(),
        "geometry_coverage": torch.from_numpy(geometry_coverage).contiguous(),
        "semantic_coverage": torch.from_numpy(semantic_coverage).contiguous(),
        "semantic_purity": torch.from_numpy(purity).contiguous(),
        "region_member_count": torch.from_numpy(member_count).contiguous(),
        "within_distance_member_count": torch.from_numpy(within_count).contiguous(),
        "annotated_member_count": torch.from_numpy(annotated_count).contiguous(),
        "nearest_distance_mean_meters": torch.from_numpy(mean_distance).contiguous(),
        "nearest_distance_p95_meters": torch.from_numpy(p95_distance).contiguous(),
    }
    if set(lineage_paths) != required_lineage:
        raise ValueError("source region semantic lineage paths differ")
    lineage = {key: _record(lineage_paths[key]) for key in sorted(required_lineage)}
    return {
        "schema": schema,
        "schema_version": 1,
        "contract": dict(contract),
        "scene_id": scene_id,
        "physical_space_id": scene_id.rsplit("_", 1)[0],
        "partition": partition,
        "region_fingerprints": [str(value) for value in region_fingerprints],
        **tensors,
        "statistics": {
            **dict(official_label_audit or {}),
            "canonical_region_count": region_count,
            "valid_region_count": int(valid.sum()),
            "valid_region_fraction": valid_region_fraction,
            "global_geometry_coverage": global_geometry_coverage,
            "global_semantic_coverage": float(annotated.sum() / max(1, annotated.size)),
            "mixed_valid_region_count": int((valid & (purity < 1.0)).sum()),
            "mixed_valid_region_fraction": float(
                (valid & (purity < 1.0)).sum() / max(1, valid.sum())
            ),
            "valid_region_purity_mean": float(purity[valid].mean()),
            "valid_region_purity_p05": float(np.quantile(purity[valid], 0.05)),
            "valid_region_purity_median": float(np.quantile(purity[valid], 0.5)),
            "valid_region_purity_p95": float(np.quantile(purity[valid], 0.95)),
            "member_nearest_distance_median_meters": float(
                np.quantile(nearest_distance, 0.5)
            ),
            "member_nearest_distance_p95_meters": float(
                np.quantile(nearest_distance, 0.95)
            ),
        },
        "lineage": lineage,
        "source_access": dict(source_access),
        "channel_sha256": {key: _tensor_sha256(value) for key, value in tensors.items()},
    }


def build_source_region_semantic_sidecar(
    *,
    scene_id: str,
    accepted_region_payload: Mapping[str, Any],
    factorized_field_payload: Mapping[str, Any],
    official_mesh_xyz: np.ndarray,
    official_vertex_labels: np.ndarray,
    lineage_paths: Mapping[str, str | Path],
    official_label_audit: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return _build_region_semantic_sidecar(
        scene_id=scene_id,
        accepted_region_payload=accepted_region_payload,
        factorized_field_payload=factorized_field_payload,
        official_mesh_xyz=official_mesh_xyz,
        official_vertex_labels=official_vertex_labels,
        lineage_paths=lineage_paths,
        official_label_audit=official_label_audit,
        schema=SCANNET_SOURCE_REGION_SEMANTIC_SCHEMA,
        contract=source_region_semantic_contract(),
        partition="source_fit",
        allowed_scenes=PREREGISTERED_SOURCE_FIT_SCENES,
        required_lineage={
            "accepted_region_authority",
            "factorized_field_authority",
            "official_mesh",
            "official_segmentation",
            "official_aggregation",
            "official_label_tsv",
            "official_train_split",
        },
        source_access={
            "official_scannet_train_scene": True,
            "source_fit_semantic_labels_opened": True,
            "development_semantic_labels_opened": False,
            "test_semantic_labels_opened": False,
            "agile3d_instance_ids_opened": False,
            "pseudo_semantic_labels_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_threshold_tuning": False,
        },
    )


def build_development_region_semantic_sidecar(
    *,
    scene_id: str,
    accepted_region_payload: Mapping[str, Any],
    factorized_field_payload: Mapping[str, Any],
    official_mesh_xyz: np.ndarray,
    official_vertex_labels: np.ndarray,
    lineage_paths: Mapping[str, str | Path],
    official_label_audit: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return _build_region_semantic_sidecar(
        scene_id=scene_id,
        accepted_region_payload=accepted_region_payload,
        factorized_field_payload=factorized_field_payload,
        official_mesh_xyz=official_mesh_xyz,
        official_vertex_labels=official_vertex_labels,
        lineage_paths=lineage_paths,
        official_label_audit=official_label_audit,
        schema=SCANNET_DEVELOPMENT_REGION_SEMANTIC_SCHEMA,
        contract=development_region_semantic_contract(),
        partition="development",
        allowed_scenes=PREREGISTERED_DEVELOPMENT_SCENES,
        required_lineage={
            "accepted_region_authority",
            "factorized_field_authority",
            "official_mesh",
            "official_segmentation",
            "official_aggregation",
            "official_label_tsv",
            "official_train_split",
            "source_gate_receipt",
        },
        source_access={
            "official_scannet_train_scene": True,
            "source_fit_semantic_labels_opened": False,
            "development_semantic_labels_opened": True,
            "test_semantic_labels_opened": False,
            "agile3d_instance_ids_opened": False,
            "pseudo_semantic_labels_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_threshold_tuning": False,
            "parameter_callback_allowed": False,
        },
    )


def _validate_region_semantic_sidecar(
    value: object,
    *,
    schema: str,
    contract: Mapping[str, Any],
    partition: str,
    allowed_scenes: Sequence[str],
    required_access: Mapping[str, bool],
    lineage_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source region semantic sidecar must be a mapping")
    payload = dict(value)
    if payload.get("schema") != schema:
        raise ValueError("unexpected source region semantic schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unexpected source region semantic schema_version")
    if payload.get("contract") != dict(contract):
        raise ValueError("source region semantic contract differs")
    if payload.get("partition") != partition or payload.get("scene_id") not in allowed_scenes:
        raise PermissionError("region semantic sidecar is outside its frozen partition")
    access = payload.get("source_access", {})
    for key, expected in required_access.items():
        if access.get(key) is not expected:
            raise PermissionError(f"source region semantic sidecar violates {key}")
    distribution = torch.as_tensor(payload.get("nyu40_class_distribution")).float()
    valid = torch.as_tensor(payload.get("valid"))
    if distribution.ndim != 2 or distribution.shape[1] != 41:
        raise ValueError("NYU40 distribution must be [R,41]")
    rows = int(distribution.shape[0])
    if valid.shape != (rows,) or valid.dtype != torch.bool:
        raise ValueError("source semantic valid mask differs")
    if not bool(torch.isfinite(distribution).all()) or bool(
        ((distribution < 0) | (distribution > 1)).any()
    ):
        raise ValueError("NYU40 distribution must be finite [0,1]")
    expected_sum = torch.where(valid, torch.ones(rows), torch.zeros(rows))
    torch.testing.assert_close(distribution.sum(dim=1), expected_sum, rtol=0.0, atol=1e-6)
    tensor_names = {
        "canonical_region_indices",
        "nyu40_class_distribution",
        "valid",
        "geometry_coverage",
        "semantic_coverage",
        "semantic_purity",
        "region_member_count",
        "within_distance_member_count",
        "annotated_member_count",
        "nearest_distance_mean_meters",
        "nearest_distance_p95_meters",
    }
    tensors = {key: torch.as_tensor(payload.get(key)).detach().cpu() for key in tensor_names}
    for name in (
        "canonical_region_indices",
        "geometry_coverage",
        "semantic_coverage",
        "semantic_purity",
        "region_member_count",
        "within_distance_member_count",
        "annotated_member_count",
        "nearest_distance_mean_meters",
        "nearest_distance_p95_meters",
    ):
        if tensors[name].shape != (rows,):
            raise ValueError(f"source semantic channel {name} differs")
    channels = payload.get("channel_sha256")
    if not isinstance(channels, Mapping) or set(channels) != tensor_names:
        raise ValueError("source region semantic channel hashes differ")
    for key, tensor in tensors.items():
        if channels.get(key) != _tensor_sha256(tensor):
            raise ValueError(f"source region semantic channel changed: {key}")
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping) or len(lineage) != lineage_count:
        raise ValueError("source region semantic lineage differs")
    for key, record in lineage.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"source region semantic lineage {key} differs")
        path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        digest = str(record.get("sha256", ""))
        if _SHA256.fullmatch(digest) is None or sha256_file(path) != digest:
            raise ValueError(f"source region semantic lineage changed: {key}")
    return {**payload, **tensors}


def validate_source_region_semantic_sidecar(value: object) -> dict[str, Any]:
    return _validate_region_semantic_sidecar(
        value,
        schema=SCANNET_SOURCE_REGION_SEMANTIC_SCHEMA,
        contract=source_region_semantic_contract(),
        partition="source_fit",
        allowed_scenes=PREREGISTERED_SOURCE_FIT_SCENES,
        required_access={
            "official_scannet_train_scene": True,
            "source_fit_semantic_labels_opened": True,
            "development_semantic_labels_opened": False,
            "test_semantic_labels_opened": False,
            "agile3d_instance_ids_opened": False,
            "pseudo_semantic_labels_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_threshold_tuning": False,
        },
        lineage_count=7,
    )


def validate_development_region_semantic_sidecar(value: object) -> dict[str, Any]:
    return _validate_region_semantic_sidecar(
        value,
        schema=SCANNET_DEVELOPMENT_REGION_SEMANTIC_SCHEMA,
        contract=development_region_semantic_contract(),
        partition="development",
        allowed_scenes=PREREGISTERED_DEVELOPMENT_SCENES,
        required_access={
            "official_scannet_train_scene": True,
            "source_fit_semantic_labels_opened": False,
            "development_semantic_labels_opened": True,
            "test_semantic_labels_opened": False,
            "agile3d_instance_ids_opened": False,
            "pseudo_semantic_labels_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
            "per_scene_threshold_tuning": False,
            "parameter_callback_allowed": False,
        },
        lineage_count=8,
    )


__all__ = [
    "MESH_NEAREST_MAX_DISTANCE_METERS",
    "PREREGISTERED_DEVELOPMENT_SCENES",
    "PREREGISTERED_SOURCE_FIT_SCENES",
    "SCANNET_SOURCE_REGION_SEMANTIC_SCHEMA",
    "SCANNET_DEVELOPMENT_REGION_SEMANTIC_SCHEMA",
    "build_development_region_semantic_sidecar",
    "build_source_region_semantic_sidecar",
    "load_scannet_raw_to_nyu40",
    "development_region_semantic_contract",
    "official_vertex_nyu40_labels",
    "sha256_file",
    "source_region_semantic_contract",
    "validate_source_region_semantic_sidecar",
    "validate_development_region_semantic_sidecar",
]
