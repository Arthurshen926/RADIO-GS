#!/usr/bin/env python3
"""Train the sole clean-source accepted-V2 full-scalar residual.

The program consumes only pre-materialized, caller-SHA-bound source shards.
Accepted-V2 descriptors and official multi-view SigLIP2 descriptors are
immutable tensors; the only optimized object is
``SurfaceRegionAcceptedV2FullScalarResidualV1``.  The validation cohort is
scene-disjoint from the training cohort and is used only for a strict
non-regression checkpoint gate against the immutable accepted-V2 base.

Benchmark queries, labels, masks, target held-out images, per-scene knobs, and
online teacher/model execution are deliberately absent from this interface.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    REGION_CAP_PER_SCENE,
    SAMPLING_CONTRACT_SHA256,
    SPARSE_V2_PREREG_FILE_SHA256,
    VIEW_CAP_PER_REGION,
    validate_selection_audit,
    validate_sparse_pair_cardinality,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
    SURFACE_REGION_FULL_SCALAR_DIM,
    SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
    apply_full_scalar_normalization,
    build_full_scalar_normalization_authority,
    validate_full_scalar_normalization_authority,
)
from radio_gs.interfaces.surface_region_full_scalar_residual_checkpoint import (
    write_surface_region_full_scalar_residual_checkpoint,
)
from radio_gs.interfaces.surface_region_full_scalar_training_certificate import (
    build_training_certificate_payload,
)
from radio_gs.interfaces.surface_region_summary import (
    ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
    surface_region_state_dict_sha256,
)
from radio_gs.losses.dual_descriptor_response_risk import (
    RELATION_GRAM_WEIGHT,
)
from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionAcceptedV2FullScalarResidualV1,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


TRAINING_SHARD_SCHEMA = (
    "radio_gs.surface_region_full_scalar_residual_training_shard.v2"
)
TRAINING_SHARD_SCHEMA_VERSION = 2
COHORT_AUTHORITY_SCHEMA = (
    "radio_gs.surface_region_full_scalar_residual_clean_cohort_authority.v2"
)
COHORT_AUTHORITY_SCHEMA_VERSION = 2
SOURCE_STATE_MANIFEST_SCHEMA = (
    "radio_gs.surface_region_full_scalar_source_state_manifest.v2"
)
TEACHER_MANIFEST_SCHEMA = (
    "radio_gs.surface_region_full_scalar_teacher_manifest.v2"
)
BENCHMARK_EXCLUSION_MANIFEST_SCHEMA = (
    "radio_gs.surface_region_full_scalar_benchmark_exclusion_manifest.v2"
)
SOURCE_MANIFEST_SCHEMA_VERSION = 2
TRAINING_ARTIFACT_TYPE = (
    "surface_region_accepted_v2_full_scalar_residual_source_only_training_v1"
)
DESCRIPTOR_DIM = 1536
TRAIN_SCENE_COUNT = 24
VALIDATION_SCENE_COUNT = 8
SEED = 0
EPOCHS = 30
PATIENCE = 8
BATCH_ROWS = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRADIENT_NORM = 1.0
MAX_ANGLE_RADIANS = 0.15
MAX_ALPHA = 0.25
NON_REGRESSION_TOLERANCE = 1e-7
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCANNET_SCENE_ID = re.compile(r"^(scene\d{4})_\d{2}$")


def canonical_physical_space_id(scene_id: object) -> str:
    """Return the ScanNet physical-space ID shared by repeated scans.

    ScanNet's ``scene####_00`` / ``_01`` / ... suffixes are repeated scans of
    one physical space, not independent scenes.  Clean train/validation and
    benchmark exclusion therefore operate on ``scene####``.  This source-only
    trainer is deliberately fail-closed for other naming schemes until a
    separately versioned dataset-specific physical-space authority exists.
    """

    value = str(scene_id)
    matched = _SCANNET_SCENE_ID.fullmatch(value)
    if matched is None:
        raise ValueError(
            "full-scalar clean cohort scene ID must use ScanNet scene####_##"
        )
    return matched.group(1)


def _source_manifest_access() -> dict[str, bool]:
    return {
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
    }


def source_state_manifest_contract() -> dict[str, Any]:
    return {
        "schema": SOURCE_STATE_MANIFEST_SCHEMA,
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "scene_records": (
            "sorted_unique_scene_id_plus_canonical_physical_space_id_plus_"
            "artifact_sha256"
        ),
        "region_records": "sorted_unique_region_row_id_plus_scene_id",
        "authority_sha256": "canonical_content_without_authority_sha256",
        "source_access": _source_manifest_access(),
    }


def teacher_manifest_contract() -> dict[str, Any]:
    return {
        "schema": TEACHER_MANIFEST_SCHEMA,
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "teacher_model_authority_sha256": True,
        "region_view_records": (
            "sorted_unique_region_row_id_scene_id_and_ordered_teacher_view_ids"
        ),
        "authority_sha256": "canonical_content_without_authority_sha256",
        "source_access": _source_manifest_access(),
    }


def benchmark_exclusion_manifest_contract() -> dict[str, Any]:
    return {
        "schema": BENCHMARK_EXCLUSION_MANIFEST_SCHEMA,
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "source_identifier": True,
        "source_artifact_sha256": True,
        "scene_ids": "sorted_unique_nonempty",
        "scene_ids_sha256": "canonical_json_sha256",
        "physical_space_ids": (
            "sorted_unique_canonical_ScanNet_scene####_derived_from_scene_ids"
        ),
        "physical_space_ids_sha256": "canonical_json_sha256",
        "authority_sha256": "canonical_content_without_authority_sha256",
        "source_access": _source_manifest_access(),
    }


SOURCE_STATE_MANIFEST_CONTRACT_SHA256 = canonical_json_sha256(
    source_state_manifest_contract()
)
TEACHER_MANIFEST_CONTRACT_SHA256 = canonical_json_sha256(
    teacher_manifest_contract()
)
BENCHMARK_EXCLUSION_MANIFEST_CONTRACT_SHA256 = canonical_json_sha256(
    benchmark_exclusion_manifest_contract()
)


def _manifest_content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("authority_sha256", None)
    return canonical_json_sha256(content)


def _validate_manifest_header(
    value: object,
    *,
    schema: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
    required: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    manifest = dict(value)
    if set(manifest) != required:
        raise ValueError(f"{label} fields differ")
    if (
        manifest.get("schema") != schema
        or manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("contract") != dict(contract)
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("source_access") != _source_manifest_access()
        or any(manifest.get("source_access", {}).values())
    ):
        raise ValueError(f"{label} contract or access differs")
    authority_sha = _require_sha256(
        manifest.get("authority_sha256"), label=f"{label} authority"
    )
    if _manifest_content_sha256(manifest) != authority_sha:
        raise ValueError(f"{label} content SHA-256 differs")
    return manifest


def validate_source_state_manifest(value: object) -> dict[str, Any]:
    manifest = _validate_manifest_header(
        value,
        schema=SOURCE_STATE_MANIFEST_SCHEMA,
        contract=source_state_manifest_contract(),
        contract_sha256=SOURCE_STATE_MANIFEST_CONTRACT_SHA256,
        required={
            "schema", "schema_version", "contract", "contract_sha256",
            "scene_records", "region_records", "source_access",
            "authority_sha256",
        },
        label="source-state manifest",
    )
    scene_records = manifest.get("scene_records")
    region_records = manifest.get("region_records")
    if not isinstance(scene_records, list) or not scene_records:
        raise ValueError("source-state manifest scene records differ")
    frozen_scenes: list[dict[str, str]] = []
    for record in scene_records:
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {"scene_id", "physical_space_id", "artifact_sha256"}
            or not isinstance(record.get("scene_id"), str)
            or not record.get("scene_id")
            or record.get("physical_space_id")
            != canonical_physical_space_id(record.get("scene_id"))
        ):
            raise ValueError("source-state manifest scene record differs")
        frozen_scenes.append({
            "scene_id": str(record["scene_id"]),
            "physical_space_id": str(record["physical_space_id"]),
            "artifact_sha256": _require_sha256(
                record["artifact_sha256"], label="source-state scene artifact"
            ),
        })
    if (
        [item["scene_id"] for item in frozen_scenes]
        != sorted(item["scene_id"] for item in frozen_scenes)
        or len({item["scene_id"] for item in frozen_scenes}) != len(frozen_scenes)
        or len({item["physical_space_id"] for item in frozen_scenes})
        != len(frozen_scenes)
    ):
        raise ValueError("source-state manifest scene or physical-space IDs differ")
    known_scenes = {item["scene_id"] for item in frozen_scenes}
    if not isinstance(region_records, list) or not region_records:
        raise ValueError("source-state manifest region records differ")
    frozen_regions: list[dict[str, str]] = []
    for record in region_records:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"region_row_id", "scene_id"}
            or not isinstance(record.get("region_row_id"), str)
            or not record.get("region_row_id")
            or record.get("scene_id") not in known_scenes
        ):
            raise ValueError("source-state manifest region record differs")
        frozen_regions.append({
            "region_row_id": str(record["region_row_id"]),
            "scene_id": str(record["scene_id"]),
        })
    region_ids = [item["region_row_id"] for item in frozen_regions]
    if region_ids != sorted(region_ids) or len(set(region_ids)) != len(region_ids):
        raise ValueError("source-state manifest region IDs differ")
    return {**manifest, "scene_records": frozen_scenes, "region_records": frozen_regions}


def validate_teacher_manifest(value: object) -> dict[str, Any]:
    manifest = _validate_manifest_header(
        value,
        schema=TEACHER_MANIFEST_SCHEMA,
        contract=teacher_manifest_contract(),
        contract_sha256=TEACHER_MANIFEST_CONTRACT_SHA256,
        required={
            "schema", "schema_version", "contract", "contract_sha256",
            "teacher_model_authority_sha256", "region_view_records",
            "source_access", "authority_sha256",
        },
        label="teacher manifest",
    )
    _require_sha256(
        manifest.get("teacher_model_authority_sha256"),
        label="teacher model authority",
    )
    records = manifest.get("region_view_records")
    if not isinstance(records, list) or not records:
        raise ValueError("teacher manifest region/view records differ")
    frozen: list[dict[str, Any]] = []
    for record in records:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"region_row_id", "scene_id", "teacher_view_ids"}
            or not isinstance(record.get("region_row_id"), str)
            or not record.get("region_row_id")
            or not isinstance(record.get("scene_id"), str)
            or not record.get("scene_id")
            or not isinstance(record.get("teacher_view_ids"), list)
            or any(
                not isinstance(view, str) or not view
                for view in record["teacher_view_ids"]
            )
            or len(set(record["teacher_view_ids"]))
            != len(record["teacher_view_ids"])
        ):
            raise ValueError("teacher manifest region/view record differs")
        frozen.append({
            "region_row_id": str(record["region_row_id"]),
            "scene_id": str(record["scene_id"]),
            "teacher_view_ids": list(record["teacher_view_ids"]),
        })
    region_ids = [item["region_row_id"] for item in frozen]
    if region_ids != sorted(region_ids) or len(set(region_ids)) != len(region_ids):
        raise ValueError("teacher manifest region IDs differ")
    return {**manifest, "region_view_records": frozen}


def validate_benchmark_exclusion_manifest(value: object) -> dict[str, Any]:
    manifest = _validate_manifest_header(
        value,
        schema=BENCHMARK_EXCLUSION_MANIFEST_SCHEMA,
        contract=benchmark_exclusion_manifest_contract(),
        contract_sha256=BENCHMARK_EXCLUSION_MANIFEST_CONTRACT_SHA256,
        required={
            "schema", "schema_version", "contract", "contract_sha256",
            "source_identifier", "source_artifact_sha256", "scene_ids",
            "scene_ids_sha256", "physical_space_ids",
            "physical_space_ids_sha256", "source_access",
            "authority_sha256",
        },
        label="benchmark exclusion manifest",
    )
    if not isinstance(manifest.get("source_identifier"), str) or not manifest.get(
        "source_identifier"
    ):
        raise ValueError("benchmark exclusion source identifier differs")
    _require_sha256(
        manifest.get("source_artifact_sha256"),
        label="benchmark exclusion source artifact",
    )
    scenes = manifest.get("scene_ids")
    if (
        not isinstance(scenes, list)
        or not scenes
        or any(not isinstance(scene, str) or not scene for scene in scenes)
        or scenes != sorted(scenes)
        or len(set(scenes)) != len(scenes)
    ):
        raise ValueError("benchmark exclusion scene IDs differ")
    if canonical_json_sha256(scenes) != _require_sha256(
        manifest.get("scene_ids_sha256"), label="benchmark exclusion scene list"
    ):
        raise ValueError("benchmark exclusion scene list SHA-256 differs")
    physical_spaces = manifest.get("physical_space_ids")
    expected_physical_spaces = sorted(
        {canonical_physical_space_id(scene) for scene in scenes}
    )
    if (
        physical_spaces != expected_physical_spaces
        or canonical_json_sha256(physical_spaces)
        != _require_sha256(
            manifest.get("physical_space_ids_sha256"),
            label="benchmark exclusion physical-space list",
        )
    ):
        raise ValueError("benchmark exclusion physical-space IDs differ")
    return {
        **manifest,
        "scene_ids": list(scenes),
        "physical_space_ids": list(physical_spaces),
    }


def _load_json_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
    validator: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = _require_sha256(expected_sha256, label=label)
    value, observed, source = load_json_object(
        path, expected_sha256=expected, label=label
    )
    return validator(value), {"path": str(source), "sha256": observed}


def _accepted_v2_authority() -> dict[str, str]:
    return {
        "checkpoint_sha256": ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
        "architecture_sha256": ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
        "state_dict_sha256": ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
        "provenance_sha256": ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
        "contract_sha256": ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    }


def _tensor_channel_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def training_shard_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "accepted_v2_e0": _tensor_channel_sha256(value["accepted_v2_e0"]),
        "raw_full_scalar_summary": _tensor_channel_sha256(
            value["raw_full_scalar_summary"]
        ),
        "eligible": _tensor_channel_sha256(value["eligible"]),
        "official_multiview_siglip2_teacher_pair_region_indices": (
            _tensor_channel_sha256(
                value["official_multiview_siglip2_teacher_pair_region_indices"]
            )
        ),
        "official_multiview_siglip2_teacher_pair_descriptors": _tensor_channel_sha256(
            value["official_multiview_siglip2_teacher_pair_descriptors"]
        ),
        "scene_ids": canonical_json_sha256(value["scene_ids"]),
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "teacher_pair_view_ids": canonical_json_sha256(
            value["teacher_pair_view_ids"]
        ),
        "sampling_audit": canonical_json_sha256(value["sampling_audit"]),
    }


def training_shard_contract() -> dict[str, Any]:
    """Return the immutable pre-materialized source-shard contract."""

    return {
        "schema": TRAINING_SHARD_SCHEMA,
        "schema_version": TRAINING_SHARD_SCHEMA_VERSION,
        "accepted_v2_authority": _accepted_v2_authority(),
        "accepted_v2_e0": {
            "dtype": "float32",
            "shape": ["rows", DESCRIPTOR_DIM],
            "gauge": "unit_l2",
            "role": "immutable_external_base",
        },
        "raw_full_scalar_summary": {
            "dtype": "float32",
            "shape": ["rows", SURFACE_REGION_FULL_SCALAR_DIM],
            "names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
            "contract_sha256": SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
            "normalization": "none_pre_materialized_raw18",
        },
        "eligible": (
            "accepted_v2_valid_and_exact_state_valid_overlap_only"
        ),
        "official_teacher": {
            "kind": "c_radio_v4_h_official_siglip2_g_multiview_descriptor",
            "dtype": "float32",
            "storage": "sparse_coo_pair_rows_no_dense_row_by_view_tensor",
            "shape": ["pairs", DESCRIPTOR_DIM],
            "pair_region_indices": "sorted_local_row_indices",
            "maximum_pairs_per_row": 4,
            "gauge": "unit_l2_every_pair",
            "role": "immutable_external_teacher",
        },
        "row_authority": {
            "region_row_ids": "globally_unique_stable_strings",
            "teacher_pair_view_ids": "pair_aligned_stable_strings",
            "channel_sha256": "dtype_shape_and_exact_content_sha256",
        },
        "lineage": (
            "accepted_source_state_cohort_file_content_and_teacher_caller_sha_bound"
        ),
        "cohort": {
            "train_scenes": TRAIN_SCENE_COUNT,
            "validation_scenes": VALIDATION_SCENE_COUNT,
            "scene_disjoint": True,
            "per_scene_hyperparameters": False,
        },
        "sampling": {
            "contract_sha256": SAMPLING_CONTRACT_SHA256,
            "preregistration_file_sha256": SPARSE_V2_PREREG_FILE_SHA256,
            "per_scene_region_cap": REGION_CAP_PER_SCENE,
            "per_region_view_cap": VIEW_CAP_PER_REGION,
            "batch_local_pair_gather_only": True,
            "global_teacher_densification_allowed": False,
        },
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
    }


TRAINING_SHARD_CONTRACT_SHA256 = canonical_json_sha256(training_shard_contract())


def _cohort_authority_access() -> dict[str, bool]:
    """Return the exact all-false benchmark-data access declaration."""

    return {
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "per_scene_hyperparameters": False,
    }


def cohort_authority_contract() -> dict[str, Any]:
    """Return the contract for the independently frozen clean cohort."""

    return {
        "schema": COHORT_AUTHORITY_SCHEMA,
        "schema_version": COHORT_AUTHORITY_SCHEMA_VERSION,
        "source_train_scene_ids": {
            "count": TRAIN_SCENE_COUNT,
            "unique": True,
            "sorted": True,
        },
        "source_validation_scene_ids": {
            "count": VALIDATION_SCENE_COUNT,
            "unique": True,
            "sorted": True,
        },
        "source_train_physical_space_ids": {
            "count": TRAIN_SCENE_COUNT,
            "unique": True,
            "sorted": True,
            "derived_from_scene_ids": "ScanNet_scene####",
        },
        "source_validation_physical_space_ids": {
            "count": VALIDATION_SCENE_COUNT,
            "unique": True,
            "sorted": True,
            "derived_from_scene_ids": "ScanNet_scene####",
        },
        "scene_and_physical_space_disjoint": True,
        "benchmark_exclusion": {
            "external_manifest_authority_sha256_required": True,
            "external_manifest_file_sha256_required": True,
            "clean_scenes_and_physical_spaces_must_not_appear": True,
        },
        "authority_sha256": (
            "canonical_json_sha256_of_all_payload_fields_except_authority_sha256"
        ),
        "source_access": _cohort_authority_access(),
    }


COHORT_AUTHORITY_CONTRACT_SHA256 = canonical_json_sha256(
    cohort_authority_contract()
)


def cohort_authority_content_sha256(value: Mapping[str, Any]) -> str:
    """Hash authority content without the self-digest to avoid recursion."""

    content = dict(value)
    content.pop("authority_sha256", None)
    return canonical_json_sha256(content)


def validate_cohort_authority_payload(value: object) -> dict[str, Any]:
    """Validate the independent 24/8 clean-scene authority fail closed."""

    if not isinstance(value, Mapping):
        raise ValueError("clean cohort authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "source_train_scene_ids",
        "source_validation_scene_ids",
        "source_train_physical_space_ids",
        "source_validation_physical_space_ids",
        "benchmark_exclusion",
        "source_access",
        "authority_sha256",
    }
    if set(authority) != required:
        raise ValueError("clean cohort authority fields differ")
    if (
        authority.get("schema") != COHORT_AUTHORITY_SCHEMA
        or authority.get("schema_version") != COHORT_AUTHORITY_SCHEMA_VERSION
        or authority.get("contract") != cohort_authority_contract()
        or authority.get("contract_sha256")
        != COHORT_AUTHORITY_CONTRACT_SHA256
        or authority.get("source_access") != _cohort_authority_access()
        or any(authority.get("source_access", {}).values())
    ):
        raise ValueError("clean cohort authority contract or access differs")

    train_scenes = authority.get("source_train_scene_ids")
    validation_scenes = authority.get("source_validation_scene_ids")
    if (
        not isinstance(train_scenes, list)
        or len(train_scenes) != TRAIN_SCENE_COUNT
        or any(not isinstance(scene, str) or not scene for scene in train_scenes)
        or len(set(train_scenes)) != TRAIN_SCENE_COUNT
        or train_scenes != sorted(train_scenes)
    ):
        raise ValueError("clean cohort authority train scenes differ")
    if (
        not isinstance(validation_scenes, list)
        or len(validation_scenes) != VALIDATION_SCENE_COUNT
        or any(
            not isinstance(scene, str) or not scene
            for scene in validation_scenes
        )
        or len(set(validation_scenes)) != VALIDATION_SCENE_COUNT
        or validation_scenes != sorted(validation_scenes)
    ):
        raise ValueError("clean cohort authority validation scenes differ")
    if set(train_scenes) & set(validation_scenes):
        raise ValueError("clean cohort authority train/validation scenes overlap")
    train_physical_spaces = authority.get("source_train_physical_space_ids")
    validation_physical_spaces = authority.get(
        "source_validation_physical_space_ids"
    )
    expected_train_physical_spaces = sorted(
        {canonical_physical_space_id(scene) for scene in train_scenes}
    )
    expected_validation_physical_spaces = sorted(
        {canonical_physical_space_id(scene) for scene in validation_scenes}
    )
    if (
        train_physical_spaces != expected_train_physical_spaces
        or len(expected_train_physical_spaces) != TRAIN_SCENE_COUNT
        or validation_physical_spaces != expected_validation_physical_spaces
        or len(expected_validation_physical_spaces) != VALIDATION_SCENE_COUNT
    ):
        raise ValueError("clean cohort authority physical-space IDs differ")
    if set(train_physical_spaces) & set(validation_physical_spaces):
        raise ValueError(
            "clean cohort authority train/validation physical spaces overlap"
        )

    exclusion = authority.get("benchmark_exclusion")
    exclusion_required = {
        "manifest_authority_sha256",
        "manifest_file_sha256",
    }
    if not isinstance(exclusion, Mapping) or set(exclusion) != exclusion_required:
        raise ValueError("clean cohort benchmark exclusion fields differ")
    exclusion = dict(exclusion)
    _require_sha256(
        exclusion.get("manifest_authority_sha256"),
        label="benchmark exclusion manifest authority",
    )
    _require_sha256(
        exclusion.get("manifest_file_sha256"),
        label="benchmark exclusion manifest file",
    )

    expected_content_sha = _require_sha256(
        authority.get("authority_sha256"), label="clean cohort authority content"
    )
    if cohort_authority_content_sha256(authority) != expected_content_sha:
        raise ValueError("clean cohort authority content SHA-256 differs")
    return {
        **authority,
        "source_train_scene_ids": list(train_scenes),
        "source_validation_scene_ids": list(validation_scenes),
        "source_train_physical_space_ids": list(train_physical_spaces),
        "source_validation_physical_space_ids": list(
            validation_physical_spaces
        ),
        "benchmark_exclusion": {
            **exclusion,
        },
    }


def load_cohort_authority(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the cohort JSON using a caller-trusted file SHA-256."""

    expected = _require_sha256(expected_sha256, label="clean cohort authority")
    value, observed, source = load_json_object(
        path,
        expected_sha256=expected,
        label="clean full-scalar residual cohort authority",
    )
    authority = validate_cohort_authority_payload(value)
    return authority, {"path": str(source), "sha256": observed}


def training_contract() -> dict[str, Any]:
    """Return the single global training, selection, and access policy."""

    return {
        "schema_version": 2,
        "artifact_type": TRAINING_ARTIFACT_TYPE,
        "seed": SEED,
        "input_shard_contract_sha256": TRAINING_SHARD_CONTRACT_SHA256,
        "cohort_authority_contract_sha256": (
            COHORT_AUTHORITY_CONTRACT_SHA256
        ),
        "sampling": {
            "contract_sha256": SAMPLING_CONTRACT_SHA256,
            "preregistration_file_sha256": SPARSE_V2_PREREG_FILE_SHA256,
            "per_scene_region_cap": REGION_CAP_PER_SCENE,
            "per_region_view_cap": VIEW_CAP_PER_REGION,
            "storage": "sparse_coo_pairs_plus_merged_csr_offsets",
            "batch_local_pair_gather_only": True,
            "global_or_cohort_teacher_densification": False,
        },
        "external_manifests": {
            "source_state_contract_sha256": SOURCE_STATE_MANIFEST_CONTRACT_SHA256,
            "teacher_contract_sha256": TEACHER_MANIFEST_CONTRACT_SHA256,
            "benchmark_exclusion_contract_sha256": (
                BENCHMARK_EXCLUSION_MANIFEST_CONTRACT_SHA256
            ),
            "caller_file_sha256_required": True,
            "stable_region_and_view_alignment_required": True,
        },
        "cohort": {
            "source_train_scene_count": TRAIN_SCENE_COUNT,
            "source_validation_scene_count": VALIDATION_SCENE_COUNT,
            "scene_disjoint": True,
            "physical_space_id": "canonical_ScanNet_scene####",
            "one_scan_per_physical_space": True,
            "train_validation_physical_space_disjoint": True,
            "external_caller_sha_bound_authority_json": True,
            "actual_scene_sets_equal_authority": True,
            "benchmark_exclusion_list_verified": True,
            "benchmark_physical_space_exclusion_verified": True,
            "cohort_authority_identical_across_all_shards": True,
            "source_state_cohort_authority_identical_across_all_shards": True,
        },
        "model": {
            "class": "SurfaceRegionAcceptedV2FullScalarResidualV1",
            "descriptor_dim": DESCRIPTOR_DIM,
            "scalar_dim": SURFACE_REGION_FULL_SCALAR_DIM,
            "hidden_dim": (
                SurfaceRegionAcceptedV2FullScalarResidualV1.HIDDEN_DIM
            ),
            "max_angle_radians": MAX_ANGLE_RADIANS,
            "max_alpha": MAX_ALPHA,
            "zero_final_projection": True,
            "trainable_object": "content_plus_scalar_full_scalar_residual_only",
        },
        "frozen": {
            "accepted_v2_e0": True,
            "official_multiview_siglip2_teacher": True,
            "source_state": True,
            "normalization_after_source_train_fit": True,
        },
        "normalization": {
            "fit_split": "source_train",
            "fit_rows": "eligible_only",
            "validation_contribution": False,
            "ood_rows_trainable": False,
            "ood_action": "bitwise_accepted_v2_base_fallback",
        },
        "objective": {
            "all_view_cosine_weight": 1.0,
            "relation_gram_smooth_l1_weight": RELATION_GRAM_WEIGHT,
            "view_aggregation": "row_mean_then_scene_mean",
            "relation_scope": "within_scene_strict_upper_off_diagonal",
            "scene_weighting": "equal_scene_gradient_accumulation",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_rows": BATCH_ROWS,
            "max_gradient_norm": MAX_GRADIENT_NORM,
        },
        "selection": {
            "split": "source_validation",
            "aggregation": "row_mean_then_scene_macro",
            "epoch_zero_is_candidate": True,
            "non_regression_tolerance": NON_REGRESSION_TOLERANCE,
            "minimum_eligible_rows_per_scene": 2,
            "minimum_in_domain_rows_per_scene": 2,
            "vacuous_in_domain_fallback_gate_allowed": False,
            "required": [
                "every_validation_scene_has_two_eligible_rows",
                "every_validation_scene_has_two_in_domain_rows",
                "mean_all_view_cosine_not_below_base",
                "p05_row_mean_all_view_cosine_not_below_base",
                "within_scene_relation_fidelity_not_below_base",
                "paired_scene_p05_and_worst_not_below_base",
                "in_domain_gate_not_below_base",
                "ood_bitwise_accepted_v2_base",
            ],
            "ranking": [
                "maximum_mean_all_view_cosine",
                "maximum_p05_row_mean_all_view_cosine",
                "maximum_within_scene_relation_fidelity",
                "earliest_epoch",
            ],
        },
        "promotion": {
            "order": "normalization_certificate_checkpoint_report",
            "checkpoint_binds_training_certificate_file_sha256": True,
            "runtime_caller_sha_loads_certificate": True,
        },
        "prohibited": {
            "benchmark_query_label_mask_or_target_heldout": True,
            "per_scene_hyperparameter_or_checkpoint_selection": True,
            "online_teacher_execution": True,
            "accepted_v2_finetuning": True,
        },
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _source_access(split: str) -> dict[str, Any]:
    return {
        "split": split,
        "source_only": True,
        "query_independent": True,
        "clean_scene_cohort": True,
        "benchmark_scenes_excluded_by_cohort_authority": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "per_scene_hyperparameters": False,
    }


def _validate_lineage(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("full-scalar training shard lineage must be a mapping")
    lineage = dict(value)
    required = {
        "accepted_v2_authority",
        "source_state_cohort_authority_sha256",
        "source_state_manifest_file_sha256",
        "cohort_authority_sha256",
        "cohort_authority_file_sha256",
        "teacher_authority_sha256",
        "teacher_manifest_file_sha256",
    }
    if set(lineage) != required:
        raise ValueError("full-scalar training shard lineage fields differ")
    if lineage.get("accepted_v2_authority") != _accepted_v2_authority():
        raise ValueError("training shard accepted-V2 authority differs")
    for name in required - {"accepted_v2_authority"}:
        _require_sha256(lineage.get(name), label=name.replace("_", " "))
    return lineage


def validate_training_shard_payload(
    value: object,
    *,
    expected_split: str | None = None,
) -> dict[str, Any]:
    """Validate one exact-key source shard without trusting descriptive data."""

    if not isinstance(value, Mapping):
        raise ValueError("full-scalar training shard must contain a mapping")
    shard = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "split",
        "accepted_v2_e0",
        "raw_full_scalar_summary",
        "eligible",
        "official_multiview_siglip2_teacher_pair_region_indices",
        "official_multiview_siglip2_teacher_pair_descriptors",
        "scene_ids",
        "region_row_ids",
        "teacher_pair_view_ids",
        "sampling_audit",
        "channel_sha256",
        "lineage",
        "source_access",
    }
    if set(shard) != required:
        raise ValueError("full-scalar training shard fields differ")
    split = str(shard.get("split", ""))
    if split not in {"source_train", "source_validation"}:
        raise ValueError("full-scalar training shard split differs")
    if expected_split is not None and split != str(expected_split):
        raise ValueError("full-scalar training shard split differs from caller")
    if (
        shard.get("schema") != TRAINING_SHARD_SCHEMA
        or shard.get("schema_version") != TRAINING_SHARD_SCHEMA_VERSION
        or shard.get("contract") != training_shard_contract()
        or shard.get("contract_sha256") != TRAINING_SHARD_CONTRACT_SHA256
        or shard.get("source_access") != _source_access(split)
    ):
        raise ValueError("full-scalar training shard contract differs")
    lineage = _validate_lineage(shard.get("lineage"))

    base = shard.get("accepted_v2_e0")
    scalars = shard.get("raw_full_scalar_summary")
    eligible = shard.get("eligible")
    pair_rows = shard.get(
        "official_multiview_siglip2_teacher_pair_region_indices"
    )
    teachers = shard.get("official_multiview_siglip2_teacher_pair_descriptors")
    scene_ids = shard.get("scene_ids")
    region_row_ids = shard.get("region_row_ids")
    teacher_view_ids = shard.get("teacher_pair_view_ids")
    sampling_audit = shard.get("sampling_audit")
    pair_count = int(pair_rows.numel()) if torch.is_tensor(pair_rows) else -1
    validate_sparse_pair_cardinality(
        selected_region_count=(int(base.shape[0]) if torch.is_tensor(base) and base.ndim else -1),
        pair_count=pair_count,
    )
    if (
        not torch.is_tensor(base)
        or base.dtype != torch.float32
        or base.ndim != 2
        or base.shape[1] != DESCRIPTOR_DIM
        or not 0 < base.shape[0] <= REGION_CAP_PER_SCENE
        or not bool(torch.isfinite(base).all())
        or not torch.is_tensor(scalars)
        or scalars.dtype != torch.float32
        or scalars.shape != (base.shape[0], SURFACE_REGION_FULL_SCALAR_DIM)
        or not bool(torch.isfinite(scalars).all())
        or not torch.is_tensor(eligible)
        or eligible.dtype != torch.bool
        or eligible.shape != (base.shape[0],)
        or not torch.is_tensor(pair_rows)
        or pair_rows.dtype != torch.long
        or pair_rows.ndim != 1
        or pair_count <= 0
        or not torch.is_tensor(teachers)
        or teachers.dtype != torch.float32
        or teachers.shape != (pair_count, DESCRIPTOR_DIM)
        or not bool(torch.isfinite(teachers).all())
        or not isinstance(scene_ids, list)
        or len(scene_ids) != base.shape[0]
        or any(not isinstance(scene, str) or not scene for scene in scene_ids)
        or not isinstance(region_row_ids, list)
        or len(region_row_ids) != base.shape[0]
        or any(
            not isinstance(row_id, str) or not row_id
            for row_id in region_row_ids
        )
        or len(set(region_row_ids)) != len(region_row_ids)
        or not isinstance(teacher_view_ids, list)
        or len(teacher_view_ids) != pair_count
        or any(not isinstance(view_id, str) or not view_id for view_id in teacher_view_ids)
    ):
        raise ValueError("full-scalar training shard tensors or scene IDs differ")
    if (
        bool((pair_rows < 0).any())
        or bool((pair_rows >= base.shape[0]).any())
        or (pair_count > 1 and bool((pair_rows[1:] < pair_rows[:-1]).any()))
    ):
        raise ValueError("sparse teacher pair rows differ")
    row_counts = torch.bincount(pair_rows, minlength=base.shape[0])
    if bool((row_counts[eligible] <= 0).any()) or bool(
        (row_counts > VIEW_CAP_PER_REGION).any()
    ):
        raise ValueError("sparse teacher per-row coverage differs")
    for row in range(base.shape[0]):
        ids = [
            teacher_view_ids[index]
            for index in torch.where(pair_rows == row)[0].tolist()
        ]
        if len(set(ids)) != len(ids):
            raise ValueError("teacher view IDs must be unique within a region row")
    base_norm = torch.linalg.vector_norm(base, dim=-1)
    if not torch.allclose(
        base_norm,
        torch.ones_like(base_norm),
        rtol=0.0,
        atol=2e-4,
    ):
        raise ValueError("training shard accepted-V2 e0 must use unit L2 gauge")
    if bool((row_counts[~eligible] > 0).any()):
        raise ValueError("ineligible shard rows must not carry teacher observations")
    active_norm = torch.linalg.vector_norm(teachers, dim=-1)
    if not torch.allclose(
        active_norm, torch.ones_like(active_norm),
        rtol=0.0,
        atol=2e-4,
    ):
        raise ValueError("active official teacher descriptors must use unit L2 gauge")
    if not isinstance(sampling_audit, Mapping) or set(sampling_audit) != {
        "scene_id", "sampling_contract_sha256", "canonical_region_indices_sha256",
        "accepted_selection_audit", "selected_region_count", "pair_count",
        "maximum_views_per_region",
    }:
        raise ValueError("training shard sampling audit differs")
    _require_sha256(
        sampling_audit.get("canonical_region_indices_sha256"),
        label="canonical region indices",
    )
    accepted_audit = validate_selection_audit(
        sampling_audit.get("accepted_selection_audit"),
        selected_count=int(base.shape[0]),
    )
    if (
        sampling_audit.get("sampling_contract_sha256")
        != SAMPLING_CONTRACT_SHA256
        or sampling_audit.get("scene_id") not in set(scene_ids)
        or set(scene_ids) != {sampling_audit.get("scene_id")}
        or int(sampling_audit.get("selected_region_count", -1)) != base.shape[0]
        or int(sampling_audit.get("pair_count", -1)) != pair_count
        or int(sampling_audit.get("maximum_views_per_region", -1))
        != int(row_counts.max())
    ):
        raise ValueError("training shard sampling counts differ")
    channel_sha = shard.get("channel_sha256")
    expected_channel_sha = training_shard_channel_sha256(shard)
    if channel_sha != expected_channel_sha:
        raise ValueError("training shard channel SHA-256 authority differs")
    # Return detached, contiguous CPU values so callers never optimize an
    # input tensor or preserve an unsafe view into a deserialized storage.
    return {
        **shard,
        "accepted_v2_e0": base.detach().cpu().contiguous(),
        "raw_full_scalar_summary": scalars.detach().cpu().contiguous(),
        "eligible": eligible.detach().cpu().contiguous(),
        "official_multiview_siglip2_teacher_pair_region_indices": (
            pair_rows.detach().cpu().contiguous()
        ),
        "official_multiview_siglip2_teacher_pair_descriptors": (
            teachers.detach().cpu().contiguous()
        ),
        "scene_ids": list(scene_ids),
        "region_row_ids": list(region_row_ids),
        "teacher_pair_view_ids": list(teacher_view_ids),
        "sampling_audit": {
            **dict(sampling_audit),
            "accepted_selection_audit": accepted_audit,
        },
        "channel_sha256": dict(expected_channel_sha),
        "lineage": lineage,
    }


def load_training_shard(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_split: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load one shard through a stable descriptor and caller-trusted SHA."""

    expected = _require_sha256(expected_sha256, label="training shard")
    value, observed, source = load_torch_mapping(
        path,
        expected_sha256=expected,
        map_location="cpu",
        label="full-scalar residual training shard",
    )
    shard = validate_training_shard_payload(value, expected_split=expected_split)
    return shard, {"path": str(source), "sha256": observed}


def _pad_and_merge_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    if not shards:
        raise ValueError(f"{split} requires at least one training shard")
    values = [validate_training_shard_payload(item, expected_split=split) for item in shards]
    descriptor_dim = {int(item["accepted_v2_e0"].shape[1]) for item in values}
    if descriptor_dim != {DESCRIPTOR_DIM}:
        raise ValueError("full-scalar shard descriptor dimensions differ")
    pair_rows: list[torch.Tensor] = []
    pair_descriptors: list[torch.Tensor] = []
    pair_view_ids: list[str] = []
    row_offset = 0
    for item in values:
        pair_rows.append(
            item["official_multiview_siglip2_teacher_pair_region_indices"]
            + row_offset
        )
        pair_descriptors.append(
            item["official_multiview_siglip2_teacher_pair_descriptors"]
        )
        pair_view_ids.extend(item["teacher_pair_view_ids"])
        row_offset += int(item["accepted_v2_e0"].shape[0])
    merged_rows = torch.cat(pair_rows)
    row_counts = torch.bincount(merged_rows, minlength=row_offset)
    row_offsets = torch.zeros(row_offset + 1, dtype=torch.long)
    row_offsets[1:] = torch.cumsum(row_counts, dim=0)
    merged = {
        "accepted_v2_e0": torch.cat(
            [item["accepted_v2_e0"] for item in values], dim=0
        ),
        "raw_full_scalar_summary": torch.cat(
            [item["raw_full_scalar_summary"] for item in values], dim=0
        ),
        "eligible": torch.cat([item["eligible"] for item in values], dim=0),
        "official_multiview_siglip2_teacher_pair_region_indices": merged_rows,
        "official_multiview_siglip2_teacher_pair_descriptors": torch.cat(
            pair_descriptors, dim=0
        ),
        "official_multiview_siglip2_teacher_pair_row_offsets": row_offsets,
        "scene_ids": [scene for item in values for scene in item["scene_ids"]],
        "region_row_ids": [
            row_id for item in values for row_id in item["region_row_ids"]
        ],
        "teacher_pair_view_ids": pair_view_ids,
        "sampling_audits": [dict(item["sampling_audit"]) for item in values],
        "lineages": [dict(item["lineage"]) for item in values],
    }
    return merged


def _sampling_authority_for_certificate(
    train_data: Mapping[str, Any],
    validation_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze per-scene sparse selection/teacher counts without densification."""

    splits: dict[str, list[dict[str, Any]]] = {}
    for split, data in (
        ("source_train", train_data),
        ("source_validation", validation_data),
    ):
        records: list[dict[str, Any]] = []
        for audit in data["sampling_audits"]:
            accepted = audit["accepted_selection_audit"]
            records.append(
                {
                    "scene_id": str(audit["scene_id"]),
                    "canonical_candidate_region_count": int(
                        accepted["canonical_candidate_region_count"]
                    ),
                    "exact_overlap_candidate_count": int(
                        accepted["exact_overlap_candidate_count"]
                    ),
                    "teacher_visible_candidate_count": int(
                        accepted["teacher_visible_candidate_count"]
                    ),
                    "selected_region_count": int(audit["selected_region_count"]),
                    "selected_count_by_scale": list(
                        accepted["selected_count_by_scale"]
                    ),
                    "teacher_pair_count": int(audit["pair_count"]),
                    "maximum_views_per_region": int(
                        audit["maximum_views_per_region"]
                    ),
                }
            )
        records.sort(key=lambda item: item["scene_id"])
        if len({record["scene_id"] for record in records}) != len(records):
            raise ValueError(f"{split} repeats a sampling scene authority")
        splits[split] = records
    return {
        "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
        "preregistration_file_sha256": SPARSE_V2_PREREG_FILE_SHA256,
        "per_scene_region_cap": REGION_CAP_PER_SCENE,
        "per_region_view_cap": VIEW_CAP_PER_REGION,
        "storage": "sparse_coo_pairs_plus_merged_csr_offsets",
        "global_or_cohort_teacher_densification": False,
        "batch_local_gather_only": True,
        "scene_records_by_split": splits,
    }


def _gather_sparse_teacher_rows(
    data: Mapping[str, Any],
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Densify only the requested batch, never the merged cohort teacher."""

    selected = torch.as_tensor(rows).long().cpu().reshape(-1)
    offsets = torch.as_tensor(
        data["official_multiview_siglip2_teacher_pair_row_offsets"]
    ).long().cpu()
    pairs = torch.as_tensor(
        data["official_multiview_siglip2_teacher_pair_descriptors"]
    ).float().cpu()
    row_count = int(offsets.numel()) - 1
    if (
        selected.numel() <= 0
        or bool((selected < 0).any())
        or bool((selected >= row_count).any())
        or offsets.shape != (row_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != pairs.shape[0]
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError("sparse teacher batch gather rows/CSR differ")
    counts = offsets[selected + 1] - offsets[selected]
    maximum = int(counts.max())
    if maximum <= 0 or maximum > VIEW_CAP_PER_REGION:
        raise ValueError("sparse teacher batch gather exceeds frozen view cap")
    output = torch.zeros(selected.numel(), maximum, DESCRIPTOR_DIM)
    mask = torch.zeros(selected.numel(), maximum, dtype=torch.bool)
    for local, row in enumerate(selected.tolist()):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        count = stop - start
        output[local, :count] = pairs[start:stop]
        mask[local, :count] = True
    return output, mask


def _sparse_teacher_view_ids(data: Mapping[str, Any]) -> list[list[str]]:
    offsets = torch.as_tensor(
        data["official_multiview_siglip2_teacher_pair_row_offsets"]
    ).long().cpu()
    pair_ids = data["teacher_pair_view_ids"]
    if not isinstance(pair_ids, list) or int(offsets[-1]) != len(pair_ids):
        raise ValueError("sparse teacher view ID CSR differs")
    return [
        list(pair_ids[int(offsets[row]) : int(offsets[row + 1])])
        for row in range(offsets.numel() - 1)
    ]


def _scene_rows(
    scene_ids: Sequence[str],
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if mask.dtype != torch.bool or mask.shape != (len(scene_ids),):
        raise ValueError("scene row mask differs")
    grouped: dict[str, list[int]] = {}
    for row, scene in enumerate(scene_ids):
        grouped.setdefault(str(scene), [])
        if bool(mask[row]):
            grouped[str(scene)].append(row)
    if any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("every cohort scene requires at least two eligible rows")
    return {
        scene: torch.tensor(rows, dtype=torch.long)
        for scene, rows in sorted(grouped.items())
    }


def validate_training_cohort(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    cohort_authority: Mapping[str, Any],
    cohort_authority_file: Mapping[str, str],
    source_state_manifest: Mapping[str, Any],
    source_state_manifest_file: Mapping[str, str],
    teacher_manifest: Mapping[str, Any],
    teacher_manifest_file: Mapping[str, str],
    benchmark_exclusion_manifest: Mapping[str, Any],
    benchmark_exclusion_manifest_file: Mapping[str, str],
) -> dict[str, Any]:
    """Bind actual scene sets and shard lineage to the loaded authority."""

    authority = validate_cohort_authority_payload(cohort_authority)
    source_manifest = validate_source_state_manifest(source_state_manifest)
    teacher = validate_teacher_manifest(teacher_manifest)
    exclusion = validate_benchmark_exclusion_manifest(
        benchmark_exclusion_manifest
    )
    if (
        not isinstance(cohort_authority_file, Mapping)
        or set(cohort_authority_file) != {"path", "sha256"}
        or not isinstance(cohort_authority_file.get("path"), str)
        or not cohort_authority_file.get("path")
    ):
        raise ValueError("clean cohort authority file record differs")
    authority_file_sha = _require_sha256(
        cohort_authority_file.get("sha256"),
        label="clean cohort authority file record",
    )
    authority_file_path = validate_file_record(
        cohort_authority_file,
        label="clean full-scalar residual cohort authority",
    )
    manifest_records: dict[str, dict[str, str]] = {}
    for label, record in (
        ("source_state_manifest", source_state_manifest_file),
        ("teacher_manifest", teacher_manifest_file),
        ("benchmark_exclusion_manifest", benchmark_exclusion_manifest_file),
    ):
        path = validate_file_record(record, label=label.replace("_", " "))
        manifest_records[label] = {
            "path": str(path),
            "sha256": _require_sha256(record.get("sha256"), label=label),
        }
    train_scenes = sorted(set(str(value) for value in train["scene_ids"]))
    validation_scenes = sorted(
        set(str(value) for value in validation["scene_ids"])
    )
    train_physical_spaces = sorted(
        {canonical_physical_space_id(scene) for scene in train_scenes}
    )
    validation_physical_spaces = sorted(
        {canonical_physical_space_id(scene) for scene in validation_scenes}
    )
    if len(train_scenes) != TRAIN_SCENE_COUNT:
        raise ValueError("source-train cohort must contain exactly 24 scenes")
    if len(validation_scenes) != VALIDATION_SCENE_COUNT:
        raise ValueError("source-validation cohort must contain exactly 8 scenes")
    if set(train_scenes) & set(validation_scenes):
        raise ValueError("source train/validation scenes overlap")
    if len(train_physical_spaces) != TRAIN_SCENE_COUNT:
        raise ValueError(
            "source-train cohort must contain exactly 24 physical spaces"
        )
    if len(validation_physical_spaces) != VALIDATION_SCENE_COUNT:
        raise ValueError(
            "source-validation cohort must contain exactly 8 physical spaces"
        )
    if set(train_physical_spaces) & set(validation_physical_spaces):
        raise ValueError("source train/validation physical spaces overlap")
    if train_scenes != authority["source_train_scene_ids"]:
        raise ValueError("source-train scenes differ from clean cohort authority")
    if validation_scenes != authority["source_validation_scene_ids"]:
        raise ValueError(
            "source-validation scenes differ from clean cohort authority"
        )
    if (
        train_physical_spaces
        != authority["source_train_physical_space_ids"]
        or validation_physical_spaces
        != authority["source_validation_physical_space_ids"]
    ):
        raise ValueError(
            "source cohort physical spaces differ from clean cohort authority"
        )
    exclusion_binding = authority["benchmark_exclusion"]
    if (
        exclusion_binding["manifest_authority_sha256"]
        != exclusion["authority_sha256"]
        or exclusion_binding["manifest_file_sha256"]
        != manifest_records["benchmark_exclusion_manifest"]["sha256"]
    ):
        raise ValueError("cohort/external benchmark exclusion authority differs")
    if (set(train_scenes) | set(validation_scenes)) & set(exclusion["scene_ids"]):
        raise ValueError("clean cohort contains a benchmark exclusion scene")
    if (set(train_physical_spaces) | set(validation_physical_spaces)) & set(
        exclusion["physical_space_ids"]
    ):
        raise ValueError(
            "clean cohort contains a benchmark exclusion physical space"
        )

    all_scene_ids = list(train["scene_ids"]) + list(validation["scene_ids"])
    all_region_ids = list(train["region_row_ids"]) + list(
        validation["region_row_ids"]
    )
    all_teacher_view_ids = _sparse_teacher_view_ids(train) + (
        _sparse_teacher_view_ids(validation)
    )
    if len(set(all_region_ids)) != len(all_region_ids):
        raise ValueError("training cohort repeats a stable region row ID")
    observed_regions = sorted(
        (
            {"region_row_id": row_id, "scene_id": scene_id}
            for row_id, scene_id in zip(all_region_ids, all_scene_ids)
        ),
        key=lambda item: item["region_row_id"],
    )
    if observed_regions != source_manifest["region_records"]:
        raise ValueError("training rows differ from source-state manifest")
    manifest_scenes = [
        record["scene_id"] for record in source_manifest["scene_records"]
    ]
    if manifest_scenes != sorted(set(all_scene_ids)):
        raise ValueError("training scenes differ from source-state manifest")
    manifest_physical_spaces = [
        record["physical_space_id"]
        for record in source_manifest["scene_records"]
    ]
    if manifest_physical_spaces != sorted(
        set(train_physical_spaces) | set(validation_physical_spaces)
    ):
        raise ValueError(
            "training physical spaces differ from source-state manifest"
        )
    observed_teacher = sorted(
        (
            {
                "region_row_id": row_id,
                "scene_id": scene_id,
                "teacher_view_ids": list(view_ids),
            }
            for row_id, scene_id, view_ids in zip(
                all_region_ids, all_scene_ids, all_teacher_view_ids
            )
        ),
        key=lambda item: item["region_row_id"],
    )
    if observed_teacher != teacher["region_view_records"]:
        raise ValueError("training teacher views differ from teacher manifest")
    all_lineages = list(train["lineages"]) + list(validation["lineages"])
    cohort_sha = {
        str(lineage["cohort_authority_sha256"]) for lineage in all_lineages
    }
    cohort_file_sha = {
        str(lineage["cohort_authority_file_sha256"])
        for lineage in all_lineages
    }
    source_manifest_file_sha = {
        str(lineage["source_state_manifest_file_sha256"])
        for lineage in all_lineages
    }
    teacher_manifest_file_sha = {
        str(lineage["teacher_manifest_file_sha256"])
        for lineage in all_lineages
    }
    source_cohort_sha = {
        str(lineage["source_state_cohort_authority_sha256"])
        for lineage in all_lineages
    }
    if len(cohort_sha) != 1:
        raise ValueError("training shards bind different cohort authorities")
    if len(cohort_file_sha) != 1:
        raise ValueError("training shards bind different cohort authority files")
    if len(source_cohort_sha) != 1:
        raise ValueError("training shards bind different source-state cohorts")
    if len(source_manifest_file_sha) != 1:
        raise ValueError("training shards bind different source-state manifests")
    if len(teacher_manifest_file_sha) != 1:
        raise ValueError("training shards bind different teacher manifests")
    if next(iter(cohort_sha)) != authority["authority_sha256"]:
        raise ValueError(
            "training shard cohort content lineage differs from loaded authority"
        )
    if next(iter(cohort_file_sha)) != authority_file_sha:
        raise ValueError(
            "training shard cohort file lineage differs from loaded authority"
        )
    if (
        next(iter(source_cohort_sha)) != source_manifest["authority_sha256"]
        or next(iter(source_manifest_file_sha))
        != manifest_records["source_state_manifest"]["sha256"]
    ):
        raise ValueError("training shard source-state manifest lineage differs")
    teacher_authority_sha = {
        str(lineage["teacher_authority_sha256"]) for lineage in all_lineages
    }
    if (
        len(teacher_authority_sha) != 1
        or next(iter(teacher_authority_sha)) != teacher["authority_sha256"]
        or next(iter(teacher_manifest_file_sha))
        != manifest_records["teacher_manifest"]["sha256"]
    ):
        raise ValueError("training shard teacher manifest lineage differs")
    _scene_rows(train["scene_ids"], train["eligible"])
    _scene_rows(validation["scene_ids"], validation["eligible"])
    return {
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "train_physical_spaces": train_physical_spaces,
        "validation_physical_spaces": validation_physical_spaces,
        "cohort_authority_sha256": next(iter(cohort_sha)),
        "cohort_authority_file": {
            "path": str(authority_file_path),
            "sha256": authority_file_sha,
        },
        "cohort_authority_file_sha256": authority_file_sha,
        "benchmark_exclusion": {
            "source_identifier": exclusion["source_identifier"],
            "source_artifact_sha256": exclusion["source_artifact_sha256"],
            "scene_ids_sha256": exclusion["scene_ids_sha256"],
            "scene_count": len(exclusion["scene_ids"]),
            "physical_space_ids_sha256": exclusion[
                "physical_space_ids_sha256"
            ],
            "physical_space_count": len(exclusion["physical_space_ids"]),
            "authority_sha256": exclusion["authority_sha256"],
            "file": manifest_records["benchmark_exclusion_manifest"],
        },
        "source_state_cohort_authority_sha256": next(iter(source_cohort_sha)),
        "source_state_manifest": {
            "authority_sha256": source_manifest["authority_sha256"],
            "file": manifest_records["source_state_manifest"],
        },
        "teacher_manifest": {
            "authority_sha256": teacher["authority_sha256"],
            "file": manifest_records["teacher_manifest"],
        },
    }


def _teacher_prototype(
    teachers: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    count = mask.sum(dim=1, keepdim=True)
    if bool((count <= 0).any()):
        raise ValueError("teacher prototype requires at least one source view")
    mean = (teachers * mask[..., None]).sum(dim=1) / count
    norm = torch.linalg.vector_norm(mean, dim=-1)
    if bool((norm <= 0).any()) or not bool(torch.isfinite(mean).all()):
        raise ValueError("teacher prototype is non-finite or zero")
    return F.normalize(mean, dim=-1)


def _lower_quantile(values: torch.Tensor, fraction: float) -> float:
    if values.ndim != 1 or values.numel() <= 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("quantile values differ")
    ordered = values.sort().values
    index = int(math.floor(float(fraction) * float(ordered.numel() - 1)))
    return float(ordered[index])


def _off_diagonal_relation(
    descriptors: torch.Tensor,
    teacher_prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if descriptors.shape != teacher_prototypes.shape or descriptors.shape[0] < 2:
        raise ValueError("off-diagonal relation requires aligned rows >= 2")
    student = F.normalize(descriptors.float(), dim=-1)
    teacher = F.normalize(teacher_prototypes.detach().float(), dim=-1)
    student_gram = student @ student.T
    teacher_gram = teacher @ teacher.T
    upper = torch.triu(
        torch.ones_like(student_gram, dtype=torch.bool), diagonal=1
    )
    difference = student_gram[upper] - teacher_gram[upper]
    return difference.abs(), F.smooth_l1_loss(
        student_gram[upper], teacher_gram[upper], reduction="none"
    )


def _active_scene_rows(
    scene_ids: Sequence[str],
    active: torch.Tensor,
    *,
    minimum_rows: int,
) -> dict[str, torch.Tensor]:
    if active.dtype != torch.bool or active.shape != (len(scene_ids),):
        raise ValueError("active scene row mask differs")
    grouped: dict[str, list[int]] = {}
    for row, (scene, use) in enumerate(zip(scene_ids, active.tolist())):
        if use:
            grouped.setdefault(str(scene), []).append(row)
    return {
        scene: torch.tensor(rows, dtype=torch.long)
        for scene, rows in sorted(grouped.items())
        if len(rows) >= minimum_rows
    }


def _scene_coverage(
    scene_ids: Sequence[str],
    active: torch.Tensor,
    *,
    minimum_rows: int,
) -> dict[str, Any]:
    """Report non-vacuous row coverage for every declared source scene."""

    if (
        active.dtype != torch.bool
        or active.shape != (len(scene_ids),)
        or int(minimum_rows) < 1
    ):
        raise ValueError("validation scene coverage inputs differ")
    counts: dict[str, int] = {}
    for scene, use in zip(scene_ids, active.tolist()):
        name = str(scene)
        counts.setdefault(name, 0)
        counts[name] += int(bool(use))
    expected = sorted(counts)
    insufficient = [
        scene for scene in expected if counts[scene] < int(minimum_rows)
    ]
    return {
        "expected_scenes": expected,
        "scene_count": len(expected),
        "minimum_rows_per_scene": int(minimum_rows),
        "per_scene_rows": {scene: counts[scene] for scene in expected},
        "covered_scene_count": len(expected) - len(insufficient),
        "missing_or_insufficient_scenes": insufficient,
        "passed": not insufficient,
    }


def _relation_metrics(
    descriptors: torch.Tensor,
    teacher_prototypes: torch.Tensor,
    scene_ids: Sequence[str],
    active: torch.Tensor,
) -> dict[str, Any]:
    groups = _active_scene_rows(
        scene_ids, active.cpu(), minimum_rows=2
    )
    scene_metrics: dict[str, dict[str, float | int]] = {}
    for scene, rows_cpu in groups.items():
        rows = rows_cpu.to(descriptors.device)
        absolute, smooth = _off_diagonal_relation(
            descriptors[rows], teacher_prototypes[rows]
        )
        scene_metrics[scene] = {
            "relation_mean_absolute_error": float(absolute.mean()),
            "relation_smooth_l1": float(smooth.mean()),
            "relation_fidelity": float(1.0 - 0.5 * absolute.mean()),
            "relation_pair_count": int(absolute.numel()),
        }
    names = (
        "relation_mean_absolute_error",
        "relation_smooth_l1",
        "relation_fidelity",
    )
    if not scene_metrics:
        return {
            "relation_mean_absolute_error": 0.0,
            "relation_smooth_l1": 0.0,
            "relation_fidelity": 1.0,
            "relation_pair_count": 0,
            "relation_scene_count": 0,
            "relation_per_scene": {},
        }
    return {
        **{
            name: sum(float(value[name]) for value in scene_metrics.values())
            / len(scene_metrics)
            for name in names
        },
        "relation_pair_count": sum(
            int(value["relation_pair_count"]) for value in scene_metrics.values()
        ),
        "relation_scene_count": len(scene_metrics),
        "relation_per_scene": scene_metrics,
    }


def _descriptor_scene_metrics(
    descriptors: torch.Tensor,
    data: Mapping[str, Any],
    global_rows: torch.Tensor,
    scene_ids: Sequence[str],
    active: torch.Tensor,
) -> dict[str, Any]:
    groups = _active_scene_rows(
        scene_ids, active.cpu(), minimum_rows=1
    )
    if not groups:
        raise ValueError("descriptor scope contains no active scene rows")
    per_scene: dict[str, dict[str, float | int]] = {}
    for scene, rows_cpu in groups.items():
        means: list[torch.Tensor] = []
        view_count = 0
        for start in range(0, rows_cpu.numel(), 256):
            local_cpu = rows_cpu[start : start + 256]
            teacher_cpu, mask_cpu = _gather_sparse_teacher_rows(
                data, global_rows[local_cpu]
            )
            local = local_cpu.to(descriptors.device)
            teacher = teacher_cpu.to(descriptors.device)
            mask = mask_cpu.to(descriptors.device)
            pair = torch.einsum(
                "bd,bvd->bv",
                F.normalize(descriptors[local].float(), dim=-1),
                F.normalize(teacher.float(), dim=-1),
            )
            means.append(
                ((pair * mask).sum(dim=1) / mask.sum(dim=1)).detach().cpu()
            )
            view_count += int(mask_cpu.sum())
        row_mean = torch.cat(means)
        per_scene[scene] = {
            "mean_all_view_cosine": float(row_mean.mean()),
            "p05_row_mean_all_view_cosine": _lower_quantile(row_mean, 0.05),
            "row_count": int(rows_cpu.numel()),
            "view_count": view_count,
        }
    return {
        "mean_all_view_cosine": sum(
            float(value["mean_all_view_cosine"]) for value in per_scene.values()
        ) / len(per_scene),
        "p05_row_mean_all_view_cosine": sum(
            float(value["p05_row_mean_all_view_cosine"])
            for value in per_scene.values()
        ) / len(per_scene),
        "descriptor_scene_count": len(per_scene),
        "descriptor_per_scene": per_scene,
    }


def _paired_scene_delta_summary(
    base: Mapping[str, Mapping[str, float | int]],
    candidate: Mapping[str, Mapping[str, float | int]],
    metric: str,
) -> dict[str, Any]:
    if set(base) != set(candidate):
        raise ValueError("paired scene metrics differ")
    if not base:
        return {
            "per_scene": {},
            "mean": 0.0,
            "p05": 0.0,
            "worst": 0.0,
            "scene_count": 0,
        }
    values = {
        scene: float(candidate[scene][metric]) - float(base[scene][metric])
        for scene in sorted(base)
    }
    tensor = torch.tensor(list(values.values()), dtype=torch.float64)
    return {
        "per_scene": values,
        "mean": float(tensor.mean()),
        "p05": _lower_quantile(tensor, 0.05),
        "worst": float(tensor.min()),
        "scene_count": len(values),
    }


def _sparse_teacher_prototypes(
    data: Mapping[str, Any],
    global_rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    prototypes: list[torch.Tensor] = []
    rows = torch.as_tensor(global_rows).long().cpu()
    for start in range(0, rows.numel(), 256):
        teacher_cpu, mask_cpu = _gather_sparse_teacher_rows(
            data, rows[start : start + 256]
        )
        prototypes.append(
            _teacher_prototype(
                teacher_cpu.to(device), mask_cpu.to(device)
            )
        )
    return torch.cat(prototypes, dim=0)


@torch.no_grad()
def evaluate(
    model: SurfaceRegionAcceptedV2FullScalarResidualV1,
    data: Mapping[str, Any],
    normalization_authority: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate source rows relative to the immutable accepted-V2 base."""

    frozen = validate_full_scalar_normalization_authority(normalization_authority)
    normalized = apply_full_scalar_normalization(
        data["raw_full_scalar_summary"], data["eligible"], frozen
    )
    eligible = data["eligible"].bool()
    if not bool(eligible.any()):
        raise ValueError("evaluation split has no eligible rows")
    eligible_scene_coverage = _scene_coverage(
        data["scene_ids"], eligible, minimum_rows=2
    )
    in_domain_global = eligible & ~normalized.ood_mask
    in_domain_scene_coverage = _scene_coverage(
        data["scene_ids"], in_domain_global, minimum_rows=2
    )
    eligible_device = eligible.to(device)
    base = data["accepted_v2_e0"].to(device)
    scalars = data["raw_full_scalar_summary"].to(device)
    ood = normalized.ood_mask.to(device)
    candidate = model(base, scalars, ood_mask=ood)
    active_base = base[eligible_device]
    active_candidate = candidate[eligible_device]
    eligible_indices = torch.where(eligible)[0]
    teacher_prototype = _sparse_teacher_prototypes(
        data, eligible_indices, device
    )
    active_scene_ids = [data["scene_ids"][int(row)] for row in eligible_indices]
    all_active = torch.ones(len(active_scene_ids), dtype=torch.bool)

    def scope(active_rows: torch.Tensor) -> dict[str, Any]:
        base_descriptor = _descriptor_scene_metrics(
            active_base,
            data,
            eligible_indices,
            active_scene_ids,
            active_rows,
        )
        candidate_descriptor = _descriptor_scene_metrics(
            active_candidate,
            data,
            eligible_indices,
            active_scene_ids,
            active_rows,
        )
        base_relation = _relation_metrics(
            active_base, teacher_prototype, active_scene_ids, active_rows
        )
        candidate_relation = _relation_metrics(
            active_candidate, teacher_prototype, active_scene_ids, active_rows
        )
        base_metrics = {**base_descriptor, **base_relation}
        candidate_metrics = {**candidate_descriptor, **candidate_relation}
        metrics = (
            "mean_all_view_cosine",
            "p05_row_mean_all_view_cosine",
            "relation_fidelity",
        )
        deltas = {
            name: float(candidate_metrics[name]) - float(base_metrics[name])
            for name in metrics
        }
        paired = {
            "mean_all_view_cosine": _paired_scene_delta_summary(
                base_descriptor["descriptor_per_scene"],
                candidate_descriptor["descriptor_per_scene"],
                "mean_all_view_cosine",
            ),
            "p05_row_mean_all_view_cosine": _paired_scene_delta_summary(
                base_descriptor["descriptor_per_scene"],
                candidate_descriptor["descriptor_per_scene"],
                "p05_row_mean_all_view_cosine",
            ),
            "relation_fidelity": _paired_scene_delta_summary(
                base_relation["relation_per_scene"],
                candidate_relation["relation_per_scene"],
                "relation_fidelity",
            ),
        }
        tolerance = NON_REGRESSION_TOLERANCE
        aggregate_checks = {
            name: deltas[name] >= -tolerance for name in metrics
        }
        paired_checks = {
            f"{name}_{statistic}": paired[name][statistic] >= -tolerance
            for name in metrics
            for statistic in ("p05", "worst")
        }
        return {
            "base": base_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_base": deltas,
            "paired_scene_deltas": paired,
            "non_regression_checks": {
                "aggregate": aggregate_checks,
                "paired_scene": paired_checks,
            },
            "non_regression_passed": all(aggregate_checks.values())
            and all(paired_checks.values()),
            "row_count": int(active_rows.sum()),
            "scene_count": len(set(
                scene for scene, use in zip(active_scene_ids, active_rows.tolist())
                if use
            )),
        }

    all_scope = scope(all_active)
    in_domain_active = (~normalized.ood_mask[eligible]).bool()
    if not bool(in_domain_active.any()):
        in_domain_scope = {
            "non_regression_passed": False,
            "vacuous_fallback_only": True,
            "row_count": 0,
            "scene_count": 0,
        }
        in_domain_sufficient = False
    else:
        in_domain_scope = scope(in_domain_active)
        in_domain_scope["vacuous_fallback_only"] = False
        in_domain_sufficient = bool(in_domain_scene_coverage["passed"])
    ood_device = ood & eligible_device
    ood_bitwise_base = bool(
        torch.equal(candidate[ood_device], base[ood_device])
    )
    return {
        "aggregation": "scene_macro",
        "base": all_scope["base"],
        "candidate": all_scope["candidate"],
        "candidate_minus_base": all_scope["candidate_minus_base"],
        "paired_scene_deltas": all_scope["paired_scene_deltas"],
        "non_regression_checks": {
            "all_eligible": all_scope["non_regression_checks"],
            "in_domain": in_domain_scope.get("non_regression_checks", {}),
            "in_domain_sufficient": in_domain_sufficient,
            "ood_bitwise_accepted_v2_base": ood_bitwise_base,
        },
        "non_regression_passed": (
            all_scope["non_regression_passed"]
            and in_domain_scope["non_regression_passed"]
            and eligible_scene_coverage["passed"]
            and in_domain_sufficient
            and ood_bitwise_base
        ),
        "eligible_scene_coverage": eligible_scene_coverage,
        "in_domain_scene_coverage": in_domain_scene_coverage,
        "in_domain": in_domain_scope,
        "ood": {
            "bitwise_accepted_v2_base": ood_bitwise_base,
            "eligible_rows": int(ood_device.sum()),
        },
        "eligible_rows": int(eligible.sum()),
        "ood_fallback_rows": int((eligible & normalized.ood_mask).sum()),
        "trained_or_residual_eligible_rows": int(
            (eligible & ~normalized.ood_mask).sum()
        ),
        "ineligible_rows": int((~eligible).sum()),
    }


def _state_copy(
    model: SurfaceRegionAcceptedV2FullScalarResidualV1,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _state_sha256(model: SurfaceRegionAcceptedV2FullScalarResidualV1) -> str:
    return surface_region_state_dict_sha256(_state_copy(model))


def _zero_init_parity(
    model: SurfaceRegionAcceptedV2FullScalarResidualV1,
    datasets: Sequence[Mapping[str, Any]],
    normalization_authority: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rows = 0
    with torch.no_grad():
        for data in datasets:
            normalized = apply_full_scalar_normalization(
                data["raw_full_scalar_summary"], data["eligible"], normalization_authority
            )
            base = data["accepted_v2_e0"].to(device)
            output = model(
                base,
                data["raw_full_scalar_summary"].to(device),
                ood_mask=normalized.ood_mask.to(device),
            )
            if not torch.equal(output, base):
                raise RuntimeError("zero-initialized residual is not bitwise base parity")
            rows += int(base.shape[0])
    return {
        "passed": True,
        "bitwise_equal": True,
        "rows_checked": rows,
        "residual_projection_weight_nonzero": int(
            torch.count_nonzero(model.residual_projection.weight)
        ),
        "residual_projection_bias_nonzero": int(
            torch.count_nonzero(model.residual_projection.bias)
        ),
    }


def _training_batches(
    scene_ids: Sequence[str],
    active: torch.Tensor,
    *,
    epoch: int,
) -> list[torch.Tensor]:
    groups = _scene_rows(scene_ids, active.cpu())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 1_000_003 * int(epoch))
    batches: list[torch.Tensor] = []
    for rows in groups.values():
        ordered = rows[torch.randperm(rows.numel(), generator=generator)]
        selected = ordered[: min(BATCH_ROWS, int(ordered.numel()))]
        if selected.numel() < 2:
            raise RuntimeError("relation training batch contains fewer than two rows")
        batches.append(selected)
    return batches


def train_one_epoch(
    model: SurfaceRegionAcceptedV2FullScalarResidualV1,
    optimizer: torch.optim.Optimizer,
    data: Mapping[str, Any],
    normalization_authority: Mapping[str, Any],
    device: torch.device,
    *,
    epoch: int,
) -> dict[str, float | int]:
    """Optimize only eligible, non-OOD source-train rows for one epoch."""

    normalized = apply_full_scalar_normalization(
        data["raw_full_scalar_summary"], data["eligible"], normalization_authority
    )
    active = data["eligible"].bool() & ~normalized.ood_mask
    if int(active.sum()) < 2:
        raise ValueError("source-train has fewer than two trainable rows")
    batches = _training_batches(data["scene_ids"], active, epoch=epoch)
    totals = {"objective": 0.0, "all_view": 0.0, "relation": 0.0}
    total_rows = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for rows_cpu in batches:
        base = data["accepted_v2_e0"][rows_cpu].to(device)
        scalars = data["raw_full_scalar_summary"][rows_cpu].to(device)
        teachers_cpu, teacher_mask_cpu = _gather_sparse_teacher_rows(
            data, rows_cpu
        )
        teachers = teachers_cpu.to(device)
        teacher_mask = teacher_mask_cpu.to(device)
        semantic = model(base, scalars, ood_mask=None)
        prototype = _teacher_prototype(teachers, teacher_mask)
        pair = torch.einsum(
            "bd,bvd->bv",
            F.normalize(semantic.float(), dim=-1),
            F.normalize(teachers.float(), dim=-1),
        )
        row_loss = ((1.0 - pair) * teacher_mask).sum(dim=1) / teacher_mask.sum(
            dim=1
        )
        all_view_loss = row_loss.mean()
        _absolute, relation_values = _off_diagonal_relation(
            semantic, prototype
        )
        relation_loss = relation_values.mean()
        objective = all_view_loss + RELATION_GRAM_WEIGHT * relation_loss
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("full-scalar residual training objective is non-finite")
        (objective / len(batches)).backward()
        size = int(rows_cpu.numel())
        total_rows += size
        totals["objective"] += float(objective.detach())
        totals["all_view"] += float(all_view_loss.detach())
        totals["relation"] += float(relation_loss.detach())
    torch.nn.utils.clip_grad_norm_(
        tuple(model.parameters()), MAX_GRADIENT_NORM, error_if_nonfinite=True
    )
    optimizer.step()
    scene_count = len(batches)
    return {
        "objective": totals["objective"] / scene_count,
        "all_view_cosine_loss": totals["all_view"] / scene_count,
        "relation_gram_smooth_l1_loss": totals["relation"] / scene_count,
        "sampled_trainable_rows": total_rows,
        "trainable_rows": total_rows,
        "available_trainable_rows": int(active.sum()),
        "scene_count": scene_count,
        "scene_weight": 1.0 / scene_count,
        "max_rows_per_scene": BATCH_ROWS,
        "ood_rows_excluded": int((data["eligible"].bool() & normalized.ood_mask).sum()),
        "ineligible_rows_excluded": int((~data["eligible"].bool()).sum()),
    }


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    """Select only a source-validation non-regressing checkpoint."""

    if not history or [int(row.get("epoch", -1)) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("full-scalar residual history must be contiguous from epoch zero")
    eligible = [
        row
        for row in history
        if row.get("validation", {}).get("non_regression_passed") is True
    ]
    if not eligible:
        raise RuntimeError("no source-validation non-regressing checkpoint exists")

    def rank(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
        metrics = row["validation"]["candidate"]
        return (
            float(metrics["mean_all_view_cosine"]),
            float(metrics["p05_row_mean_all_view_cosine"]),
            float(metrics["relation_fidelity"]),
            -int(row["epoch"]),
        )

    return int(max(eligible, key=rank)["epoch"])


def _resolve_shard_arguments(
    paths: Sequence[str],
    digests: Sequence[str],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if len(paths) != len(digests) or not paths:
        raise ValueError(f"{split} shard paths and SHA-256 values must align")
    shards: list[dict[str, Any]] = []
    records: list[dict[str, str]] = []
    for path, digest in zip(paths, digests):
        shard, record = load_training_shard(
            path, expected_sha256=digest, expected_split=split
        )
        shards.append(shard)
        records.append(record)
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError(f"{split} repeats a training shard")
    return shards, records


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train, gate, and write one immutable full-scalar residual."""

    output = Path(args.output).expanduser().resolve()
    normalization_output = output.with_suffix(output.suffix + ".normalization.pt")
    certificate_output = output.with_suffix(output.suffix + ".certificate.json")
    report_output = output.with_suffix(output.suffix + ".json")
    existing = [
        path
        for path in (output, normalization_output, certificate_output, report_output)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "full-scalar residual outputs must be new: "
            + ", ".join(str(path) for path in existing)
        )

    cohort_authority, cohort_authority_file = load_cohort_authority(
        args.cohort_authority,
        expected_sha256=args.expected_cohort_authority_sha256,
    )
    source_state_manifest, source_state_manifest_file = _load_json_manifest(
        args.source_state_manifest,
        expected_sha256=args.expected_source_state_manifest_sha256,
        label="source-state manifest",
        validator=validate_source_state_manifest,
    )
    teacher_manifest, teacher_manifest_file = _load_json_manifest(
        args.teacher_manifest,
        expected_sha256=args.expected_teacher_manifest_sha256,
        label="teacher manifest",
        validator=validate_teacher_manifest,
    )
    exclusion_manifest, exclusion_manifest_file = _load_json_manifest(
        args.benchmark_exclusion_manifest,
        expected_sha256=args.expected_benchmark_exclusion_manifest_sha256,
        label="benchmark exclusion manifest",
        validator=validate_benchmark_exclusion_manifest,
    )
    train_shards, train_records = _resolve_shard_arguments(
        args.train_shard,
        args.expected_train_shard_sha256,
        split="source_train",
    )
    validation_shards, validation_records = _resolve_shard_arguments(
        args.validation_shard,
        args.expected_validation_shard_sha256,
        split="source_validation",
    )
    train_data = _pad_and_merge_shards(train_shards, split="source_train")
    validation_data = _pad_and_merge_shards(
        validation_shards, split="source_validation"
    )
    cohort = validate_training_cohort(
        train_data,
        validation_data,
        cohort_authority,
        cohort_authority_file,
        source_state_manifest,
        source_state_manifest_file,
        teacher_manifest,
        teacher_manifest_file,
        exclusion_manifest,
        exclusion_manifest_file,
    )
    normalization = build_full_scalar_normalization_authority(
        train_data["raw_full_scalar_summary"],
        train_data["eligible"],
        source_state_cohort_sha256=(
            cohort["source_state_cohort_authority_sha256"]
        ),
    )
    validate_full_scalar_normalization_authority(normalization)
    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=DESCRIPTOR_DIM,
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=MAX_ANGLE_RADIANS,
        max_alpha=MAX_ALPHA,
    ).to(device)
    zero_parity = _zero_init_parity(
        model, (train_data, validation_data), normalization, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history: list[dict[str, Any]] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    epoch_zero_validation = evaluate(
        model, validation_data, normalization, device
    )
    history.append(
        {
            "epoch": 0,
            "training": None,
            "validation": epoch_zero_validation,
            "model_state_dict_sha256": _state_sha256(model),
        }
    )
    states[0] = _state_copy(model)
    best_epoch = select_best_epoch(history)
    epochs_without_improvement = 0
    for epoch in range(1, EPOCHS + 1):
        training_metrics = train_one_epoch(
            model,
            optimizer,
            train_data,
            normalization,
            device,
            epoch=epoch,
        )
        validation_metrics = evaluate(
            model, validation_data, normalization, device
        )
        record = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
            "model_state_dict_sha256": _state_sha256(model),
        }
        history.append(record)
        states[epoch] = _state_copy(model)
        selected = select_best_epoch(history)
        if selected != best_epoch:
            best_epoch = selected
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(json.dumps(record, sort_keys=True), flush=True)
        if epochs_without_improvement >= PATIENCE:
            break
    selected_epoch = select_best_epoch(history)
    model.load_state_dict(states[selected_epoch], strict=True)
    selected_validation = evaluate(
        model, validation_data, normalization, device
    )
    if selected_validation != history[selected_epoch]["validation"]:
        raise RuntimeError("restored full-scalar checkpoint metrics differ")
    if not selected_validation["non_regression_passed"]:
        raise RuntimeError("selected full-scalar checkpoint regresses accepted V2")
    model.cpu()

    normalization_path = write_torch_noclobber(
        normalization_output, normalization
    )
    normalization_sha256 = sha256_file(normalization_path)
    certificate = build_training_certificate_payload(
        training_contract=training_contract(),
        model_authority={
            "class": type(model).__name__,
            "architecture": model.architecture(),
            "architecture_sha256": canonical_json_sha256(model.architecture()),
            "state_dict_sha256": _state_sha256(model),
        },
        normalization_authority=file_record(normalization_path),
        cohort_authority={
            "file": cohort["cohort_authority_file"],
            "authority_sha256": cohort["cohort_authority_sha256"],
        },
        source_state_manifest=cohort["source_state_manifest"],
        teacher_manifest=cohort["teacher_manifest"],
        benchmark_exclusion_manifest={
            "file": cohort["benchmark_exclusion"]["file"],
            "authority_sha256": cohort["benchmark_exclusion"][
                "authority_sha256"
            ],
        },
        input_shards={
            "source_train": train_records,
            "source_validation": validation_records,
        },
        sampling_authority=_sampling_authority_for_certificate(
            train_data, validation_data
        ),
        selected_epoch=selected_epoch,
        selected_validation=selected_validation,
    )
    certificate_path = write_frozen_json(certificate_output, certificate)
    certificate_sha256 = sha256_file(certificate_path)
    checkpoint_path, checkpoint_sha256 = (
        write_surface_region_full_scalar_residual_checkpoint(
            output,
            model,
            normalization_authority=normalization,
            normalization_authority_sha256=normalization_sha256,
            source_state_cohort_authority_sha256=(
                cohort["source_state_cohort_authority_sha256"]
            ),
            training_certificate=certificate,
            training_certificate_sha256=certificate_sha256,
        )
    )
    report = {
        "schema_version": 2,
        "artifact_type": TRAINING_ARTIFACT_TYPE,
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "normalization_authority": file_record(normalization_path),
        "training_certificate": file_record(certificate_path),
        "normalization_source": {
            "split": "source_train",
            "validation_rows_used": False,
            "eligible_rows": int(train_data["eligible"].sum()),
            "source_count": int(normalization["source_count"]),
        },
        "selected_epoch": selected_epoch,
        "selected_validation": selected_validation,
        "zero_initialization_parity": zero_parity,
        "history": history,
        "cohort": cohort,
        "input_shards": {
            "source_train": train_records,
            "source_validation": validation_records,
        },
        "sampling_authority": certificate["sampling_authority"],
        "lineage": {
            "accepted_v2_authority": _accepted_v2_authority(),
            "source_state_cohort_authority_sha256": (
                cohort["source_state_cohort_authority_sha256"]
            ),
            "source_state_manifest": cohort["source_state_manifest"],
            "cohort_authority_sha256": cohort["cohort_authority_sha256"],
            "cohort_authority_file_sha256": (
                cohort["cohort_authority_file_sha256"]
            ),
            "teacher_manifest": cohort["teacher_manifest"],
        },
        "frozen": {
            "accepted_v2": True,
            "official_multiview_siglip2_teacher": True,
            "base_or_teacher_parameters_in_optimizer": False,
            "trainable_model_class": type(model).__name__,
        },
        "source_access": {
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "target_heldout_opened": False,
            "text_queries_opened": False,
            "per_scene_hyperparameters": False,
        },
    }
    write_frozen_json(report_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", action="append", required=True)
    parser.add_argument(
        "--expected-train-shard-sha256", action="append", required=True
    )
    parser.add_argument("--validation-shard", action="append", required=True)
    parser.add_argument(
        "--expected-validation-shard-sha256", action="append", required=True
    )
    parser.add_argument("--cohort-authority", required=True)
    parser.add_argument("--expected-cohort-authority-sha256", required=True)
    parser.add_argument("--source-state-manifest", required=True)
    parser.add_argument("--expected-source-state-manifest-sha256", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--expected-teacher-manifest-sha256", required=True)
    parser.add_argument("--benchmark-exclusion-manifest", required=True)
    parser.add_argument(
        "--expected-benchmark-exclusion-manifest-sha256", required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
