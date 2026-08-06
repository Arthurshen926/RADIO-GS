"""Immutable promotion certificate for a trained full-scalar residual.

The certificate is deliberately written before the checkpoint.  It binds the
selected model state (not the checkpoint file), every source authority, and
the scene-macro validation gate.  The checkpoint can therefore bind the
certificate file SHA without creating a hash cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    REGION_CAP_PER_SCENE,
    SAMPLING_CONTRACT_SHA256,
    SPARSE_V2_PREREG_FILE_SHA256,
    VIEW_CAP_PER_REGION,
    validate_selection_audit,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
)


SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA = (
    "radio_gs.surface_region_full_scalar_training_certificate.v2"
)
SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_VALIDATION_SCENE_COUNT = 8
_MINIMUM_VALIDATION_ROWS_PER_SCENE = 2
_EXPECTED_TRAINING_SHARD_CONTRACT_SHA256 = (
    "2fde4a68900647dff38e54695b345684f73a1de5a77ffd84f2a762c0b7ab8e43"
)
_EXPECTED_TRAINING_CONTRACT_SHA256 = (
    "0be37bfd7508c9d89039394afba2e39a0f9b704983ce54387657a382531b7f9a"
)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _file_record(value: object, *, label: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256"}
        or not isinstance(value.get("path"), str)
        or not str(value.get("path"))
    ):
        raise ValueError(f"{label} file record differs")
    return {
        "path": str(value["path"]),
        "sha256": _require_sha256(value["sha256"], label=label),
    }


def _manifest_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "file",
        "authority_sha256",
    }:
        raise ValueError(f"{label} manifest record differs")
    return {
        "file": _file_record(value["file"], label=label),
        "authority_sha256": _require_sha256(
            value["authority_sha256"], label=f"{label} authority"
        ),
    }


def training_certificate_contract() -> dict[str, Any]:
    return {
        "schema": SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA,
        "schema_version": (
            SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA_VERSION
        ),
        "write_order": "normalization_then_certificate_then_checkpoint_then_report",
        "model_binding": "architecture_and_state_dict_sha256",
        "selection_binding": (
            "selected_source_validation_scene_macro_paired_non_regression_gate"
        ),
        "selection_coverage": {
            "source_validation_scene_count": _SOURCE_VALIDATION_SCENE_COUNT,
            "minimum_eligible_rows_per_scene": (
                _MINIMUM_VALIDATION_ROWS_PER_SCENE
            ),
            "minimum_in_domain_rows_per_scene": (
                _MINIMUM_VALIDATION_ROWS_PER_SCENE
            ),
            "vacuous_in_domain_fallback_gate_allowed": False,
            "validator_recomputes_coverage_from_per_scene_row_counts": True,
        },
        "source_binding": (
            "cohort_source_state_teacher_exclusion_and_input_shard_file_sha256"
        ),
        "sampling_binding": {
            "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
            "preregistration_file_sha256": SPARSE_V2_PREREG_FILE_SHA256,
            "per_scene_region_cap": REGION_CAP_PER_SCENE,
            "per_region_view_cap": VIEW_CAP_PER_REGION,
            "per_scene_statistics": True,
            "storage": "sparse_coo_pairs_plus_merged_csr_offsets",
            "global_or_cohort_teacher_densification": False,
            "batch_local_gather_only": True,
        },
        "content_sha256": (
            "canonical_json_sha256_of_all_fields_except_content_sha256"
        ),
        "query_independent": True,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
    }


SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_CONTRACT_SHA256 = (
    canonical_json_sha256(training_certificate_contract())
)


def training_certificate_content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("content_sha256", None)
    return canonical_json_sha256(content)


def _validate_scene_coverage(
    value: object,
    *,
    expected_scene_count: int,
    label: str,
) -> dict[str, Any]:
    """Recompute a selected-validation coverage claim from its row counts."""

    if not isinstance(value, Mapping):
        raise ValueError(f"full-scalar certificate {label} coverage differs")
    coverage = dict(value)
    required = {
        "expected_scenes",
        "scene_count",
        "minimum_rows_per_scene",
        "per_scene_rows",
        "covered_scene_count",
        "missing_or_insufficient_scenes",
        "passed",
    }
    if set(coverage) != required:
        raise ValueError(f"full-scalar certificate {label} coverage differs")
    scenes = coverage.get("expected_scenes")
    counts = coverage.get("per_scene_rows")
    minimum = coverage.get("minimum_rows_per_scene")
    if (
        not isinstance(scenes, list)
        or any(not isinstance(scene, str) or not scene for scene in scenes)
        or scenes != sorted(set(scenes))
        or len(scenes) != int(expected_scene_count)
        or coverage.get("scene_count") != len(scenes)
        or minimum != _MINIMUM_VALIDATION_ROWS_PER_SCENE
        or not isinstance(counts, Mapping)
        or set(counts) != set(scenes)
        or any(not isinstance(counts[scene], int) or counts[scene] < 0 for scene in scenes)
    ):
        raise ValueError(f"full-scalar certificate {label} coverage differs")
    insufficient = [scene for scene in scenes if counts[scene] < minimum]
    if (
        coverage.get("covered_scene_count") != len(scenes) - len(insufficient)
        or coverage.get("missing_or_insufficient_scenes") != insufficient
        or coverage.get("passed") is not (not insufficient)
    ):
        raise ValueError(f"full-scalar certificate {label} coverage differs")
    return {
        **coverage,
        "per_scene_rows": {scene: int(counts[scene]) for scene in scenes},
    }


def _validate_sampling_authority(
    value: object,
    *,
    expected_train_scene_count: int,
    expected_validation_scene_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "sampling_contract_sha256",
        "preregistration_file_sha256",
        "per_scene_region_cap",
        "per_region_view_cap",
        "storage",
        "global_or_cohort_teacher_densification",
        "batch_local_gather_only",
        "scene_records_by_split",
    }:
        raise ValueError("full-scalar certificate sampling authority differs")
    authority = dict(value)
    if (
        authority.get("sampling_contract_sha256") != SAMPLING_CONTRACT_SHA256
        or authority.get("preregistration_file_sha256")
        != SPARSE_V2_PREREG_FILE_SHA256
        or authority.get("per_scene_region_cap") != REGION_CAP_PER_SCENE
        or authority.get("per_region_view_cap") != VIEW_CAP_PER_REGION
        or authority.get("storage")
        != "sparse_coo_pairs_plus_merged_csr_offsets"
        or authority.get("global_or_cohort_teacher_densification") is not False
        or authority.get("batch_local_gather_only") is not True
    ):
        raise ValueError("full-scalar certificate sampling authority differs")
    split_records = authority.get("scene_records_by_split")
    if not isinstance(split_records, Mapping) or set(split_records) != {
        "source_train",
        "source_validation",
    }:
        raise ValueError("full-scalar certificate sampling scenes differ")
    frozen: dict[str, list[dict[str, Any]]] = {}
    for split, expected_count in (
        ("source_train", expected_train_scene_count),
        ("source_validation", expected_validation_scene_count),
    ):
        records = split_records.get(split)
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError("full-scalar certificate sampling scenes differ")
        scenes: list[str] = []
        frozen_records: list[dict[str, Any]] = []
        for record in records:
            required = {
                "scene_id",
                "canonical_candidate_region_count",
                "exact_overlap_candidate_count",
                "teacher_visible_candidate_count",
                "selected_region_count",
                "selected_count_by_scale",
                "teacher_pair_count",
                "maximum_views_per_region",
            }
            if not isinstance(record, Mapping) or set(record) != required:
                raise ValueError("full-scalar certificate sampling record differs")
            scene = record.get("scene_id")
            selected = int(record.get("selected_region_count", -1))
            pair_count = int(record.get("teacher_pair_count", -1))
            maximum_views = int(record.get("maximum_views_per_region", -1))
            accepted_audit = {
                "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
                "canonical_candidate_region_count": record.get(
                    "canonical_candidate_region_count"
                ),
                "exact_overlap_candidate_count": record.get(
                    "exact_overlap_candidate_count"
                ),
                "teacher_visible_candidate_count": record.get(
                    "teacher_visible_candidate_count"
                ),
                "selected_region_count": selected,
                "selected_count_by_scale": record.get("selected_count_by_scale"),
            }
            if (
                not isinstance(scene, str)
                or not scene
                or selected <= 0
                or pair_count < selected
                or pair_count > selected * VIEW_CAP_PER_REGION
                or not 1 <= maximum_views <= VIEW_CAP_PER_REGION
            ):
                raise ValueError("full-scalar certificate sampling record differs")
            validate_selection_audit(accepted_audit, selected_count=selected)
            scenes.append(scene)
            frozen_records.append(dict(record))
        if scenes != sorted(set(scenes)):
            raise ValueError("full-scalar certificate sampling scenes differ")
        frozen[split] = frozen_records
    return {**authority, "scene_records_by_split": frozen}


def validate_training_certificate_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("full-scalar training certificate must be a mapping")
    certificate = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "training_contract",
        "training_contract_sha256",
        "model_authority",
        "normalization_authority",
        "cohort_authority",
        "source_state_manifest",
        "teacher_manifest",
        "benchmark_exclusion_manifest",
        "input_shards",
        "sampling_authority",
        "selected_epoch",
        "selected_validation",
        "selection",
        "source_access",
        "content_sha256",
    }
    if set(certificate) != required:
        raise ValueError("full-scalar training certificate fields differ")
    if (
        certificate.get("schema")
        != SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA
        or certificate.get("schema_version")
        != SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA_VERSION
        or certificate.get("contract") != training_certificate_contract()
        or certificate.get("contract_sha256")
        != SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_CONTRACT_SHA256
    ):
        raise ValueError("full-scalar training certificate contract differs")
    training_contract = certificate.get("training_contract")
    training_contract_sha = _require_sha256(
        certificate.get("training_contract_sha256"), label="training contract"
    )
    if (
        not isinstance(training_contract, Mapping)
        or canonical_json_sha256(dict(training_contract)) != training_contract_sha
        or training_contract_sha != _EXPECTED_TRAINING_CONTRACT_SHA256
        or training_contract.get("schema_version") != 2
        or training_contract.get("input_shard_contract_sha256")
        != _EXPECTED_TRAINING_SHARD_CONTRACT_SHA256
    ):
        raise ValueError("full-scalar certificate training contract differs")
    expected_sampling_policy = {
        "contract_sha256": SAMPLING_CONTRACT_SHA256,
        "preregistration_file_sha256": SPARSE_V2_PREREG_FILE_SHA256,
        "per_scene_region_cap": REGION_CAP_PER_SCENE,
        "per_region_view_cap": VIEW_CAP_PER_REGION,
        "storage": "sparse_coo_pairs_plus_merged_csr_offsets",
        "batch_local_pair_gather_only": True,
        "global_or_cohort_teacher_densification": False,
    }
    if training_contract.get("sampling") != expected_sampling_policy:
        raise ValueError("full-scalar certificate training sampling policy differs")

    model = certificate.get("model_authority")
    if not isinstance(model, Mapping) or set(model) != {
        "class",
        "architecture",
        "architecture_sha256",
        "state_dict_sha256",
    }:
        raise ValueError("full-scalar certificate model authority differs")
    architecture = model.get("architecture")
    if (
        model.get("class") != "SurfaceRegionAcceptedV2FullScalarResidualV1"
        or not isinstance(architecture, Mapping)
        or canonical_json_sha256(dict(architecture))
        != _require_sha256(
            model.get("architecture_sha256"), label="model architecture"
        )
    ):
        raise ValueError("full-scalar certificate model architecture differs")
    _require_sha256(model.get("state_dict_sha256"), label="model state dictionary")

    normalization = _file_record(
        certificate.get("normalization_authority"),
        label="normalization authority",
    )
    cohort = certificate.get("cohort_authority")
    if not isinstance(cohort, Mapping) or set(cohort) != {
        "file",
        "authority_sha256",
    }:
        raise ValueError("full-scalar certificate cohort authority differs")
    cohort_record = _manifest_record(cohort, label="cohort authority")
    source_manifest = _manifest_record(
        certificate.get("source_state_manifest"), label="source-state"
    )
    teacher_manifest = _manifest_record(
        certificate.get("teacher_manifest"), label="teacher"
    )
    exclusion_manifest = _manifest_record(
        certificate.get("benchmark_exclusion_manifest"),
        label="benchmark exclusion",
    )

    input_shards = certificate.get("input_shards")
    if not isinstance(input_shards, Mapping) or set(input_shards) != {
        "source_train",
        "source_validation",
    }:
        raise ValueError("full-scalar certificate input shards differ")
    frozen_shards: dict[str, list[dict[str, str]]] = {}
    for split in ("source_train", "source_validation"):
        records = input_shards.get(split)
        if not isinstance(records, list) or not records:
            raise ValueError("full-scalar certificate input shard records differ")
        frozen_shards[split] = [
            _file_record(record, label=f"{split} shard") for record in records
        ]
        if len({item["sha256"] for item in frozen_shards[split]}) != len(records):
            raise ValueError("full-scalar certificate repeats an input shard")

    cohort_policy = training_contract.get("cohort")
    expected_train_scenes = (
        cohort_policy.get("source_train_scene_count")
        if isinstance(cohort_policy, Mapping)
        else None
    )
    expected_validation_scenes = (
        cohort_policy.get("source_validation_scene_count")
        if isinstance(cohort_policy, Mapping)
        else None
    )
    if not isinstance(expected_train_scenes, int) or not isinstance(
        expected_validation_scenes, int
    ):
        raise ValueError("full-scalar certificate cohort sampling policy differs")
    sampling_authority = _validate_sampling_authority(
        certificate.get("sampling_authority"),
        expected_train_scene_count=expected_train_scenes,
        expected_validation_scene_count=expected_validation_scenes,
    )
    if (
        sampling_authority["sampling_contract_sha256"]
        != training_contract["sampling"]["contract_sha256"]
        or sampling_authority["preregistration_file_sha256"]
        != training_contract["sampling"]["preregistration_file_sha256"]
        or sampling_authority["per_scene_region_cap"]
        != training_contract["sampling"]["per_scene_region_cap"]
        or sampling_authority["per_region_view_cap"]
        != training_contract["sampling"]["per_region_view_cap"]
        or sampling_authority["storage"] != training_contract["sampling"]["storage"]
        or sampling_authority["batch_local_gather_only"]
        != training_contract["sampling"]["batch_local_pair_gather_only"]
        or sampling_authority["global_or_cohort_teacher_densification"]
        != training_contract["sampling"]["global_or_cohort_teacher_densification"]
    ):
        raise ValueError("full-scalar certificate sampling bindings differ")

    selected_epoch = certificate.get("selected_epoch")
    selected_validation = certificate.get("selected_validation")
    selection = certificate.get("selection")
    selection_policy = training_contract.get("selection")
    if not isinstance(selected_epoch, int) or selected_epoch < 0:
        raise ValueError("full-scalar certificate selected epoch differs")
    if (
        not isinstance(selected_validation, Mapping)
        or selected_validation.get("aggregation") != "scene_macro"
        or selected_validation.get("non_regression_passed") is not True
        or not isinstance(expected_validation_scenes, int)
        or expected_validation_scenes != _SOURCE_VALIDATION_SCENE_COUNT
        or not isinstance(selection_policy, Mapping)
        or selection_policy.get("minimum_eligible_rows_per_scene")
        != _MINIMUM_VALIDATION_ROWS_PER_SCENE
        or selection_policy.get("minimum_in_domain_rows_per_scene")
        != _MINIMUM_VALIDATION_ROWS_PER_SCENE
        or selection_policy.get("vacuous_in_domain_fallback_gate_allowed")
        is not False
    ):
        raise ValueError("full-scalar certificate validation gate differs")
    eligible_coverage = _validate_scene_coverage(
        selected_validation.get("eligible_scene_coverage"),
        expected_scene_count=expected_validation_scenes,
        label="eligible-scene",
    )
    in_domain_coverage = _validate_scene_coverage(
        selected_validation.get("in_domain_scene_coverage"),
        expected_scene_count=expected_validation_scenes,
        label="in-domain-scene",
    )
    in_domain = selected_validation.get("in_domain")
    if (
        eligible_coverage["passed"] is not True
        or in_domain_coverage["passed"] is not True
        or not isinstance(in_domain, Mapping)
        or in_domain.get("non_regression_passed") is not True
        or in_domain.get("vacuous_fallback_only") is not False
        or int(in_domain.get("row_count", -1))
        != sum(in_domain_coverage["per_scene_rows"].values())
        or int(selected_validation.get("eligible_rows", -1))
        != sum(eligible_coverage["per_scene_rows"].values())
        or int(selected_validation.get("trained_or_residual_eligible_rows", -1))
        != sum(in_domain_coverage["per_scene_rows"].values())
    ):
        raise ValueError("full-scalar certificate validation coverage differs")
    if selection != {
        "split": "source_validation",
        "epoch_zero_was_candidate": True,
        "scene_macro_gate_passed": True,
        "paired_scene_non_regression_passed": True,
        "in_domain_non_regression_passed": True,
        "every_validation_scene_in_domain_covered": True,
        "vacuous_in_domain_fallback_gate_forbidden": True,
    }:
        raise ValueError("full-scalar certificate selection assertion differs")
    source_access = certificate.get("source_access")
    expected_access = {
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "per_scene_hyperparameters": False,
    }
    if source_access != expected_access:
        raise ValueError("full-scalar certificate source access differs")
    expected_content = _require_sha256(
        certificate.get("content_sha256"), label="training certificate content"
    )
    if training_certificate_content_sha256(certificate) != expected_content:
        raise ValueError("full-scalar training certificate content SHA-256 differs")
    return {
        **certificate,
        "normalization_authority": normalization,
        "cohort_authority": cohort_record,
        "source_state_manifest": source_manifest,
        "teacher_manifest": teacher_manifest,
        "benchmark_exclusion_manifest": exclusion_manifest,
        "input_shards": frozen_shards,
        "sampling_authority": sampling_authority,
        "training_contract": dict(training_contract),
        "model_authority": {**dict(model), "architecture": dict(architecture)},
        "selected_validation": dict(selected_validation),
    }


def build_training_certificate_payload(
    *,
    training_contract: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    normalization_authority: Mapping[str, str],
    cohort_authority: Mapping[str, Any],
    source_state_manifest: Mapping[str, Any],
    teacher_manifest: Mapping[str, Any],
    benchmark_exclusion_manifest: Mapping[str, Any],
    input_shards: Mapping[str, Sequence[Mapping[str, str]]],
    sampling_authority: Mapping[str, Any],
    selected_epoch: int,
    selected_validation: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA,
        "schema_version": (
            SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA_VERSION
        ),
        "contract": training_certificate_contract(),
        "contract_sha256": (
            SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_CONTRACT_SHA256
        ),
        "training_contract": dict(training_contract),
        "training_contract_sha256": canonical_json_sha256(dict(training_contract)),
        "model_authority": dict(model_authority),
        "normalization_authority": dict(normalization_authority),
        "cohort_authority": dict(cohort_authority),
        "source_state_manifest": dict(source_state_manifest),
        "teacher_manifest": dict(teacher_manifest),
        "benchmark_exclusion_manifest": dict(benchmark_exclusion_manifest),
        "input_shards": {
            split: [dict(record) for record in input_shards[split]]
            for split in ("source_train", "source_validation")
        },
        "sampling_authority": dict(sampling_authority),
        "selected_epoch": int(selected_epoch),
        "selected_validation": dict(selected_validation),
        "selection": {
            "split": "source_validation",
            "epoch_zero_was_candidate": True,
            "scene_macro_gate_passed": True,
            "paired_scene_non_regression_passed": True,
            "in_domain_non_regression_passed": True,
            "every_validation_scene_in_domain_covered": True,
            "vacuous_in_domain_fallback_gate_forbidden": True,
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
    payload["content_sha256"] = training_certificate_content_sha256(payload)
    return validate_training_certificate_payload(payload)


def load_training_certificate(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = _require_sha256(expected_sha256, label="training certificate")
    value, observed, source = load_json_object(
        path,
        expected_sha256=expected,
        label="full-scalar training certificate",
    )
    certificate = validate_training_certificate_payload(value)
    return certificate, {"path": str(source), "sha256": observed}


__all__ = [
    "SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA",
    "SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_SCHEMA_VERSION",
    "SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_CONTRACT_SHA256",
    "training_certificate_contract",
    "training_certificate_content_sha256",
    "validate_training_certificate_payload",
    "build_training_certificate_payload",
    "load_training_certificate",
]
