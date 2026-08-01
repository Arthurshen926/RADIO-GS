#!/usr/bin/env python3
"""Materialize paired Surface readout/teacher descriptors on CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import (
    canonical_json_sha256,
    row_identity_sha256,
    tensor_sha256,
)
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_text_response_descriptor_pair"
_DISTILL_RUN_ARTIFACT_TYPE = "surface_region_text_response_distill_run"
_MATERIALIZER_RELATIVE_PATH = (
    "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
)
_AUTHORITY_DISTILL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "candidate",
        "surface_promotion",
        "train_caches",
        "validation_caches",
        "fit_text_bank",
        "radio_checkpoint",
        "calibration_manifest",
        "outputs",
        "training_contract",
        "thermal_safety_contract",
        "implementation_sources",
        "authority_status",
        "calibration_audit",
        "initial_gpu_preflight",
        "gpu_identity",
        "runtime_closure",
        "authority_contract",
        "training_command_contract",
    }
)
_AUTHORITY_DISTILL_OUTPUT_FIELDS = frozenset(
    {
        "seed",
        "checkpoint",
        "report",
        "training_log",
        "audit_report",
        "guard_command",
        "guard_telemetry",
        "guard_receipt",
        "kernel_journal",
        "gpu_preflight",
        "gpu_postflight",
        "terminal",
    }
)
_QUERY_FREE_FLAGS = (
    "uses_benchmark_scenes",
    "uses_benchmark_test_vocabulary",
    "annotations_opened",
    "labels_opened",
    "instances_opened",
    "masks_opened",
    "text_opened",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path) -> Mapping:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _canonical_absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be a non-empty absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} is missing or cannot be resolved") from error
    if path != resolved:
        raise ValueError(f"{label} must be a canonical non-symlink path")
    return resolved


def _validate_authority_materializer_source(
    manifest: Mapping[str, Any],
) -> None:
    authority = manifest.get("authority_contract")
    if not isinstance(authority, Mapping):
        raise ValueError("authority distill run lacks its authority contract")
    snapshot_root = _canonical_absolute_path(
        authority.get("source_snapshot_root"),
        label="authority source_snapshot_root",
    )
    if not snapshot_root.is_dir():
        raise ValueError("authority source_snapshot_root is not a directory")
    producer_source = _canonical_absolute_path(
        str(snapshot_root / _MATERIALIZER_RELATIVE_PATH),
        label="authority producer materializer",
    )
    try:
        producer_source.relative_to(snapshot_root)
    except ValueError as error:
        raise ValueError("authority producer materializer escapes its snapshot") from error
    if not producer_source.is_file():
        raise ValueError("authority producer materializer is not a regular file")

    implementations = manifest.get("implementation_sources")
    producer_sha = (
        implementations.get(_MATERIALIZER_RELATIVE_PATH)
        if isinstance(implementations, Mapping)
        else None
    )
    if (
        not isinstance(producer_sha, str)
        or len(producer_sha) != 64
        or any(character not in "0123456789abcdef" for character in producer_sha)
        or producer_sha != _sha256_file(producer_source)
    ):
        raise ValueError(
            "authority distill run binds another producer materializer implementation"
        )


def _assert_query_free(metadata: Mapping, source: str) -> None:
    for key in _QUERY_FREE_FLAGS:
        if metadata.get(key) is not False:
            raise ValueError(f"{source} must explicitly certify {key}=false")


def _legacy_region_id(record: Mapping) -> str:
    """Build a context-independent ID for pre-region_id schema-v3 caches."""

    identity = {
        "scene": record.get("scene"),
        "seed": record.get("seed"),
        "physical_radius_m": record.get("physical_radius_m"),
        "teacher_views": record.get("teacher_views"),
        "teacher_target_sha256": record.get("teacher_target_sha256"),
        "teacher_support_sha256": record.get("teacher_support_sha256"),
    }
    if not str(identity["scene"] or "") or identity["seed"] is None:
        raise ValueError("legacy region record lacks stable scene/seed identity")
    return "legacy-" + canonical_json_sha256(identity)


def _load_validation_caches(
    paths: list[Path],
    *,
    include_summary_tokens: bool = False,
) -> tuple[dict, dict]:
    if not paths:
        raise ValueError("at least one validation cache is required")
    tensor_keys = (
        "radio_features",
        "geometry",
        "token_mask",
        "reliability",
        "official_crop_summaries",
        "teacher_mask",
        "anchor_index",
    )
    if include_summary_tokens:
        # Weight-interpolation diagnostics need token fidelity as well as the
        # descriptor pair materialized by this module.  Keeping this opt-in
        # preserves the descriptor artifact schema while still loading each
        # immutable validation cache only once.
        tensor_keys = (*tensor_keys, "official_summary_tokens")
    parts = {key: [] for key in tensor_keys}
    scene_ids: list[str] = []
    region_ids: list[str] = []
    cache_records = []
    contract_hashes = set()
    radio_hashes = set()
    split_hashes = set()
    contract_specs = []
    teacher_specs = []
    excluded_spaces: list[str] | None = None
    exclusion_files: list[dict[str, str]] | None = None
    for raw_path in sorted(Path(value).resolve() for value in paths):
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _torch_load(path)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path} lacks cache metadata")
        if metadata.get("schema_version") != 3 or metadata.get("split_role") != "validation":
            raise ValueError(f"{path} is not a schema-v3 validation cache")
        _assert_query_free(metadata, str(path))
        if (
            metadata.get("physical_space_disjoint") is not True
            or metadata.get("complete_scene_regions") is not True
            or metadata.get("failed_scenes")
            or metadata.get("teacher_regions_saturated") != 0
        ):
            raise ValueError(f"{path} violates the complete disjoint validation contract")
        records = metadata.get("region_records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path} lacks row-aligned region_records")
        row_count = len(records)
        for key in tensor_keys:
            tensor = torch.as_tensor(payload.get(key))
            if tensor.device.type != "cpu" or tensor.shape[0] != row_count:
                raise ValueError(f"{path} has a misaligned CPU tensor {key}")
            parts[key].append(tensor)
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"{path} contains a non-object region record")
            scene = str(record.get("scene", ""))
            region = str(record.get("region_id", "")) or _legacy_region_id(record)
            if not scene:
                raise ValueError(f"{path} contains an empty scene identity")
            scene_ids.append(scene)
            region_ids.append(region)
        local_scenes = sorted({str(record.get("scene", "")) for record in records})
        local_counts = {
            scene: sum(str(record.get("scene", "")) == scene for record in records)
            for scene in local_scenes
        }
        if (
            metadata.get("scene_names") != local_scenes
            or metadata.get("scene_region_counts") != local_counts
        ):
            raise ValueError(f"{path} has inconsistent scene metadata")
        contract_hash = str(metadata.get("region_contract_sha256", ""))
        radio_hash = str(metadata.get("radio_checkpoint_sha256", ""))
        split_hash = str(metadata.get("split_file_sha256", ""))
        if any(len(value) != 64 for value in (contract_hash, radio_hash, split_hash)):
            raise ValueError(f"{path} lacks contract/checkpoint hashes")
        contract_hashes.add(contract_hash)
        radio_hashes.add(radio_hash)
        split_hashes.add(split_hash)
        contract_specs.append(metadata.get("region_contract"))
        teacher_specs.append(
            {
                "semantics": metadata.get("teacher_region_semantics"),
                "contract": metadata.get("teacher_region_contract"),
                "contract_sha256": metadata.get("teacher_region_contract_sha256"),
                "target_source": metadata.get("teacher_target_source"),
                "target_protocol_sha256": metadata.get(
                    "teacher_target_protocol_sha256"
                ),
            }
        )
        current_spaces = list(metadata.get("excluded_physical_spaces", []))
        current_files = list(metadata.get("exclusion_files", []))
        if excluded_spaces is None:
            excluded_spaces = current_spaces
            exclusion_files = current_files
        elif excluded_spaces != current_spaces or exclusion_files != current_files:
            raise ValueError("validation cache exclusion contracts differ")
        cache_records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": row_count,
                "split_file_sha256": split_hash,
                "region_contract_sha256": contract_hash,
                "radio_checkpoint_sha256": radio_hash,
                "teacher_target_protocol_sha256": str(
                    metadata.get("teacher_target_protocol_sha256", "")
                ),
            }
        )
    if (
        len(contract_hashes) != 1
        or len(radio_hashes) != 1
        or len(split_hashes) != 1
        or any(value != contract_specs[0] for value in contract_specs[1:])
        or any(value != teacher_specs[0] for value in teacher_specs[1:])
    ):
        raise ValueError(
            "validation caches do not share one split/region/teacher/RADIO contract"
        )
    if len(set(zip(scene_ids, region_ids))) != len(scene_ids):
        raise ValueError("validation caches contain duplicate scene/region identities")
    merged = {key: torch.cat(value, dim=0) for key, value in parts.items()}
    bound_paths = [record["path"] for record in cache_records]
    cache_bindings = [
        {"path": record["path"], "sha256": record["sha256"]}
        for record in cache_records
    ]
    checkpoint_validation = {
        "scenes": sorted(set(scene_ids)),
        "split_hashes": sorted(split_hashes),
        "cache_paths": bound_paths,
        "region_contract_sha256": next(iter(contract_hashes)),
        "region_contract": contract_specs[0],
        "teacher_region": teacher_specs[0],
        "radio_checkpoint_sha256": next(iter(radio_hashes)),
        "excluded_physical_spaces": excluded_spaces or [],
        "exclusion_files": exclusion_files or [],
        "physical_space_disjoint": True,
        "cache_bindings": cache_bindings,
    }
    return merged, {
        "scene_ids": scene_ids,
        "region_ids": region_ids,
        "caches": cache_records,
        "cache_bindings": cache_bindings,
        "cache_paths": bound_paths,
        "split_hashes": sorted(split_hashes),
        "scenes": sorted(set(scene_ids)),
        "region_contract_sha256": next(iter(contract_hashes)),
        "region_contract": contract_specs[0],
        "radio_checkpoint_sha256": next(iter(radio_hashes)),
        "teacher_region": teacher_specs[0],
        "excluded_physical_spaces": excluded_spaces or [],
        "exclusion_files": exclusion_files or [],
        "checkpoint_validation": checkpoint_validation,
    }


def _teacher_descriptor(
    official_crop_summaries: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    descriptors = F.normalize(official_crop_summaries.float(), dim=-1, eps=1e-8)
    mask = teacher_mask.bool()
    if descriptors.ndim != 3 or mask.shape != descriptors.shape[:2]:
        raise ValueError("official teacher descriptors/mask are misaligned")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every region requires at least one teacher descriptor")
    weights = mask.float() / mask.sum(dim=1, keepdim=True)
    return F.normalize(
        (descriptors * weights[..., None]).sum(dim=1),
        dim=-1,
        eps=1e-8,
    )


def _validate_distill_run_manifest(
    binding: object,
    *,
    checkpoint_path: Path,
    report_path: Path,
    seed: int,
    cache_meta: Mapping[str, Any],
    radio_path: Path,
    radio_sha256: str,
) -> dict[str, str]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "candidate",
    }:
        raise ValueError("readout checkpoint lacks an exact distill run-manifest binding")
    bound_manifest_path = Path(str(binding.get("path", "")))
    manifest_path = bound_manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("bound distill run manifest is missing")
    manifest_sha = _sha256_file(manifest_path)
    if binding.get("sha256") != manifest_sha:
        raise ValueError("distill run-manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or type(schema_version) is not int
        or schema_version not in {1, 2}
        or manifest.get("artifact_type") != _DISTILL_RUN_ARTIFACT_TYPE
    ):
        raise ValueError("invalid distill run-manifest schema")
    if schema_version == 2 and set(manifest) != _AUTHORITY_DISTILL_MANIFEST_FIELDS:
        raise ValueError("authority distill run-manifest fields differ")
    if schema_version == 2 and (
        not bound_manifest_path.is_absolute() or bound_manifest_path != manifest_path
    ):
        raise ValueError(
            "authority distill run-manifest path must be canonical and non-symlinked"
        )
    if manifest.get("candidate") != binding.get("candidate"):
        raise ValueError("distill checkpoint/run candidate binding differs")
    if manifest.get("validation_caches") != cache_meta["cache_bindings"]:
        raise ValueError("distill run manifest binds different validation caches")
    if manifest.get("radio_checkpoint") != {
        "path": str(radio_path),
        "sha256": radio_sha256,
    }:
        raise ValueError("distill run manifest binds a different RADIO checkpoint")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("distill run manifest lacks its seed output index")
    expected_output = {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
    }
    if schema_version == 1:
        rows = [
            value
            for value in outputs
            if isinstance(value, Mapping) and value.get("seed") == seed
        ]
        if rows != [expected_output]:
            raise ValueError("distill run manifest binds another checkpoint/report")
    else:
        if (
            len(outputs) != 3
            or any(
                not isinstance(row, Mapping)
                or set(row) != _AUTHORITY_DISTILL_OUTPUT_FIELDS
                or type(row.get("seed")) is not int
                for row in outputs
            )
        ):
            raise ValueError("authority distill output index fields differ")
        by_seed = {row["seed"]: row for row in outputs}
        if set(by_seed) != {0, 1, 2}:
            raise ValueError("authority distill output seeds differ")
        for row in outputs:
            for field in _AUTHORITY_DISTILL_OUTPUT_FIELDS - {"seed"}:
                value = row.get(field)
                if not isinstance(value, str) or not value or not Path(value).is_absolute():
                    raise ValueError(
                        f"authority distill output {field} must be an absolute path"
                    )
                if str(Path(value)) != value:
                    raise ValueError(
                        f"authority distill output {field} must be a canonical path"
                    )
        row = by_seed[seed]
        if any(row.get(key) != value for key, value in expected_output.items()):
            raise ValueError("distill run manifest binds another checkpoint/report")
    implementations = manifest.get("implementation_sources")
    if schema_version == 1:
        source = Path(__file__).resolve()
        if (
            not isinstance(implementations, Mapping)
            or implementations.get(_MATERIALIZER_RELATIVE_PATH)
            != _sha256_file(source)
        ):
            raise ValueError(
                "distill run manifest binds another materializer implementation"
            )
    else:
        _validate_authority_materializer_source(manifest)
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "candidate": str(binding["candidate"]),
    }


def _validate_checkpoint_report(
    report_path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: Mapping[str, Any],
    cache_meta: Mapping[str, Any],
    run_manifest: Mapping[str, str],
) -> dict[str, str]:
    if not report_path.is_file():
        raise FileNotFoundError("readout checkpoint report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("readout checkpoint report must contain an object")
    provenance = checkpoint["provenance"]
    distillation = provenance.get("text_response_distillation")
    if not isinstance(distillation, Mapping):
        raise ValueError("readout checkpoint lacks text-response provenance")
    required = {
        "output",
        "checkpoint_sha256",
        "architecture",
        "best_epoch",
        "best_selection_score",
        "untrained_baseline",
        "selection_score_delta",
        "validation",
        "response_lambda",
        "calibration_manifest",
        "calibration_manifest_sha256",
        "fit_text_bank_sha256",
        "fit_query_count",
        "distill_run_manifest",
        "distill_run_manifest_sha256",
        "validation_caches",
        "train_scenes",
        "validation_scenes",
        "scene_overlap",
    }
    if set(report) != required:
        raise ValueError("readout checkpoint report fields differ from the fixed schema")
    baseline = checkpoint.get("untrained_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("readout checkpoint lacks its untrained baseline")
    baseline_score = 0.5 * (
        float(baseline["mean_descriptor_cosine"])
        + float(baseline["all_view_descriptor_cosine"])
    )
    best_score = float(checkpoint.get("best_selection_score"))
    expected = {
        "output": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": checkpoint.get("architecture"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_selection_score": best_score,
        "untrained_baseline": baseline,
        "selection_score_delta": best_score - baseline_score,
        "response_lambda": distillation.get("response_lambda"),
        "calibration_manifest": distillation.get("calibration_manifest"),
        "calibration_manifest_sha256": distillation.get(
            "calibration_manifest_sha256"
        ),
        "fit_text_bank_sha256": distillation.get("fit_text_bank", {}).get(
            "artifact_sha256"
        ),
        "fit_query_count": distillation.get("fit_text_bank", {}).get(
            "query_count"
        ),
        "distill_run_manifest": run_manifest["path"],
        "distill_run_manifest_sha256": run_manifest["sha256"],
        "validation_caches": cache_meta["cache_bindings"],
        "train_scenes": len(provenance["train"]["scenes"]),
        "validation_scenes": len(cache_meta["scenes"]),
        "scene_overlap": [],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"readout checkpoint report {key} binding differs")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or set(validation) != {
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    }:
        raise ValueError("readout checkpoint report validation metrics differ")
    validation_score = 0.5 * (
        float(validation["mean_descriptor_cosine"])
        + float(validation["all_view_descriptor_cosine"])
    )
    if not math.isclose(validation_score, best_score, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("readout report validation does not reproduce the best score")
    return {"path": str(report_path), "sha256": _sha256_file(report_path)}


def _validate_legacy_report(
    report_path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: Mapping[str, Any],
    cache_meta: Mapping[str, Any],
) -> dict[str, str]:
    if not report_path.is_file():
        raise FileNotFoundError("legacy readout checkpoint report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "output",
        "checkpoint_sha256",
        "architecture",
        "best_epoch",
        "best_selection_score",
        "untrained_baseline",
        "selection_score_delta",
        "validation",
        "train_scenes",
        "validation_scenes",
        "scene_overlap",
    }
    if not isinstance(report, Mapping) or set(report) != required:
        raise ValueError("legacy readout report fields differ from the fixed schema")
    baseline = checkpoint.get("untrained_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("legacy readout checkpoint lacks its baseline")
    baseline_score = 0.5 * (
        float(baseline["mean_descriptor_cosine"])
        + float(baseline["all_view_descriptor_cosine"])
    )
    best_score = float(checkpoint.get("best_selection_score"))
    expected = {
        "output": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": checkpoint.get("architecture"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_selection_score": best_score,
        "untrained_baseline": baseline,
        "selection_score_delta": best_score - baseline_score,
        "train_scenes": len(checkpoint["provenance"]["train"]["scenes"]),
        "validation_scenes": len(cache_meta["scenes"]),
        "scene_overlap": [],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"legacy readout report {key} binding differs")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or set(validation) != {
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    }:
        raise ValueError("legacy readout report validation metrics differ")
    validation_score = 0.5 * (
        float(validation["mean_descriptor_cosine"])
        + float(validation["all_view_descriptor_cosine"])
    )
    if not math.isclose(validation_score, best_score, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("legacy readout validation does not reproduce its best score")
    return {"path": str(report_path), "sha256": _sha256_file(report_path)}


def _validate_legacy_promotion_bundle(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    report_path: Path,
    seed: int,
    cache_meta: Mapping[str, Any],
) -> dict[str, str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("legacy readout requires a query-free promotion bundle")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("schema_version") != 1
        or bundle.get("artifact_type")
        != "surface_region_query_free_three_seed_bundle"
        or bundle.get("status")
        != "query_free_three_seed_bundle_frozen_benchmark_gate_closed"
        or bundle.get("seed_selection_policy")
        != "all_required_seeds_no_single_seed_selection"
        or bundle.get("required_seeds") != [0, 1, 2]
    ):
        raise ValueError("invalid legacy query-free promotion bundle")
    candidate = str(bundle.get("selected_candidate", ""))
    if not candidate:
        raise ValueError("legacy promotion bundle lacks a selected candidate")
    gate = bundle.get("benchmark_gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("status") != "closed_not_evaluated"
        or gate.get("main_result_eligible") is not False
    ):
        raise ValueError("legacy promotion bundle has an open downstream gate")

    completion_path = path.parent / "query_free_promotion.complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError("legacy promotion completion is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        not isinstance(completion, Mapping)
        or completion.get("artifact_type")
        != "surface_region_query_free_promotion_completion"
        or completion.get("promotion_manifest") != str(path)
        or completion.get("promotion_manifest_sha256") != _sha256_file(path)
        or completion.get("selected_candidate") != candidate
        or completion.get("required_seeds") != [0, 1, 2]
        or completion.get("main_result_eligible") is not False
    ):
        raise ValueError("legacy promotion completion does not bind its bundle")

    bindings = bundle.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("legacy promotion bundle lacks artifact bindings")
    for label in (
        "finalizer",
        "run_manifest",
        "cache_pairing",
        "query_free_screen",
        "screen_completion",
    ):
        record = bindings.get(label)
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"legacy promotion bundle lacks {label} binding")
        bound = Path(str(record["path"])).resolve()
        if not bound.is_file() or record["sha256"] != _sha256_file(bound):
            raise ValueError(f"legacy promotion bundle {label} SHA256 mismatch")

    readouts = bindings.get("all_compared_readouts")
    if not isinstance(readouts, list):
        raise ValueError("legacy promotion bundle lacks compared readouts")
    matches = [
        value
        for value in readouts
        if isinstance(value, Mapping)
        and value.get("candidate") == candidate
        and value.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError("legacy promotion bundle lacks the selected candidate/seed")
    readout = matches[0]
    if (
        Path(str(readout.get("checkpoint", ""))).resolve() != checkpoint_path
        or readout.get("checkpoint_sha256") != checkpoint_sha256
        or Path(str(readout.get("sidecar", ""))).resolve() != report_path
        or readout.get("sidecar_sha256") != _sha256_file(report_path)
    ):
        raise ValueError("legacy promotion bundle readout binding differs")
    selected_rows = bundle.get("selected_readouts")
    compared_selected = [
        value
        for value in readouts
        if isinstance(value, Mapping) and value.get("candidate") == candidate
    ]
    if (
        not isinstance(selected_rows, list)
        or selected_rows != compared_selected
        or len(selected_rows) != 3
        or {value.get("seed") for value in selected_rows} != {0, 1, 2}
        or not any(
            value.get("seed") == seed
            and value.get("checkpoint_sha256") == checkpoint_sha256
            for value in selected_rows
        )
    ):
        raise ValueError("legacy readout is not in the selected three-seed bundle")

    cache_rows = bindings.get("caches")
    if not isinstance(cache_rows, list):
        raise ValueError("legacy promotion bundle lacks cache bindings")
    selected_validation = [
        value
        for value in cache_rows
        if isinstance(value, Mapping)
        and value.get("candidate") == candidate
        and value.get("role") == "validation"
    ]
    selected_validation.sort(key=lambda value: int(value.get("shard", -1)))
    expected_caches = [
        {"path": value.get("path"), "sha256": value.get("sha256")}
        for value in selected_validation
    ]
    if expected_caches != cache_meta["cache_bindings"]:
        raise ValueError("legacy promotion bundle binds different validation caches")
    for value in selected_validation:
        sidecar = Path(str(value.get("sidecar", ""))).resolve()
        if (
            not sidecar.is_file()
            or value.get("sidecar_sha256") != _sha256_file(sidecar)
        ):
            raise ValueError("legacy promotion cache sidecar SHA256 mismatch")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "completion": str(completion_path),
        "completion_sha256": _sha256_file(completion_path),
        "candidate": candidate,
    }


def _validate_attention_postcache_binding(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    report_path: Path,
    seed: int,
    cache_meta: Mapping[str, Any],
) -> dict[str, str]:
    from radio_gs.scripts import surface_text_response_distill_authority as authority

    path = Path(path).resolve()
    screen = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(screen, Mapping)
        or screen.get("artifact_type")
        != "surface_c1024_attention_pooling_postcache_continuation"
        or screen.get("selected_variant") != "joint_attention_v1"
        or screen.get("selection_status") != "joint_attention_retained"
        or screen.get("promotion_gate_passed") is not False
        or screen.get("benchmark_queries_opened") is not False
        or screen.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("invalid attention-postcache readout binding screen")
    pairing_path = Path(
        str(screen.get("cache_pairing_report", {}).get("path", ""))
    ).resolve()
    if (
        not pairing_path.is_file()
        or screen.get("cache_pairing_report", {}).get("sha256")
        != _sha256_file(pairing_path)
    ):
        raise ValueError("attention-postcache cache-pairing binding differs")
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    rows = pairing.get("rows")
    if not isinstance(rows, list):
        raise ValueError("attention-postcache cache pairing lacks rows")
    train = [
        dict(row["c1024"])
        for row in rows
        if isinstance(row, Mapping) and row.get("role") == "train"
    ]
    validation = [
        dict(row["c1024"])
        for row in rows
        if isinstance(row, Mapping) and row.get("role") == "validation"
    ]
    train.sort(key=lambda record: record["path"])
    validation.sort(key=lambda record: record["path"])
    binding = authority._surface_binding(
        surface_root=path.parent,
        candidate="context_c1024_geometric",
        train=train,
        validation=validation,
    )
    if binding.get("binding_mode") != authority.ATTENTION_BINDING_MODE:
        raise ValueError("attention-postcache authority mode differs")
    variants = screen.get("variants")
    joint = variants.get("joint_attention_v1") if isinstance(variants, Mapping) else None
    seeds = joint.get("seeds") if isinstance(joint, Mapping) else None
    matches = [
        row
        for row in seeds or []
        if isinstance(row, Mapping) and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError("attention-postcache screen lacks the selected joint seed")
    checkpoint_record = matches[0].get("checkpoint")
    if (
        not isinstance(checkpoint_record, Mapping)
        or Path(str(checkpoint_record.get("path", ""))).resolve() != checkpoint_path
        or checkpoint_record.get("sha256") != checkpoint_sha256
        or not report_path.is_file()
        or _sha256_file(report_path)
        != _sha256_file(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"))
    ):
        raise ValueError("attention-postcache selected readout binding differs")
    if validation != cache_meta["cache_bindings"]:
        raise ValueError("attention-postcache screen binds different validation caches")
    completion_path = path.parent / "screen.complete"
    if not completion_path.is_file():
        raise FileNotFoundError("attention-postcache screen completion is missing")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "completion": str(completion_path),
        "completion_sha256": _sha256_file(completion_path),
        "candidate": "context_c1024_geometric",
    }


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict:
    if str(args.device).lower() != "cpu":
        raise ValueError("this materializer is CPU-only; --device must be cpu")
    cache_paths = [Path(value) for value in args.validation_cache]
    data, cache_meta = _load_validation_caches(cache_paths)
    checkpoint_path = Path(args.readout_checkpoint).resolve()
    radio_path = Path(args.radio_checkpoint).resolve()
    if not checkpoint_path.is_file() or not radio_path.is_file():
        raise FileNotFoundError("readout/RADIO checkpoint is missing")
    checkpoint_sha = _sha256_file(checkpoint_path)
    radio_sha = _sha256_file(radio_path)
    if radio_sha != cache_meta["radio_checkpoint_sha256"]:
        raise ValueError("RADIO checkpoint does not match validation cache provenance")

    model, checkpoint = SurfaceRegionSummaryReadoutV2.from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    provenance = checkpoint.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("readout checkpoint lacks provenance")
    if provenance.get("uses_benchmark_scenes") is not False:
        raise ValueError("readout checkpoint used benchmark scenes")
    if provenance.get("uses_benchmark_test_vocabulary") is not False:
        raise ValueError("readout checkpoint used benchmark test vocabulary")
    if (
        provenance.get("scene_disjoint") is not True
        or provenance.get("custom_text_projection") is not False
    ):
        raise ValueError("readout checkpoint is not a frozen scene-disjoint readout")
    checkpoint_validation = provenance.get("validation")
    if not isinstance(checkpoint_validation, Mapping):
        raise ValueError("readout checkpoint lacks validation-cache provenance")
    distillation = provenance.get("text_response_distillation")
    is_distilled = isinstance(distillation, Mapping)
    legacy_binding_raw = getattr(args, "readout_binding_manifest", None)
    expected_validation = dict(cache_meta["checkpoint_validation"])
    if not is_distilled:
        expected_validation.pop("cache_bindings")
    if checkpoint_validation != expected_validation:
        raise ValueError(
            "provided validation caches differ from checkpoint provenance "
            "(path/SHA/scene/split/teacher/exclusion/RADIO contract)"
        )
    checkpoint_train = provenance.get("train")
    if not isinstance(checkpoint_train, Mapping):
        raise ValueError("readout checkpoint lacks training-cache provenance")
    if set(checkpoint_train.get("scenes", [])) & set(cache_meta["scenes"]):
        raise ValueError("readout checkpoint train/validation scenes overlap")
    if provenance.get("region_contract_sha256") != cache_meta["region_contract_sha256"]:
        raise ValueError("readout/cache SurfaceRegion contracts differ")
    if provenance.get("region_contract") != cache_meta["region_contract"]:
        raise ValueError("readout/cache SurfaceRegion contract payloads differ")
    training_config = checkpoint.get("training_config", {})
    seed = training_config.get("seed")
    if seed not in {0, 1, 2}:
        raise ValueError("readout checkpoint lacks one of the frozen seeds 0/1/2")
    if provenance.get("random_seed_contract") != {
        "seed": seed,
        "model_initialization": True,
        "data_order": True,
        "canonical_noise": True,
    }:
        raise ValueError("readout checkpoint seed provenance differs")
    architecture = checkpoint.get("architecture", {})
    if (
        architecture.get("name") != "surface_region_summary_readout_v2"
        or architecture.get("contract_sha256")
        != cache_meta["region_contract_sha256"]
    ):
        raise ValueError("only SurfaceRegionSummaryReadoutV2 checkpoints are supported")
    report_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if is_distilled:
        if legacy_binding_raw:
            raise ValueError("distilled checkpoints cannot use the legacy bundle path")
        run_manifest = _validate_distill_run_manifest(
            provenance.get("distill_run_manifest"),
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            seed=int(seed),
            cache_meta=cache_meta,
            radio_path=radio_path,
            radio_sha256=radio_sha,
        )
        report_binding = _validate_checkpoint_report(
            report_path,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            checkpoint=checkpoint,
            cache_meta=cache_meta,
            run_manifest=run_manifest,
        )
        authority = {
            "type": "embedded_distill_run_manifest",
            "path": run_manifest["path"],
            "sha256": run_manifest["sha256"],
            "candidate": run_manifest["candidate"],
        }
    else:
        if not legacy_binding_raw:
            raise ValueError(
                "legacy readout requires --readout-binding-manifest with the "
                "query-free promotion bundle"
            )
        binding_path = Path(legacy_binding_raw)
        binding_payload = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            isinstance(binding_payload, Mapping)
            and binding_payload.get("artifact_type")
            == "surface_c1024_attention_pooling_postcache_continuation"
        ):
            legacy_bundle = _validate_attention_postcache_binding(
                binding_path,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                report_path=report_path,
                seed=int(seed),
                cache_meta=cache_meta,
            )
            authority_type = "attention_postcache_screen"
        else:
            legacy_bundle = _validate_legacy_promotion_bundle(
                binding_path,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                report_path=report_path,
                seed=int(seed),
                cache_meta=cache_meta,
            )
            authority_type = "query_free_promotion_bundle"
        report_binding = _validate_legacy_report(
            report_path,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            checkpoint=checkpoint,
            cache_meta=cache_meta,
        )
        authority = {
            "type": authority_type,
            **legacy_bundle,
        }

    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).cpu().eval()
    model = model.cpu().eval()
    model.requires_grad_(False)
    head.requires_grad_(False)
    students = []
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(data["radio_features"]), batch_size):
        stop = min(start + batch_size, len(data["radio_features"]))
        predicted_token = model(
            data["radio_features"][start:stop],
            data["geometry"][start:stop],
            anchor_index=data["anchor_index"][start:stop],
            token_mask=data["token_mask"][start:stop],
            reliability=data["reliability"][start:stop],
        )
        descriptor = head(predicted_token[:, None])[:, 0]
        students.append(F.normalize(descriptor.float(), dim=-1, eps=1e-8).cpu())
    student = torch.cat(students, dim=0).contiguous()
    teacher = _teacher_descriptor(
        data["official_crop_summaries"],
        data["teacher_mask"],
    ).contiguous()
    if student.shape != teacher.shape:
        raise ValueError(
            f"student/teacher descriptor shape mismatch: {student.shape} vs {teacher.shape}"
        )

    descriptor_rows_sha = row_identity_sha256(
        cache_meta["scene_ids"], cache_meta["region_ids"]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "method_id": str(args.method_id),
        "seed": int(seed),
        "split_role": "validation",
        "student_descriptors": student,
        "teacher_descriptors": teacher,
        "scene_ids": cache_meta["scene_ids"],
        "region_ids": cache_meta["region_ids"],
        "student_descriptors_sha256": tensor_sha256(student),
        "teacher_descriptors_sha256": tensor_sha256(teacher),
        "descriptor_rows_sha256": descriptor_rows_sha,
        "descriptor_space": {
            "name": "official_siglip2_g_summary",
            "dimension": int(student.shape[1]),
            "normalization": "l2",
            "official_summary_head": "c-radio_v4 _heads.siglip2-g",
        },
        "provenance": {
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "annotations_opened": False,
            "labels_opened": False,
            "instances_opened": False,
            "masks_opened": False,
            "text_opened": False,
            "device": "cpu",
            "readout_checkpoint": str(checkpoint_path),
            "readout_checkpoint_sha256": checkpoint_sha,
            "readout_report": report_binding["path"],
            "readout_report_sha256": report_binding["sha256"],
            "readout_binding_authority": authority,
            "radio_checkpoint": str(radio_path),
            "radio_checkpoint_sha256": radio_sha,
            "region_contract_sha256": cache_meta["region_contract_sha256"],
            "validation_split_sha256": cache_meta["split_hashes"][0],
            "validation_scenes": cache_meta["scenes"],
            "teacher_region": cache_meta["teacher_region"],
            "validation_caches": cache_meta["caches"],
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"descriptor artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output),
        "sha256": _sha256_file(output),
        "method_id": str(args.method_id),
        "seed": int(seed),
        "regions": int(student.shape[0]),
        "descriptor_dimension": int(student.shape[1]),
        "scenes": len(set(cache_meta["scene_ids"])),
        "descriptor_rows_sha256": descriptor_rows_sha,
        "student_descriptors_sha256": payload["student_descriptors_sha256"],
        "teacher_descriptors_sha256": payload["teacher_descriptors_sha256"],
        "device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", action="append", required=True)
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument(
        "--readout-binding-manifest",
        help=(
            "Required only for a legacy query-free baseline checkpoint; must "
            "be its frozen query_free_promotion_bundle.json. Distilled "
            "checkpoints must use their embedded run-manifest binding."
        ),
    )
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not str(args.method_id).strip():
        raise ValueError("method_id cannot be empty")
    print(json.dumps(materialize(args), indent=2))


if __name__ == "__main__":
    main()
