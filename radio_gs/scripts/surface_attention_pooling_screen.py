#!/usr/bin/env python3
"""Fail-closed authority for the isolated Surface c1024 attention screen.

This module intentionally keeps the scientific comparison smaller than the
historical capacity runner: one geometric c1024 cache is consumed by both
``joint_attention_v1`` and ``core_context_separate_attention_v1``.  It also
proves that the two legacy train control shards are the previously audited
bytes before they can be used as fixed-teacher replay sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

import torch

from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SEPARATE_CONTEXT_POOLING,
)
from radio_gs.scripts.surface_region_run_guard import (
    _telemetry_rows,
    audit_attempt_inventory,
    build_runtime_closure,
    canonical_json_sha256,
    summarize_canary_telemetry,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SCREEN_NAME = "surface-c1024-attention-pooling-v1"
TRAIN_SHARDS = 4
VALIDATION_SHARDS = 2
SEEDS = (0, 1, 2)
CONTROL_NAME = "control_c256_geometric"
CACHE_NAME = "context_c1024_geometric"
VARIANTS = (JOINT_CONTEXT_POOLING, SEPARATE_CONTEXT_POOLING)
DESCRIPTOR_COMPONENTS = (
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
)
ALL_VALIDATION_COMPONENTS = (
    "summary_token_cosine",
    *DESCRIPTOR_COMPONENTS,
)

# These bytes are the two completed, independently reopened p8 control shards
# documented in feature_recovery_continuation_20260731.md.  Relocating the
# files is allowed; changing any byte is not.
LEGACY_MANIFEST_SHA256 = (
    "77d162b286355c5ce1d369790f299c819397bfbd67302efdde4f72ae63409c2a"
)
LEGACY_CONTROL_SHA256 = {
    0: "02cfa45af46cf8274c17ccde28e8953fb4b659527652525340703a241cad22bc",
    1: "dfd2c0857aca2495b867da393b4cda298554670e313334392bdc030876d84460",
}
LEGACY_SIDECAR_SHA256 = {
    0: "37f19d5778b59e9efb32af97d0c9eeec3daf7b2ebf9f88c9886886a9aaf33e0c",
    1: "4b3dff0f57d13b1a387b13246cff85ad0dbb1de9520d176db4d567112b1047d2",
}
LEGACY_BUILDER_SHA256 = (
    "182408a3f16dcd8a50b0190157c885de81d35a87b59a1f5262b7ed6d81ab8d63"
)
TEACHER_REPLAY_AUTHORITY_ARTIFACT_TYPE = (
    "surface-region-teacher-replay-authority-v1"
)

EXPECTED_CACHE_CONTRACT = {
    "frames_per_scene": 8,
    "regions_per_scene": 12,
    "region_radii_m": [0.25, 0.45, 0.70],
    "maximum_tokens": 256,
    "teacher_region_candidate_limit": 4096,
    "path_cost_mode": "appearance_boundary_geometric",
    "path_affinity_floor": 1e-4,
    "token_subsampling": "core_context_radial_stratified_v1",
    "core_token_fraction": 0.60,
    "teacher_views": 3,
    "seed": 0,
}
THERMAL_KEYS = {
    "physical_gpu",
    "gpu_uuid",
    "maximum_temperature_c",
    "maximum_start_temperature_c",
    "maximum_power_limit_w",
    "poll_seconds",
    "soft_pause_temperature_c",
    "soft_resume_temperature_c",
    "peer_gpu",
    "peer_pause_temperature_c",
    "peer_resume_temperature_c",
    "peer_quiet_seconds_before_launch",
    "peer_max_power_w",
    "peer_max_memory_mib",
    "peer_max_utilization_pct",
    "peer_activity_action",
    "owner_pid_namespace_mode",
    "radio_pacing_seconds_per_image",
    "canary_max_temp_c",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _split_scenes(path: Path, shard: int, shard_count: int) -> list[str]:
    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[shard::shard_count][:100]


def _expected_shape(rows: int) -> dict[str, tuple[int, ...]]:
    return {
        "radio_features": (rows, 256, 1280),
        "geometry": (rows, 256, 14),
        "token_mask": (rows, 256),
        "reliability": (rows, 256, 1),
        "official_summary_tokens": (rows, 3, 1280),
        "official_crop_summaries": (rows, 3, 1536),
        "teacher_mask": (rows, 3),
        "anchor_index": (rows,),
    }


def _teacher_identity(record: Mapping) -> dict:
    keys = (
        "region_id",
        "scene",
        "seed",
        "physical_radius_m",
        "teacher_views",
        "teacher_medoid",
        "teacher_region_tokens",
        "teacher_support_sha256",
        "teacher_region_saturated",
        "teacher_target_sha256",
    )
    return {key: record.get(key) for key in keys}


def _validate_tensor_payload(payload: Mapping, rows: int, *, label: str) -> None:
    for key, shape in _expected_shape(rows).items():
        value = payload.get(key)
        _require(torch.is_tensor(value), f"{label} lacks tensor {key}")
        _require(tuple(value.shape) == shape, f"{label} {key} shape differs")
        if value.is_floating_point() or value.is_complex():
            _require(bool(torch.isfinite(value).all()), f"{label} {key} is nonfinite")
    _require(bool(payload["token_mask"].any(dim=1).all()), f"{label} has empty tokens")
    _require(bool(payload["teacher_mask"].any(dim=1).all()), f"{label} has empty teacher")


def _validate_region_contract(
    metadata: Mapping,
    *,
    candidate_limit: int,
    context_ratio: float,
) -> None:
    contract = metadata.get("region_contract")
    _require(isinstance(contract, Mapping), "cache lacks region contract")
    observed_limit = int(
        contract.get("token_candidate_limit", contract.get("maximum_tokens", -1))
    )
    _require(
        contract.get("version") == "surface-region-contract-v2"
        and [float(value) for value in contract.get("radii_m", [])]
        == [0.25, 0.45, 0.70]
        and math.isclose(float(contract.get("context_ratio", -1)), context_ratio)
        and int(contract.get("maximum_tokens", -1)) == 256
        and int(contract.get("minimum_tokens", -1)) == 24
        and observed_limit == candidate_limit
        and contract.get("reliability_semantics")
        == "geometric_mean_observation_agreement"
        and contract.get("token_subsampling")
        == "core_context_radial_stratified_v1"
        and contract.get("path_cost_mode")
        == "appearance_boundary_geometric"
        and math.isclose(float(contract.get("path_affinity_floor", -1)), 1e-4),
        "cache region contract differs from isolated screen",
    )


def _validate_current_cache_sidecar(
    cache_path: Path,
    metadata: Mapping,
    *,
    label: str,
) -> dict:
    sidecar_path = cache_path.with_suffix(cache_path.suffix + ".json")
    sidecar, sidecar_sha, sidecar_source = load_json_object(
        sidecar_path,
        label=f"{label} sidecar",
    )
    expected = {
        "output": str(cache_path.resolve()),
        "regions": len(metadata["region_records"]),
        "scenes": len(metadata["scene_names"]),
        "failed_scenes": {},
        "split_role": metadata["split_role"],
        "split_file_sha256": metadata["split_file_sha256"],
        "teacher_target_source": metadata["teacher_target_source"],
        "teacher_replay_cache": metadata["teacher_replay_cache"],
        "teacher_replay_authority": metadata.get(
            "teacher_replay_authority", {}
        ),
    }
    # Mirror the builder's serialized schema exactly.  Legacy train0/1 were
    # built without the scene-intermediate fastpath, so both their metadata
    # and sidecars deliberately omit this optional field.  Missing and an
    # explicitly published empty object are not interchangeable immutable
    # artifact schemas.
    if "scene_intermediate" in metadata:
        expected["scene_intermediate"] = metadata["scene_intermediate"]
    _require(sidecar == expected, f"{label} sidecar differs from cache metadata")
    return {"path": str(sidecar_source), "sha256": sidecar_sha}


def _validate_control_payload(
    path: Path,
    *,
    split_file: Path,
    split_role: str,
    shard: int,
    shard_count: int,
    checkpoint_sha256: str,
    expected_sha256: str | None = None,
    expected_builder_sha256: str | None = None,
) -> tuple[dict, str, Path]:
    payload, digest, source = load_torch_mapping(
        path,
        map_location="cpu",
        expected_sha256=expected_sha256,
        label=f"Surface control {split_role} shard {shard}",
    )
    metadata = payload.get("metadata")
    _require(isinstance(metadata, Mapping), "control cache lacks metadata")
    scenes = _split_scenes(split_file, shard, shard_count)
    rows = len(scenes) * EXPECTED_CACHE_CONTRACT["regions_per_scene"]
    _require(
        metadata.get("schema_version") == 3
        and metadata.get("split_role") == split_role
        and metadata.get("split_file_sha256") == sha256_file(split_file)
        and metadata.get("scene_names") == scenes
        and metadata.get("scene_region_counts")
        == {scene: EXPECTED_CACHE_CONTRACT["regions_per_scene"] for scene in scenes}
        and metadata.get("failed_scenes") == {}
        and metadata.get("complete_scene_regions") is True
        and metadata.get("physical_space_disjoint") is True
        and metadata.get("uses_benchmark_scenes") is False
        and metadata.get("teacher_target_source") == "fresh_official_runtime"
        and metadata.get("teacher_replay_cache") == {}
        and int(metadata.get("teacher_regions_saturated", -1)) == 0
        and metadata.get("radio_checkpoint_sha256") == checkpoint_sha256,
        "control cache provenance differs from fixed-teacher contract",
    )
    if expected_builder_sha256 is not None:
        _require(
            metadata.get("builder_script_sha256") == expected_builder_sha256,
            "control cache builder provenance differs",
        )
    protocol = metadata.get("teacher_target_protocol")
    _require(
        isinstance(protocol, Mapping)
        and metadata.get("teacher_target_protocol_sha256")
        == canonical_json_sha256(protocol),
        "control teacher-target protocol digest differs",
    )
    _validate_region_contract(metadata, candidate_limit=256, context_ratio=1.20)
    records = metadata.get("region_records")
    _require(isinstance(records, list) and len(records) == rows, "control records differ")
    _require(
        {str(record.get("scene", "")) for record in records} == set(scenes),
        "control record scenes differ",
    )
    _require(
        len({str(record.get("region_id", "")) for record in records}) == rows,
        "control region IDs are empty or duplicated",
    )
    _validate_tensor_payload(payload, rows, label="control cache")
    return payload, digest, source


def validate_external_controls(
    *,
    external_root: Path,
    train_split: Path,
    validation_split: Path,
    radio_checkpoint: Path,
    pfir_dev: Path,
    pfir_test: Path,
) -> dict:
    requested_root = Path(os.path.abspath(os.fspath(external_root)))
    resolved_root = requested_root.resolve(strict=True)
    _require(resolved_root.is_dir(), "legacy Surface root is not a directory")
    # The workspace output alias may be a symlink once.  After resolving that
    # explicit root, every descendant directory and final file is no-follow.
    for relative in (Path("caches"), Path("caches") / CONTROL_NAME):
        directory = resolved_root / relative
        info = os.lstat(directory)
        _require(
            stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"legacy Surface descendant is symlinked/non-directory: {relative}",
        )
    manifest_path = resolved_root / "run_manifest.json"
    legacy, legacy_sha, legacy_source = load_json_object(
        manifest_path,
        expected_sha256=LEGACY_MANIFEST_SHA256,
        label="legacy Surface control manifest",
    )
    checkpoint_sha = sha256_file(radio_checkpoint)
    legacy_exclusions = legacy.get("exclusion_files")
    _require(
        legacy.get("schema_version") == 1
        and legacy.get("screen") == "surface-region-fixed-teacher-replay-v2"
        and legacy.get("train_split_sha256") == sha256_file(train_split)
        and legacy.get("validation_split_sha256") == sha256_file(validation_split)
        and legacy.get("radio_checkpoint_sha256") == checkpoint_sha
        and isinstance(legacy_exclusions, Mapping)
        and set(legacy_exclusions.values())
        == {sha256_file(pfir_dev), sha256_file(pfir_test)},
        "legacy Surface control manifest is ineligible",
    )
    legacy_cache = legacy.get("cache_contract", {})
    for key, expected in EXPECTED_CACHE_CONTRACT.items():
        observed = legacy_cache.get(key)
        if isinstance(expected, float):
            _require(math.isclose(float(observed), expected), f"legacy {key} differs")
        else:
            _require(observed == expected, f"legacy {key} differs")
    _require(
        legacy_cache.get("train_shards") == TRAIN_SHARDS
        and legacy_cache.get("validation_shards") == VALIDATION_SHARDS,
        "legacy shard protocol differs",
    )

    bindings = []
    protocol_hashes: set[str] = set()
    teacher_contract_hashes: set[str] = set()
    for shard in (0, 1):
        cache_path = resolved_root / "caches" / CONTROL_NAME / f"train_shard{shard}.pt"
        sidecar_path = cache_path.with_suffix(cache_path.suffix + ".json")
        payload, digest, source = _validate_control_payload(
            cache_path,
            split_file=train_split,
            split_role="train",
            shard=shard,
            shard_count=TRAIN_SHARDS,
            checkpoint_sha256=checkpoint_sha,
            expected_sha256=LEGACY_CONTROL_SHA256[shard],
            expected_builder_sha256=LEGACY_BUILDER_SHA256,
        )
        sidecar, sidecar_sha, sidecar_source = load_json_object(
            sidecar_path,
            expected_sha256=LEGACY_SIDECAR_SHA256[shard],
            label=f"legacy control shard {shard} sidecar",
        )
        expected_scenes = _split_scenes(train_split, shard, TRAIN_SHARDS)
        _require(
            int(sidecar.get("regions", -1))
            == len(expected_scenes) * EXPECTED_CACHE_CONTRACT["regions_per_scene"]
            and int(sidecar.get("scenes", -1)) == len(expected_scenes)
            and sidecar.get("failed_scenes") == {}
            and sidecar.get("split_role") == "train"
            and sidecar.get("split_file_sha256") == sha256_file(train_split)
            and sidecar.get("teacher_target_source") == "fresh_official_runtime"
            and sidecar.get("teacher_replay_cache") == {}
            and Path(str(sidecar.get("output", ""))).resolve() == source.resolve(),
            "legacy control sidecar differs",
        )
        metadata = payload["metadata"]
        protocol_hashes.add(str(metadata["teacher_target_protocol_sha256"]))
        teacher_contract_hashes.add(str(metadata["teacher_region_contract_sha256"]))
        bindings.append(
            {
                "kind": "validated_external_legacy_control",
                "role": "train",
                "shard": shard,
                "cache": {"path": str(source), "sha256": digest},
                "sidecar": {"path": str(sidecar_source), "sha256": sidecar_sha},
                "scene_names": expected_scenes,
                "region_contract_sha256": metadata["region_contract_sha256"],
                "teacher_target_protocol_sha256": metadata[
                    "teacher_target_protocol_sha256"
                ],
                "teacher_region_contract_sha256": metadata[
                    "teacher_region_contract_sha256"
                ],
                "radio_checkpoint_sha256": checkpoint_sha,
                "builder_script_sha256": LEGACY_BUILDER_SHA256,
            }
        )
    _require(len(protocol_hashes) == 1, "external control target protocols differ")
    _require(len(teacher_contract_hashes) == 1, "external teacher contracts differ")
    return {
        "status": "conditional_train01_prescreen_reuse_only",
        "requested_output_root": str(requested_root),
        "resolved_output_root": str(resolved_root),
        "root_alias_resolved_once_then_descendants_nofollow": True,
        "coverage": {
            "role": "train",
            "shards": [0, 1],
            "scenes": sum(len(row["scene_names"]) for row in bindings),
            "complete_four_shard_control": False,
            "claim": "validated_16_scene_prescreen_subset_only",
        },
        "legacy_manifest": {"path": str(legacy_source), "sha256": legacy_sha},
        "controls": bindings,
        "teacher_target_protocol_sha256": next(iter(protocol_hashes)),
        "teacher_region_contract_sha256": next(iter(teacher_contract_hashes)),
    }


def _record_sources(repo_root: Path, relatives: Sequence[str]) -> dict[str, str]:
    return {relative: sha256_file(repo_root / relative) for relative in relatives}


def _validate_thermal_values(values: Mapping) -> None:
    _require(
        THERMAL_KEYS <= set(values)
        and int(values["physical_gpu"]) == 1
        and values["peer_gpu"] is None
        and values["peer_activity_action"] == "terminate"
        and values["owner_pid_namespace_mode"]
        == "exclusive-singleton-after-clear-v1"
        and str(values["gpu_uuid"]).startswith("GPU-")
        and 0 < int(values["maximum_start_temperature_c"])
        < int(values["canary_max_temp_c"])
        < int(values["soft_pause_temperature_c"])
        < int(values["maximum_temperature_c"])
        and int(values["soft_resume_temperature_c"])
        < int(values["soft_pause_temperature_c"])
        and int(values["poll_seconds"]) > 0
        and float(values["maximum_power_limit_w"]) > 0
        and int(values["peer_pause_temperature_c"]) == 0
        and int(values["peer_resume_temperature_c"]) == 0
        and int(values["peer_quiet_seconds_before_launch"]) == 0
        and float(values["peer_max_power_w"]) == 0
        and int(values["peer_max_memory_mib"]) == 0
        and int(values["peer_max_utilization_pct"]) == 100
        and float(values["radio_pacing_seconds_per_image"]) > 0,
        "thermal safety values are invalid or internally inconsistent",
    )


def create_manifest(args: argparse.Namespace) -> dict:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    runner = Path(args.runner).resolve()
    authority = Path(__file__).resolve()
    external = validate_external_controls(
        external_root=Path(args.external_control_root),
        train_split=Path(args.train_split),
        validation_split=Path(args.validation_split),
        radio_checkpoint=Path(args.radio_checkpoint),
        pfir_dev=Path(args.pfir_dev),
        pfir_test=Path(args.pfir_test),
    )
    _require(
        args.external_reuse_authorization == "1:environment_override",
        "full run requires explicit SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE=1",
    )
    checkpoint_sha = sha256_file(args.radio_checkpoint)
    closure = build_runtime_closure(
        repo_root=repo_root,
        radio_repo=args.radio_repo,
        radio_checkpoint=args.radio_checkpoint,
        checkpoint_sha256=checkpoint_sha,
    )
    implementation_relatives = (
        "radio_gs/scripts/build_scannet_surface_region_cache.py",
        "radio_gs/scripts/train_surface_region_summary_readout.py",
        "radio_gs/interfaces/surface_region_contract.py",
        "radio_gs/interfaces/surface_region_summary.py",
        "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
        "radio_gs/scripts/surface_region_run_guard.py",
    )
    thermal_values = json.loads(args.thermal_values)
    thermal_sources = json.loads(args.thermal_sources)
    _require(
        set(thermal_values) == THERMAL_KEYS
        and set(thermal_sources) == THERMAL_KEYS,
        "thermal values/provenance fields differ",
    )
    _validate_thermal_values(thermal_values)
    control_sources = []
    external_by_shard = {row["shard"]: row for row in external["controls"]}
    for role, count in (("train", TRAIN_SHARDS), ("validation", VALIDATION_SHARDS)):
        for shard in range(count):
            if role == "train" and shard in external_by_shard:
                control_sources.append(external_by_shard[shard])
            else:
                path = (
                    output_root
                    / "caches"
                    / CONTROL_NAME
                    / f"{role}_shard{shard}.pt"
                )
                control_sources.append(
                    {
                        "kind": "run_local_fresh_control",
                        "role": role,
                        "shard": shard,
                        "cache_path": str(path),
                        "sidecar_path": str(path.with_suffix(path.suffix + ".json")),
                    }
                )
    payload = {
        "schema_version": 1,
        "screen": SCREEN_NAME,
        "source_snapshot_root": closure["runtime_fingerprint"][
            "repository_import_root"
        ],
        "source_snapshot_import_root": closure["runtime_fingerprint"][
            "repository_import_root"
        ],
        "source_snapshot_tree_sha256": closure["repository_sources"]["digest"],
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "train_split": str(Path(args.train_split).resolve()),
        "train_split_sha256": sha256_file(args.train_split),
        "validation_split": str(Path(args.validation_split).resolve()),
        "validation_split_sha256": sha256_file(args.validation_split),
        "radio_repo": str(Path(args.radio_repo).resolve()),
        "radio_version": args.radio_version,
        "radio_checkpoint": str(Path(args.radio_checkpoint).resolve()),
        "radio_checkpoint_sha256": checkpoint_sha,
        "exclusion_files": {
            str(Path(args.pfir_dev).resolve()): sha256_file(args.pfir_dev),
            str(Path(args.pfir_test).resolve()): sha256_file(args.pfir_test),
        },
        "pfir_dev": str(Path(args.pfir_dev).resolve()),
        "pfir_test": str(Path(args.pfir_test).resolve()),
        "excluded_scene_names": sorted(args.excluded_scene_names.split(",")),
        "cache_contract": {
            "train_shards": TRAIN_SHARDS,
            "validation_shards": VALIDATION_SHARDS,
            **EXPECTED_CACHE_CONTRACT,
            "adaptor_batch_size": int(args.adaptor_batch_size),
            "radio_thermal_pacing_seconds_per_image": float(
                thermal_values["radio_pacing_seconds_per_image"]
            ),
            "durable_scene_resume": True,
        },
        "cache_candidate": {
            "name": CACHE_NAME,
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        },
        "deferred_cache_branches": [
            "context_c1024_uniform",
            "core_c1024_geometric",
            "context_c4096_complete_core_then_context",
        ],
        "control_sources": control_sources,
        "external_control_authority": external,
        "external_control_reuse_authorization": {
            "enabled": True,
            "source": "environment_override",
            "environment_variable": (
                "SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE=1"
            ),
            "scope": "exact_allowlisted_train_shards_0_1_only",
            "not_a_complete_control_claim": True,
        },
        "legacy_teacher_replay_authorities": [
            {
                "role": "train",
                "shard": shard,
                "path": str(
                    output_root
                    / "teacher_replay_authorities"
                    / f"train_shard{shard}.json"
                ),
            }
            for shard in (0, 1)
        ],
        "readout_contract": {
            "same_cache_for_every_variant": True,
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "hidden_dim": 256,
            "epochs": 60,
            "patience": 10,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "token_weight": 0.25,
            "relation_weight": 0.1,
            "reliability_attention_mode": "log_prior",
        },
        "selection_contract": {
            "baseline": JOINT_CONTEXT_POOLING,
            "candidate": SEPARATE_CONTEXT_POOLING,
            "minimum_mean_score_gain": 0.001,
            "minimum_seed_wins": 2,
            "maximum_descriptor_component_drop": 0.002,
            "descriptor_components": list(DESCRIPTOR_COMPONENTS),
            "uses_benchmark_queries": False,
        },
        "thermal_safety_contract": {
            **thermal_values,
            "peer_activity_interrupt_exit_code": 87,
            "override_provenance": {
                key: {
                    "source": thermal_sources[key],
                    "effective_value": thermal_values[key],
                }
                for key in sorted(thermal_values)
            },
            "balanced_default_evidence": (
                "GPU1-only p6 selected between validated p8 <=71C and faster "
                "p4 <=73C; start65/soft75-resume70/hard78 replaces all peer gating"
            ),
        },
        "canary_contract": {
            "stage": "cache_control_c256_geometric_train_2",
            "terminal": str(
                output_root / "caches" / CONTROL_NAME / "train_shard2.pt"
            ),
            "maximum_temperature_c": int(thermal_values["canary_max_temp_c"]),
            "default_fail_closed_after_pass": True,
            "resume_environment_variable": "SURFACE_CANARY_RESUME=1",
        },
        "intermediate_fastpath": {
            "mode": args.intermediate_fastpath,
            "required_for_run_local_shards": (
                args.intermediate_fastpath == "required_local_shards"
            ),
            "legacy_train_shards_without_intermediate": [0, 1],
            "fallback": (
                "none"
                if args.intermediate_fastpath == "required_local_shards"
                else "explicit_full_builder"
            ),
            "note": (
                "The scientific contract does not depend on an intermediate fastpath. "
                "Required-local mode covers newly built train2/3 and validation0/1; "
                "legacy train0/1 use the full builder because no historical intermediate exists."
            ),
        },
        # Compatibility field required by the existing closure/attempt auditor.
        # The active new runner is independently bound immediately below.
        "runner_sha256": closure["repository_sources"]["files"][
            "radio_gs/scripts/run_surface_region_context_recovery_screen.sh"
        ],
        "closure_guard_compatibility": {
            "runner_sha256_field": (
                "legacy_surface_closure_guard_contract_only"
            ),
            "active_orchestration_runner_field": "active_runner",
            "active_runner_verified_by_authority_before_and_after_every_stage": True,
        },
        "active_runner": file_record(runner),
        "active_authority": file_record(authority),
        "implementation_sources": _record_sources(repo_root, implementation_relatives),
        "runtime_closure": closure,
        "attempt_receipt_contract": {
            "artifact_type": "surface-region-stage-attempt-v1",
            "schema_version": 1,
            "root": str(output_root / "stage_attempts"),
            "log_root": str(output_root / "logs"),
            "telemetry_path": str(Path(args.telemetry).resolve()),
            "immutable_no_clobber": True,
            "owner_audit_required": True,
            "owner_audit_location": "beside_receipt",
        },
    }
    output = Path(args.manifest)
    if output.is_file():
        previous, _, _ = load_json_object(output, label="attention screen manifest")
        _require(previous == payload, "OUTPUT_ROOT belongs to another immutable run")
    else:
        existing = [
            path
            for path in output_root.rglob("*")
            if path.is_file() and path != output
        ]
        _require(not existing, "OUTPUT_ROOT contains artifacts without a manifest")
        write_frozen_json(output, payload)
    return payload


def verify_manifest(path: Path) -> dict:
    manifest, _, _ = load_json_object(path, label="attention screen manifest")
    output_root = path.resolve().parent
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("screen") == SCREEN_NAME,
        "wrong attention screen manifest",
    )
    current_closure = build_runtime_closure(
        repo_root=Path(__file__).resolve().parents[2],
        radio_repo=manifest["radio_repo"],
        radio_checkpoint=manifest["radio_checkpoint"],
        checkpoint_sha256=manifest["radio_checkpoint_sha256"],
    )
    _require(current_closure == manifest["runtime_closure"], "runtime closure changed")
    _require(
        manifest.get("runner_sha256")
        == current_closure["repository_sources"]["files"][
            "radio_gs/scripts/run_surface_region_context_recovery_screen.sh"
        ]
        and manifest.get("closure_guard_compatibility")
        == {
            "runner_sha256_field": "legacy_surface_closure_guard_contract_only",
            "active_orchestration_runner_field": "active_runner",
            "active_runner_verified_by_authority_before_and_after_every_stage": True,
        },
        "closure-guard compatibility binding differs",
    )
    validate_file_record(manifest["active_runner"], label="active attention runner")
    validate_file_record(manifest["active_authority"], label="attention authority")
    for relative, digest in manifest["implementation_sources"].items():
        _require(
            sha256_file(Path(__file__).resolve().parents[2] / relative) == digest,
            f"implementation source changed: {relative}",
        )
    current_external = validate_external_controls(
        external_root=Path(
            manifest["external_control_authority"]["requested_output_root"]
        ),
        train_split=Path(manifest["train_split"]),
        validation_split=Path(manifest["validation_split"]),
        radio_checkpoint=Path(manifest["radio_checkpoint"]),
        pfir_dev=Path(manifest["pfir_dev"]),
        pfir_test=Path(manifest["pfir_test"]),
    )
    _require(
        current_external == manifest["external_control_authority"],
        "external control authority changed",
    )
    _require(
        manifest.get("external_control_reuse_authorization")
        == {
            "enabled": True,
            "source": "environment_override",
            "environment_variable": (
                "SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE=1"
            ),
            "scope": "exact_allowlisted_train_shards_0_1_only",
            "not_a_complete_control_claim": True,
        },
        "external control reuse was not explicitly authorized",
    )
    _require(
        manifest.get("legacy_teacher_replay_authorities")
        == [
            {
                "role": "train",
                "shard": shard,
                "path": str(
                    output_root
                    / "teacher_replay_authorities"
                    / f"train_shard{shard}.json"
                ),
            }
            for shard in (0, 1)
        ],
        "legacy teacher replay authority inventory differs",
    )
    _require(
        sha256_file(manifest["train_split"]) == manifest["train_split_sha256"]
        and sha256_file(manifest["validation_split"])
        == manifest["validation_split_sha256"]
        and sha256_file(manifest["radio_checkpoint"])
        == manifest["radio_checkpoint_sha256"]
        and sha256_file(manifest["pfir_dev"])
        == manifest["exclusion_files"][manifest["pfir_dev"]]
        and sha256_file(manifest["pfir_test"])
        == manifest["exclusion_files"][manifest["pfir_test"]],
        "screen input digest changed",
    )
    cache_contract = manifest.get("cache_contract", {})
    _require(
        cache_contract.get("train_shards") == TRAIN_SHARDS
        and cache_contract.get("validation_shards") == VALIDATION_SHARDS
        and all(cache_contract.get(key) == value for key, value in EXPECTED_CACHE_CONTRACT.items())
        and cache_contract.get("durable_scene_resume") is True,
        "cache contract differs",
    )
    _require(
        manifest.get("cache_candidate")
        == {
            "name": CACHE_NAME,
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        }
        and manifest.get("deferred_cache_branches")
        == [
            "context_c1024_uniform",
            "core_c1024_geometric",
            "context_c4096_complete_core_then_context",
        ],
        "isolated cache candidate/deferred branches differ",
    )
    readout = manifest.get("readout_contract", {})
    _require(
        readout
        == {
            "same_cache_for_every_variant": True,
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "hidden_dim": 256,
            "epochs": 60,
            "patience": 10,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "token_weight": 0.25,
            "relation_weight": 0.1,
            "reliability_attention_mode": "log_prior",
        }
        and manifest.get("selection_contract")
        == {
            "baseline": JOINT_CONTEXT_POOLING,
            "candidate": SEPARATE_CONTEXT_POOLING,
            "minimum_mean_score_gain": 0.001,
            "minimum_seed_wins": 2,
            "maximum_descriptor_component_drop": 0.002,
            "descriptor_components": list(DESCRIPTOR_COMPONENTS),
            "uses_benchmark_queries": False,
        },
        "attention comparison or promotion gate differs",
    )
    controls = manifest.get("control_sources", [])
    _require(
        isinstance(controls, list)
        and len(controls) == TRAIN_SHARDS + VALIDATION_SHARDS
        and {(row.get("role"), row.get("shard")) for row in controls}
        == {
            *(("train", shard) for shard in range(TRAIN_SHARDS)),
            *(("validation", shard) for shard in range(VALIDATION_SHARDS)),
        }
        and controls[:2] == current_external["controls"],
        "control source inventory differs",
    )
    for row in controls[2:]:
        expected = (
            output_root
            / "caches"
            / CONTROL_NAME
            / f"{row['role']}_shard{row['shard']}.pt"
        )
        _require(
            row
            == {
                "kind": "run_local_fresh_control",
                "role": row["role"],
                "shard": row["shard"],
                "cache_path": str(expected),
                "sidecar_path": str(expected.with_suffix(expected.suffix + ".json")),
            },
            "run-local control path escaped OUTPUT_ROOT",
        )
    thermal = manifest.get("thermal_safety_contract", {})
    _validate_thermal_values(thermal)
    _require(
        int(thermal.get("peer_activity_interrupt_exit_code", -1)) == 87
        and set(thermal.get("override_provenance", {})) == THERMAL_KEYS
        and all(
            thermal["override_provenance"][key].get("effective_value")
            == thermal[key]
            and thermal["override_provenance"][key].get("source")
            in {
                "balanced_default",
                "environment_override",
                "frozen_contract",
                "runtime_attested",
            }
            for key in THERMAL_KEYS
        ),
        "thermal override provenance differs",
    )
    fastpath = manifest.get("intermediate_fastpath", {})
    _require(
        fastpath.get("mode") in {"required_local_shards", "disabled"}
        and fastpath.get("required_for_run_local_shards")
        is (fastpath.get("mode") == "required_local_shards")
        and fastpath.get("legacy_train_shards_without_intermediate") == [0, 1],
        "intermediate fastpath contract differs",
    )
    canary = manifest.get("canary_contract", {})
    _require(
        canary.get("stage") == "cache_control_c256_geometric_train_2"
        and Path(str(canary.get("terminal", ""))).resolve()
        == output_root / "caches" / CONTROL_NAME / "train_shard2.pt"
        and int(canary.get("maximum_temperature_c", -1))
        == int(thermal["canary_max_temp_c"]),
        "canary contract differs",
    )
    attempt = manifest.get("attempt_receipt_contract", {})
    _require(
        Path(str(attempt.get("root", ""))).resolve()
        == output_root / "stage_attempts"
        and Path(str(attempt.get("log_root", ""))).resolve()
        == output_root / "logs"
        and attempt.get("artifact_type") == "surface-region-stage-attempt-v1"
        and attempt.get("schema_version") == 1
        and attempt.get("owner_audit_required") is True
        and attempt.get("owner_audit_location") == "beside_receipt",
        "attempt receipt contract differs",
    )
    return manifest


def _control_path(manifest: Mapping, role: str, shard: int) -> Path:
    matches = [
        row
        for row in manifest["control_sources"]
        if row["role"] == role and int(row["shard"]) == shard
    ]
    _require(len(matches) == 1, "control source mapping is not unique")
    row = matches[0]
    return Path(row["cache"]["path"] if "cache" in row else row["cache_path"])


def _legacy_replay_authority_payload(
    manifest_path: Path,
    manifest: Mapping,
    shard: int,
) -> dict:
    _require(shard in (0, 1), "legacy replay authority is limited to train0/1")
    controls = [
        row
        for row in manifest["external_control_authority"]["controls"]
        if row["role"] == "train" and int(row["shard"]) == shard
    ]
    _require(len(controls) == 1, "legacy replay control binding is not unique")
    control = controls[0]
    return {
        "artifact_type": TEACHER_REPLAY_AUTHORITY_ARTIFACT_TYPE,
        "schema_version": 1,
        "authorization_scope": (
            "exact_historical_cache_fixed_teacher_replay_only"
        ),
        "run_manifest": file_record(manifest_path),
        "cache": control["cache"],
        "split_role": "train",
        "split_file_sha256": manifest["train_split_sha256"],
        "scene_names": control["scene_names"],
        "teacher_region_contract_sha256": control[
            "teacher_region_contract_sha256"
        ],
        "teacher_target_protocol_sha256": control[
            "teacher_target_protocol_sha256"
        ],
        "radio_checkpoint_sha256": manifest["radio_checkpoint_sha256"],
        "source_builder_script_sha256": LEGACY_BUILDER_SHA256,
    }


def write_legacy_replay_authority(
    manifest_path: Path,
    shard: int,
    output: Path,
) -> dict:
    manifest = verify_manifest(manifest_path)
    expected_outputs = {
        int(row["shard"]): Path(row["path"])
        for row in manifest["legacy_teacher_replay_authorities"]
    }
    _require(
        shard in expected_outputs
        and output.resolve() == expected_outputs[shard].resolve(),
        "legacy replay authority output escaped the manifest",
    )
    payload = _legacy_replay_authority_payload(manifest_path, manifest, shard)
    write_frozen_json(output, payload)
    return file_record(output)


def verify_cache_pairing(manifest_path: Path, output: Path) -> dict:
    manifest = verify_manifest(manifest_path)
    root = manifest_path.parent
    rows = []
    protocol_hashes: set[str] = set()
    teacher_contract_hashes: set[str] = set()
    for role, count, split_path in (
        ("train", TRAIN_SHARDS, Path(manifest["train_split"])),
        ("validation", VALIDATION_SHARDS, Path(manifest["validation_split"])),
    ):
        for shard in range(count):
            control_path = _control_path(manifest, role, shard)
            expected = None
            expected_builder = manifest["implementation_sources"][
                "radio_gs/scripts/build_scannet_surface_region_cache.py"
            ]
            if role == "train" and shard < 2:
                expected = LEGACY_CONTROL_SHA256[shard]
                expected_builder = LEGACY_BUILDER_SHA256
            control, control_sha, control_source = _validate_control_payload(
                control_path,
                split_file=split_path,
                split_role=role,
                shard=shard,
                shard_count=count,
                checkpoint_sha256=manifest["radio_checkpoint_sha256"],
                expected_sha256=expected,
                expected_builder_sha256=expected_builder,
            )
            control_sidecar = None
            if not (role == "train" and shard < 2):
                control_sidecar = _validate_current_cache_sidecar(
                    control_source,
                    control["metadata"],
                    label=f"run-local control {role} shard {shard}",
                )
            treatment_path = root / "caches" / CACHE_NAME / f"{role}_shard{shard}.pt"
            treatment, treatment_sha, treatment_source = load_torch_mapping(
                treatment_path,
                map_location="cpu",
                label=f"c1024 {role} shard {shard}",
            )
            metadata = treatment.get("metadata", {})
            control_meta = control["metadata"]
            scenes = _split_scenes(split_path, shard, count)
            row_count = len(scenes) * EXPECTED_CACHE_CONTRACT["regions_per_scene"]
            expected_replay_authority: dict[str, str] = {}
            if role == "train" and shard < 2:
                authority_path = Path(
                    manifest["legacy_teacher_replay_authorities"][shard]["path"]
                )
                authority_payload, authority_sha, authority_source = load_json_object(
                    authority_path,
                    label=f"legacy teacher replay authority train shard {shard}",
                )
                _require(
                    authority_payload
                    == _legacy_replay_authority_payload(
                        manifest_path, manifest, shard
                    ),
                    "legacy teacher replay authority payload differs",
                )
                expected_replay_authority = {
                    "path": str(authority_source),
                    "sha256": authority_sha,
                }
            _require(
                metadata.get("schema_version") == 3
                and metadata.get("split_role") == role
                and metadata.get("split_file_sha256") == sha256_file(split_path)
                and metadata.get("scene_names") == scenes
                and metadata.get("scene_region_counts")
                == {scene: EXPECTED_CACHE_CONTRACT["regions_per_scene"] for scene in scenes}
                and metadata.get("failed_scenes") == {}
                and metadata.get("complete_scene_regions") is True
                and metadata.get("teacher_target_source") == "exact_cache_replay"
                and metadata.get("teacher_regions_saturated") == 0
                and metadata.get("teacher_target_protocol_sha256")
                == control_meta.get("teacher_target_protocol_sha256")
                and metadata.get("teacher_region_contract_sha256")
                == control_meta.get("teacher_region_contract_sha256")
                and metadata.get("radio_checkpoint_sha256")
                == manifest["radio_checkpoint_sha256"],
                "c1024 cache provenance differs",
            )
            _require(
                metadata.get("builder_script_sha256")
                == manifest["implementation_sources"][
                    "radio_gs/scripts/build_scannet_surface_region_cache.py"
                ],
                "c1024 cache was not built by the frozen implementation",
            )
            _require(
                metadata.get("teacher_replay_cache")
                == {"path": str(control_source), "sha256": control_sha},
                "c1024 teacher replay cache binding differs",
            )
            _require(
                metadata.get("teacher_replay_authority", {})
                == expected_replay_authority,
                "c1024 historical replay authority binding differs",
            )
            treatment_sidecar = _validate_current_cache_sidecar(
                treatment_source,
                metadata,
                label=f"c1024 {role} shard {shard}",
            )
            fastpath_mode = manifest["intermediate_fastpath"]["mode"]
            control_intermediate = control_meta.get("scene_intermediate", {})
            treatment_intermediate = metadata.get("scene_intermediate", {})
            uses_legacy_without_intermediate = role == "train" and shard < 2
            if fastpath_mode == "required_local_shards" and not uses_legacy_without_intermediate:
                _require(
                    isinstance(control_intermediate, Mapping)
                    and control_intermediate.get("mode") == "fresh_publish"
                    and isinstance(treatment_intermediate, Mapping)
                    and treatment_intermediate.get("mode") == "exact_replay"
                    and treatment_intermediate.get("manifest")
                    == control_intermediate.get("manifest")
                    and treatment_intermediate.get("scene_records")
                    == control_intermediate.get("scene_records"),
                    "c1024 cache did not replay the bound local intermediate",
                )
            else:
                _require(
                    control_intermediate == {} and treatment_intermediate == {},
                    "unexpected scene-intermediate provenance",
                )
            _validate_region_contract(metadata, candidate_limit=1024, context_ratio=1.20)
            _validate_tensor_payload(treatment, row_count, label="c1024 cache")
            treatment_records = [
                _teacher_identity(row) for row in metadata.get("region_records", [])
            ]
            control_records = [
                _teacher_identity(row) for row in control_meta.get("region_records", [])
            ]
            _require(treatment_records == control_records, "fixed teacher identities differ")
            for key in (
                "official_summary_tokens",
                "official_crop_summaries",
                "teacher_mask",
            ):
                _require(torch.equal(treatment[key], control[key]), f"{key} replay differs")
            protocol_hashes.add(str(metadata["teacher_target_protocol_sha256"]))
            teacher_contract_hashes.add(str(metadata["teacher_region_contract_sha256"]))
            rows.append(
                {
                    "role": role,
                    "shard": shard,
                    "control": {"path": str(control_source), "sha256": control_sha},
                    "control_sidecar": (
                        control_sidecar
                        if control_sidecar is not None
                        else manifest["external_control_authority"]["controls"][shard][
                            "sidecar"
                        ]
                    ),
                    "c1024": {"path": str(treatment_source), "sha256": treatment_sha},
                    "c1024_sidecar": treatment_sidecar,
                    "regions": row_count,
                    "teacher_target_protocol_sha256": metadata[
                        "teacher_target_protocol_sha256"
                    ],
                }
            )
    _require(len(protocol_hashes) == 1, "c1024 target protocols differ")
    _require(len(teacher_contract_hashes) == 1, "c1024 teacher contracts differ")
    report = {
        "schema_version": 1,
        "artifact_type": "surface_c1024_exact_teacher_pairing",
        "status": "single_c1024_cache_exact_teacher_replay_verified",
        "run_manifest": file_record(manifest_path),
        "rows": rows,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
    }
    write_frozen_json(output, report)
    return report


def intermediate_binding(control_sidecar: Path, expected_root: Path) -> dict:
    """Return the exact fastpath authority bound by a fresh control report."""

    report, report_sha, report_source = load_json_object(
        control_sidecar,
        label="Surface control sidecar with scene intermediate",
    )
    provenance = report.get("scene_intermediate")
    _require(
        isinstance(provenance, Mapping)
        and provenance.get("mode") == "fresh_publish"
        and Path(str(provenance.get("root", ""))).resolve(strict=True)
        == expected_root.resolve(strict=True)
        and isinstance(provenance.get("manifest"), Mapping)
        and isinstance(provenance.get("scene_records"), list)
        and provenance.get("scene_records"),
        "control sidecar lacks the required fresh intermediate authority",
    )
    manifest_path = validate_file_record(
        provenance["manifest"],
        label="control scene-intermediate manifest",
    )
    _require(
        manifest_path.parent.resolve(strict=True) == expected_root.resolve(strict=True),
        "control scene-intermediate manifest escaped its dedicated root",
    )
    return {
        "control_sidecar": {"path": str(report_source), "sha256": report_sha},
        "root": str(expected_root.resolve(strict=True)),
        "manifest": dict(provenance["manifest"]),
        "scene_records": provenance["scene_records"],
    }


def audit_canary(manifest_path: Path, output: Path) -> dict:
    manifest = verify_manifest(manifest_path)
    contract = manifest["canary_contract"]
    stage = contract["stage"]
    attempt_contract = manifest["attempt_receipt_contract"]
    inventory = audit_attempt_inventory(
        manifest_path=manifest_path,
        attempt_root=attempt_contract["root"],
        log_root=attempt_contract["log_root"],
    )
    attempts = [row for row in inventory["attempts"] if row["stage"] == stage]
    _require(attempts and attempts[-1]["result"] == "completed", "canary lacks completion")
    _require(
        all(int(row["kernel_journal"]["fault_count"]) == 0 for row in attempts),
        "canary kernel journal contains a GPU fault",
    )
    telemetry_path = Path(attempt_contract["telemetry_path"])
    start_line = min(int(row["telemetry_interval"]["start_line"]) for row in attempts)
    end_line = max(int(row["telemetry_interval"]["end_line"]) for row in attempts)
    telemetry_rows = _telemetry_rows(
        telemetry_path,
        start_line=start_line,
        end_line=end_line,
    )
    thermal = summarize_canary_telemetry(
        telemetry_rows,
        expected_gpu=int(manifest["thermal_safety_contract"]["physical_gpu"]),
        maximum_temperature_c=int(contract["maximum_temperature_c"]),
        peer_gpu=manifest["thermal_safety_contract"]["peer_gpu"],
    )
    authorized_interrupts = sum(
        row["result"]
        == "peer_activity_interrupted_cuda_released_retry_authorized"
        for row in attempts
    )
    _require(
        authorized_interrupts == 0
        and int(thermal["peer_interrupt_count"]) == 0,
        "GPU1-only canary contains a forbidden peer interruption",
    )
    terminal = Path(contract["terminal"])
    _, cache_sha, cache_source = _validate_control_payload(
        terminal,
        split_file=Path(manifest["train_split"]),
        split_role="train",
        shard=2,
        shard_count=TRAIN_SHARDS,
        checkpoint_sha256=manifest["radio_checkpoint_sha256"],
    )
    report = {
        "schema_version": 1,
        "artifact_type": "surface_attention_p6_full_shard_canary",
        "status": "canary_passed_resume_authorized",
        "run_manifest": file_record(manifest_path),
        "stage": stage,
        "attempts": attempts,
        "authorized_peer_interrupts": authorized_interrupts,
        "telemetry_interval": {"start_line": start_line, "end_line": end_line},
        "thermal_summary": thermal,
        "cache_terminal": {"path": str(cache_source), "sha256": cache_sha},
    }
    write_frozen_json(output, report)
    return report


def promotion_decision(rows: Mapping[str, Mapping], gate: Mapping) -> dict:
    """Apply the frozen paired-seed gate without consulting any benchmark."""

    baseline = rows[JOINT_CONTEXT_POOLING]
    candidate = rows[SEPARATE_CONTEXT_POOLING]
    baseline_seeds = baseline["seeds"]
    candidate_seeds = candidate["seeds"]
    _require(
        [row["seed"] for row in baseline_seeds] == list(SEEDS)
        and [row["seed"] for row in candidate_seeds] == list(SEEDS),
        "attention gate requires paired seeds 0/1/2",
    )
    gain = float(candidate["mean_selection_score"]) - float(
        baseline["mean_selection_score"]
    )
    wins = sum(
        float(current["best_selection_score"])
        > float(reference["best_selection_score"])
        for current, reference in zip(candidate_seeds, baseline_seeds)
    )
    drops = {
        key: float(baseline["mean_validation"][key])
        - float(candidate["mean_validation"][key])
        for key in DESCRIPTOR_COMPONENTS
    }
    numeric_epsilon = 1e-12
    passed = (
        gain + numeric_epsilon >= float(gate["minimum_mean_score_gain"])
        and wins >= int(gate["minimum_seed_wins"])
        and max(drops.values())
        <= float(gate["maximum_descriptor_component_drop"]) + numeric_epsilon
    )
    return {
        "mean_score_gain_over_joint": gain,
        "seed_wins_over_joint": wins,
        "descriptor_component_drops_from_joint": drops,
        "eligible_for_query_free_promotion": passed,
    }


def finalize(manifest_path: Path, pairing_path: Path, output: Path) -> dict:
    manifest = verify_manifest(manifest_path)
    pairing, _, _ = load_json_object(pairing_path, label="c1024 pairing report")
    _require(
        pairing.get("run_manifest") == file_record(manifest_path)
        and pairing.get("status") == "single_c1024_cache_exact_teacher_replay_verified",
        "c1024 pairing report differs",
    )
    expected_cache_records = {
        (row["role"], int(row["shard"])): row["c1024"] for row in pairing["rows"]
    }
    _require(len(expected_cache_records) == 6, "pairing report lacks c1024 shards")
    root = manifest_path.parent
    rows: dict[str, dict] = {}
    for variant in VARIANTS:
        seed_rows = []
        for seed in SEEDS:
            checkpoint_path = root / "readouts" / f"{variant}_seed{seed}.pt"
            checkpoint, checkpoint_sha, checkpoint_source = load_torch_mapping(
                checkpoint_path,
                map_location="cpu",
                label=f"attention readout {variant} seed {seed}",
            )
            report_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
            report, _, _ = load_json_object(report_path, label="attention readout report")
            architecture = checkpoint.get("architecture", {})
            observed_mode = architecture.get("context_pooling_mode", JOINT_CONTEXT_POOLING)
            config = checkpoint.get("training_config", {})
            provenance = checkpoint.get("provenance", {})
            train_paths = provenance.get("train", {}).get("cache_paths")
            validation_paths = provenance.get("validation", {}).get("cache_paths")
            expected_train = [
                expected_cache_records[("train", shard)]["path"]
                for shard in range(TRAIN_SHARDS)
            ]
            expected_validation = [
                expected_cache_records[("validation", shard)]["path"]
                for shard in range(VALIDATION_SHARDS)
            ]
            _require(
                checkpoint.get("schema_version") == 3
                and observed_mode == variant
                and config.get("context_pooling_mode") == variant
                and int(config.get("seed", -1)) == seed
                and train_paths == expected_train
                and validation_paths == expected_validation
                and report.get("checkpoint_sha256") == checkpoint_sha
                and report.get("architecture") == architecture,
                "readout is not bound to the requested variant/cache/seed",
            )
            validation = report.get("validation", {})
            _require(
                all(
                    isinstance(validation.get(key), (int, float))
                    and not isinstance(validation.get(key), bool)
                    and math.isfinite(float(validation[key]))
                    for key in ALL_VALIDATION_COMPONENTS
                ),
                "readout validation metrics are missing or nonfinite",
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "checkpoint": {"path": str(checkpoint_source), "sha256": checkpoint_sha},
                    "best_epoch": int(report["best_epoch"]),
                    "best_selection_score": float(report["best_selection_score"]),
                    "validation": {
                        key: float(validation[key]) for key in ALL_VALIDATION_COMPONENTS
                    },
                }
            )
        rows[variant] = {
            "seeds": seed_rows,
            "mean_selection_score": sum(
                row["best_selection_score"] for row in seed_rows
            )
            / len(seed_rows),
            "mean_validation": {
                key: sum(row["validation"][key] for row in seed_rows) / len(seed_rows)
                for key in ALL_VALIDATION_COMPONENTS
            },
        }

    gate = manifest["selection_contract"]
    decision = promotion_decision(rows, gate)
    rows[SEPARATE_CONTEXT_POOLING].update(decision)
    passed = bool(decision["eligible_for_query_free_promotion"])
    report = {
        "schema_version": 1,
        "artifact_type": "surface_c1024_attention_pooling_screen",
        "selection_status": (
            "separate_attention_promoted_benchmark_gate_still_closed"
            if passed
            else "joint_attention_retained"
        ),
        "selected_variant": (
            SEPARATE_CONTEXT_POOLING if passed else JOINT_CONTEXT_POOLING
        ),
        "promotion_gate_passed": passed,
        "run_manifest": file_record(manifest_path),
        "cache_pairing_report": file_record(pairing_path),
        "variants": rows,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "next_gate": (
            "freeze winning c1024 readout before any c4096 or text-response screen"
        ),
    }
    write_frozen_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    external = sub.add_parser("validate-external")
    external.add_argument("--external-control-root", required=True, type=Path)
    external.add_argument("--train-split", required=True, type=Path)
    external.add_argument("--validation-split", required=True, type=Path)
    external.add_argument("--radio-checkpoint", required=True, type=Path)
    external.add_argument("--pfir-dev", required=True, type=Path)
    external.add_argument("--pfir-test", required=True, type=Path)

    create = sub.add_parser("create-manifest")
    for name in (
        "repo-root",
        "output-root",
        "runner",
        "manifest",
        "dataset-root",
        "train-split",
        "validation-split",
        "radio-repo",
        "radio-version",
        "radio-checkpoint",
        "pfir-dev",
        "pfir-test",
        "excluded-scene-names",
        "external-control-root",
        "telemetry",
        "thermal-values",
        "thermal-sources",
        "intermediate-fastpath",
        "external-reuse-authorization",
    ):
        create.add_argument(f"--{name}", required=True)
    create.add_argument("--adaptor-batch-size", required=True, type=int)

    verify = sub.add_parser("verify-manifest")
    verify.add_argument("--manifest", required=True, type=Path)
    pairing = sub.add_parser("verify-pairing")
    pairing.add_argument("--manifest", required=True, type=Path)
    pairing.add_argument("--output", required=True, type=Path)
    canary = sub.add_parser("audit-canary")
    canary.add_argument("--manifest", required=True, type=Path)
    canary.add_argument("--output", required=True, type=Path)
    binding = sub.add_parser("intermediate-binding")
    binding.add_argument("--control-sidecar", required=True, type=Path)
    binding.add_argument("--expected-root", required=True, type=Path)
    binding.add_argument(
        "--output-format",
        choices=("json", "lines"),
        default="json",
    )
    replay_authority = sub.add_parser("legacy-replay-authority")
    replay_authority.add_argument("--manifest", required=True, type=Path)
    replay_authority.add_argument("--shard", required=True, type=int)
    replay_authority.add_argument("--output", required=True, type=Path)
    replay_authority.add_argument(
        "--output-format",
        choices=("json", "lines"),
        default="json",
    )
    finish = sub.add_parser("finalize")
    finish.add_argument("--manifest", required=True, type=Path)
    finish.add_argument("--pairing", required=True, type=Path)
    finish.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate-external":
        result = validate_external_controls(
            external_root=args.external_control_root,
            train_split=args.train_split,
            validation_split=args.validation_split,
            radio_checkpoint=args.radio_checkpoint,
            pfir_dev=args.pfir_dev,
            pfir_test=args.pfir_test,
        )
    elif args.command == "create-manifest":
        result = create_manifest(args)
    elif args.command == "verify-manifest":
        result = verify_manifest(args.manifest)
    elif args.command == "verify-pairing":
        result = verify_cache_pairing(args.manifest, args.output)
    elif args.command == "audit-canary":
        result = audit_canary(args.manifest, args.output)
    elif args.command == "intermediate-binding":
        result = intermediate_binding(args.control_sidecar, args.expected_root)
        if args.output_format == "lines":
            print(result["manifest"]["path"])
            print(result["manifest"]["sha256"])
            return
    elif args.command == "legacy-replay-authority":
        result = write_legacy_replay_authority(
            args.manifest, args.shard, args.output
        )
        if args.output_format == "lines":
            print(result["path"])
            print(result["sha256"])
            return
    else:
        result = finalize(args.manifest, args.pairing, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
