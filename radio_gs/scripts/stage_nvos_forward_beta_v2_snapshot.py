#!/usr/bin/env python3
"""Stage the independent NVOS Forward-Beta-v2 source and run authority.

The reliability cohort is fully validated exactly once while a new snapshot is
staged.  Snapshot resume checks are structural and byte-bound; scene execution
then revalidates only the selected scene's cache/report pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.scripts.bind_evaluation_protocol_freeze import (
    BindingError,
    write_binding_receipt,
)
from radio_gs.scripts.bind_nvos_beta_v2_reliability_manifest import (
    ORDERED_SCENES,
    validate_manifest_payload,
)
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    canonical_json_sha256,
    validate_authority_payload,
)
from radio_gs.scripts.bind_nvos_forward_beta_v2_protocol_authority import (
    BetaV2AuthorityError,
    CANDIDATE_ID,
    EXPECTED_METHOD_NAMESPACE,
    FORWARD_MODE,
    RELIABILITY_MARKER,
    build_v2_authority,
    load_candidate_contract,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _candidate_method_manifest_contract,
)
from radio_gs.scripts.stage_nvos_forward_beta_coverage_v1_snapshot import (
    HOST_MEMORY_POLICY,
    LEGACY_CLOSURE_COMPATIBILITY_RELATIVE,
    MAXIMUM_CONCURRENT_SCENE_EVALUATORS,
    PHYSICAL_GPU_BINDING,
    SCENE_GPU_ASSIGNMENT,
    SERIAL_SCENE_GPU_PLAN,
    SOURCE_SELECTION,
    StagingError,
    _decode_json,
    _decode_yaml,
    _file_record,
    _file_record_from_bytes,
    _make_readonly,
    _mapping,
    _require,
    _selected_repository_sources,
    _stable_bytes,
    _verify_record,
    _write_exclusive,
)


SCHEMA_VERSION = 1
ELIGIBILITY = "protocol_authority_bound_non_exact_diagnostic"
ORDERED_TASKS = ORDERED_SCENES
CANDIDATE_RELATIVE = Path(
    "paper/artifacts/nvos_forward_beta_balanced_residual_v2_candidate_20260802.yaml"
)
RELIABILITY_MANIFEST_RELATIVE = Path(
    "paper/artifacts/nvos_forward_beta_balanced_residual_v2_reliability_manifest_20260802.json"
)
AUTHORITY_RECEIPT_RELATIVE = Path(
    "paper/artifacts/nvos_forward_beta_balanced_residual_v2_protocol_authority.json"
)
STAGING_MANIFEST_RELATIVE = Path(
    "paper/artifacts/nvos_forward_beta_balanced_residual_v2_snapshot_staging.json"
)
RUNNER_RELATIVE = Path("radio_gs/scripts/run_nvos_forward_beta_v2_queue.sh")
SCENE_AUTHORITY_RELATIVE = Path(
    "radio_gs/scripts/nvos_forward_beta_v2_scene_authority.py"
)
AGGREGATOR_RELATIVE = Path(
    "radio_gs/scripts/aggregate_nvos_forward_beta_v2_full8_nonexact.py"
)


def validate_candidate_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate a decoded candidate without trusting its declarative namespace."""

    _require(payload.get("schema_version") == 1, "candidate schema differs")
    _require(payload.get("candidate_id") == CANDIDATE_ID, "candidate id differs")
    _require(payload.get("status") == ELIGIBILITY, "candidate status differs")
    _require(payload.get("promoted") is False, "candidate cannot be promoted")
    _require(
        payload.get("v1_result_or_receipt_reuse_permitted") is False,
        "candidate must forbid v1 result or receipt reuse",
    )
    eligibility = _mapping(payload.get("eligibility"), "candidate eligibility")
    _require(
        eligibility.get("protocol_authority_bound_non_exact_diagnostic") is True,
        "candidate diagnostic eligibility differs",
    )
    for key in (
        "strict_unseen_eligible",
        "strict_unseen_protocol_exact_match",
        "frozen_diagnostic_eligible",
        "main_result_eligible",
        "target_metric_may_select_method_stage_or_continuation",
    ):
        _require(eligibility.get(key) is False, f"candidate {key} must be false")
    cohort = _mapping(payload.get("cohort"), "candidate cohort")
    _require(cohort.get("benchmark") == "NVOS", "candidate benchmark differs")
    _require(
        cohort.get("ordered_tasks") == list(ORDERED_TASKS),
        "candidate cohort differs",
    )
    _require(
        cohort.get("execution") == "fixed_full_eight_without_metric_continuation",
        "candidate execution differs",
    )
    _require(
        cohort.get("aggregation") == "equal_weight_macro_over_8_tasks",
        "candidate aggregation differs",
    )
    reuse = _mapping(payload.get("protocol_reuse"), "protocol reuse")
    for key in (
        "dataset_exact",
        "cohort_exact",
        "prompt_exact",
        "visibility_exact",
        "calibration_policy_exact",
        "aggregation_exact",
        "resize_exact",
        "threshold_exact",
    ):
        _require(reuse.get(key) is True, f"protocol reuse {key} differs")
    for key in (
        "score_semantics_exact",
        "prediction_representation_exact",
        "overall_protocol_exact",
        "target_metric_controls_continuation",
    ):
        _require(reuse.get(key) is False, f"protocol reuse {key} must be false")
    _require(reuse.get("target_mask_use") == "scoring_only", "target mask role differs")
    namespace = dict(_mapping(payload.get("method_namespace"), "method namespace"))
    _require(
        namespace == EXPECTED_METHOD_NAMESPACE,
        "candidate namespace differs from the fixed v2 invocation",
    )
    method = _candidate_method_manifest_contract(argparse.Namespace(**namespace))
    forward = _mapping(method.get("registered_forward_unary"), "forward unary")
    _require(forward.get("mode") == FORWARD_MODE, "derived forward mode differs")
    _require(
        method.get("canonical_reliability_cache") == RELIABILITY_MARKER,
        "derived reliability source marker differs",
    )
    return method, canonical_json_sha256(method)


def load_and_validate_candidate(
    path: str | Path,
) -> tuple[Mapping[str, Any], dict[str, Any], str, bytes]:
    candidate_path = Path(path)
    encoded = _stable_bytes(candidate_path, label="Forward-Beta-v2 candidate YAML")
    payload = _decode_yaml(encoded, label="Forward-Beta-v2 candidate YAML")
    method, method_sha256 = validate_candidate_payload(payload)
    return payload, method, method_sha256, encoded


def _validate_bound_authority(
    authority: Mapping[str, Any],
    *,
    method_sha256: str,
    scoring_contract: Mapping[str, Any],
) -> None:
    validate_authority_payload(authority)
    candidate = _mapping(authority.get("candidate"), "authority candidate")
    _require(
        candidate.get("method_contract_sha256") == method_sha256,
        "authority method SHA differs",
    )
    _require(
        authority.get("scoring_contract") == scoring_contract,
        "authority scoring contract differs",
    )
    _require(
        authority.get("strict_unseen_protocol_exact_match") is False
        and authority.get("strict_unseen_exact_match_blockers")
        == ["score_semantics_differs", "prediction_representation_differs"],
        "v2 authority must remain strict-unseen non-exact",
    )


def publish_authority_receipt(
    *,
    repo_root: str | Path,
    output: str | Path,
    candidate: str | Path,
    reliability_manifest: str | Path,
) -> tuple[dict[str, Any], bytes, dict[str, object], dict[str, Any], str]:
    """Fully validate v2 inputs, then atomically publish/reuse exact authority."""

    try:
        authority, reliability_record, method, method_sha256 = build_v2_authority(
            repo_root=repo_root,
            candidate=candidate,
            reliability_manifest=reliability_manifest,
        )
    except BetaV2AuthorityError as error:
        raise StagingError(str(error)) from error
    forward = _mapping(method.get("registered_forward_unary"), "forward unary")
    scoring = _mapping(forward.get("scoring_adapter"), "scoring adapter")
    _validate_bound_authority(
        authority,
        method_sha256=method_sha256,
        scoring_contract=scoring,
    )
    destination = Path(output)
    if os.path.lexists(destination):
        encoded = _stable_bytes(destination, label="existing v2 authority receipt")
        existing = _decode_json(encoded, label="existing v2 authority receipt")
        _require(existing == authority, "existing immutable v2 authority differs")
        return dict(existing), encoded, reliability_record, method, method_sha256
    try:
        write_binding_receipt(destination, authority)
    except BindingError as error:
        raise StagingError(str(error)) from error
    encoded = _stable_bytes(destination, label="published v2 authority receipt")
    published = _decode_json(encoded, label="published v2 authority receipt")
    _require(published == authority, "published v2 authority changed")
    return authority, encoded, reliability_record, method, method_sha256


def stage_snapshot(
    *,
    repo_root: str | Path,
    snapshot_root: str | Path,
    authority_receipt: str | Path,
    reliability_manifest: str | Path,
) -> dict[str, Any]:
    """Create a new immutable v2 snapshot after one full reliability validation."""

    root = Path(os.path.abspath(os.fspath(repo_root)))
    snapshot = Path(os.path.abspath(os.fspath(snapshot_root)))
    _require(root.is_dir() and not root.is_symlink(), "repository root is unsafe")
    _require(not os.path.lexists(snapshot), "snapshot root must not already exist")
    _require(
        snapshot != root and root not in snapshot.parents,
        "snapshot must be outside the live repository",
    )
    candidate_path = root / CANDIDATE_RELATIVE
    candidate_payload, local_method, local_method_sha, candidate_bytes = (
        load_and_validate_candidate(candidate_path)
    )
    reliability_path = Path(reliability_manifest).absolute()
    authority, authority_bytes, reliability_record, method, method_sha = (
        publish_authority_receipt(
            repo_root=root,
            output=authority_receipt,
            candidate=candidate_path,
            reliability_manifest=reliability_path,
        )
    )
    _require(
        method == local_method and method_sha == local_method_sha,
        "candidate changed during authority publication",
    )
    reliability_bytes = _stable_bytes(
        reliability_path, label="validated reliability manifest"
    )
    _require(
        reliability_record
        == {
            "path": str(reliability_path),
            **_file_record_from_bytes(reliability_bytes),
        },
        "reliability manifest changed after full validation",
    )
    compatibility = root / LEGACY_CLOSURE_COMPATIBILITY_RELATIVE
    compatibility_bytes = _stable_bytes(
        compatibility, label="legacy closure compatibility contract"
    )
    sources = _selected_repository_sources(root)
    snapshot.mkdir(parents=False, mode=0o700)
    records: dict[str, dict[str, object]] = {}
    for source in sources:
        relative = source.relative_to(root)
        encoded = _stable_bytes(source, label=f"repository source {relative}")
        _write_exclusive(snapshot / relative, encoded)
        records[str(relative)] = _file_record_from_bytes(encoded)
    extras = {
        CANDIDATE_RELATIVE: candidate_bytes,
        AUTHORITY_RECEIPT_RELATIVE: authority_bytes,
        RELIABILITY_MANIFEST_RELATIVE: reliability_bytes,
        LEGACY_CLOSURE_COMPATIBILITY_RELATIVE: compatibility_bytes,
    }
    for relative, encoded in extras.items():
        _write_exclusive(snapshot / relative, encoded)
        records[str(relative)] = _file_record_from_bytes(encoded)
    files_payload: dict[str, Any] = {
        "selection": [
            *SOURCE_SELECTION,
            str(CANDIDATE_RELATIVE),
            str(AUTHORITY_RECEIPT_RELATIVE),
            str(RELIABILITY_MANIFEST_RELATIVE),
            str(LEGACY_CLOSURE_COMPATIBILITY_RELATIVE),
        ],
        "files": records,
    }
    files_payload["digest"] = canonical_json_sha256(files_payload)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "nvos_forward_beta_v2_source_snapshot_staging",
        "candidate": CANDIDATE_ID,
        "eligibility": ELIGIBILITY,
        "strict_unseen_protocol_exact_match": False,
        "frozen_diagnostic_eligible": False,
        "main_result_eligible": False,
        "v1_result_or_receipt_reuse_permitted": False,
        "snapshot_root": str(snapshot),
        "source_repository_root": str(root),
        "dependency_dag": [
            "evaluator_derived_v2_method_contract",
            "query_independent_reliability_manifest_full_validation",
            "candidate_method_contract_sha256",
            "protocol_authority_receipt",
            "readonly_source_snapshot",
            "run_manifest",
        ],
        "target_data_read": False,
        "gpu_state_read": False,
        "ordered_tasks": list(ORDERED_TASKS),
        "execution": "fixed_full_eight_without_metric_continuation",
        "maximum_concurrent_scene_evaluators": MAXIMUM_CONCURRENT_SCENE_EVALUATORS,
        "host_memory_policy": HOST_MEMORY_POLICY,
        "serial_scene_gpu_plan": json.loads(json.dumps(SERIAL_SCENE_GPU_PLAN)),
        "candidate_contract": {
            "path": str(CANDIDATE_RELATIVE),
            **_file_record_from_bytes(candidate_bytes),
        },
        "method_contract": method,
        "candidate_method_contract_sha256": method_sha,
        "protocol_authority_receipt": {
            "path": str(AUTHORITY_RECEIPT_RELATIVE),
            **_file_record_from_bytes(authority_bytes),
        },
        "protocol_authority": authority,
        "protocol_authority_canonical_json_sha256": canonical_json_sha256(authority),
        "reliability_cache_manifest": reliability_record,
        "staged_reliability_cache_manifest": {
            "path": str(RELIABILITY_MANIFEST_RELATIVE),
            **_file_record_from_bytes(reliability_bytes),
        },
        "reliability_manifest_full_validation_count": 1,
        "snapshot_files": files_payload,
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_exclusive(snapshot / STAGING_MANIFEST_RELATIVE, encoded_manifest)
    _make_readonly(snapshot)
    validate_snapshot(snapshot)
    return manifest


def validate_snapshot(snapshot_root: str | Path) -> dict[str, Any]:
    """Revalidate immutable snapshot bytes without rehashing 17GB sidecars."""

    snapshot = Path(os.path.abspath(os.fspath(snapshot_root)))
    root_info = os.stat(snapshot, follow_symlinks=False)
    _require(stat.S_ISDIR(root_info.st_mode), "snapshot root must be a real directory")
    manifest_bytes = _stable_bytes(
        snapshot / STAGING_MANIFEST_RELATIVE, label="v2 snapshot staging manifest"
    )
    manifest = dict(_decode_json(manifest_bytes, label="v2 snapshot staging manifest"))
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "staging schema differs")
    _require(
        manifest.get("artifact_type")
        == "nvos_forward_beta_v2_source_snapshot_staging",
        "staging artifact type differs",
    )
    _require(manifest.get("candidate") == CANDIDATE_ID, "staging candidate differs")
    _require(manifest.get("snapshot_root") == str(snapshot), "staging root differs")
    _require(
        manifest.get("v1_result_or_receipt_reuse_permitted") is False,
        "staging must forbid v1 result or receipt reuse",
    )
    _require(
        manifest.get("ordered_tasks") == list(ORDERED_TASKS),
        "staging cohort differs",
    )
    _require(
        manifest.get("maximum_concurrent_scene_evaluators")
        == MAXIMUM_CONCURRENT_SCENE_EVALUATORS
        and manifest.get("host_memory_policy") == HOST_MEMORY_POLICY
        and manifest.get("serial_scene_gpu_plan")
        == json.loads(json.dumps(SERIAL_SCENE_GPU_PLAN)),
        "staging resource plan differs",
    )

    candidate_record = _mapping(
        manifest.get("candidate_contract"), "staged candidate record"
    )
    candidate_bytes = _verify_record(
        snapshot / CANDIDATE_RELATIVE,
        {key: candidate_record[key] for key in ("bytes", "sha256")},
        label="staged v2 candidate",
    )
    candidate_payload = _decode_yaml(candidate_bytes, label="staged v2 candidate")
    method, method_sha = validate_candidate_payload(candidate_payload)
    _require(manifest.get("method_contract") == method, "staged method differs")
    _require(
        manifest.get("candidate_method_contract_sha256") == method_sha,
        "staged method SHA differs",
    )
    receipt_record = _mapping(
        manifest.get("protocol_authority_receipt"), "staged authority record"
    )
    receipt_bytes = _verify_record(
        snapshot / AUTHORITY_RECEIPT_RELATIVE,
        {key: receipt_record[key] for key in ("bytes", "sha256")},
        label="staged v2 authority receipt",
    )
    authority = _decode_json(receipt_bytes, label="staged v2 authority receipt")
    scoring = _mapping(
        _mapping(method.get("registered_forward_unary"), "forward unary").get(
            "scoring_adapter"
        ),
        "scoring adapter",
    )
    _validate_bound_authority(
        authority, method_sha256=method_sha, scoring_contract=scoring
    )
    _require(manifest.get("protocol_authority") == authority, "inline authority differs")
    _require(
        manifest.get("protocol_authority_canonical_json_sha256")
        == canonical_json_sha256(authority),
        "inline authority digest differs",
    )

    staged_reliability = _mapping(
        manifest.get("staged_reliability_cache_manifest"),
        "staged reliability manifest record",
    )
    reliability_bytes = _verify_record(
        snapshot / RELIABILITY_MANIFEST_RELATIVE,
        {key: staged_reliability[key] for key in ("bytes", "sha256")},
        label="staged reliability manifest",
    )
    reliability_payload = _decode_json(
        reliability_bytes, label="staged reliability manifest"
    )
    # Structural/canonical validation only: full file verification occurred once
    # before authority publication and is recorded in this immutable staging row.
    validate_manifest_payload(reliability_payload, verify_files=False)
    origin_record = _mapping(
        manifest.get("reliability_cache_manifest"),
        "origin reliability manifest record",
    )
    _require(
        origin_record.get("bytes") == staged_reliability.get("bytes")
        and origin_record.get("sha256") == staged_reliability.get("sha256"),
        "origin/staged reliability manifest bytes differ",
    )
    _require(
        manifest.get("reliability_manifest_full_validation_count") == 1,
        "reliability manifest full-validation count differs",
    )

    files_payload = _mapping(manifest.get("snapshot_files"), "snapshot files")
    declared_files = _mapping(files_payload.get("files"), "snapshot file records")
    expected_relatives = {
        str(path.relative_to(snapshot))
        for path in _selected_repository_sources(snapshot)
    }
    expected_relatives.update(
        {
            str(CANDIDATE_RELATIVE),
            str(AUTHORITY_RECEIPT_RELATIVE),
            str(RELIABILITY_MANIFEST_RELATIVE),
            str(LEGACY_CLOSURE_COMPATIBILITY_RELATIVE),
        }
    )
    _require(set(declared_files) == expected_relatives, "snapshot file selection differs")
    for relative in sorted(expected_relatives):
        record = _mapping(declared_files[relative], f"snapshot record {relative}")
        _verify_record(snapshot / relative, record, label=f"snapshot file {relative}")
    digest_payload = {
        "selection": files_payload.get("selection"),
        "files": dict(declared_files),
    }
    _require(
        files_payload.get("digest") == canonical_json_sha256(digest_payload),
        "snapshot file-set digest differs",
    )

    discovered: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        snapshot, topdown=True, followlinks=False
    ):
        current = Path(directory)
        _require(not current.is_symlink(), f"snapshot directory is symlinked: {current}")
        _require(
            current.stat().st_mode & 0o222 == 0,
            f"snapshot directory is writable: {current}",
        )
        for name in directory_names:
            path = current / name
            info = os.stat(path, follow_symlinks=False)
            _require(
                not path.is_symlink() and stat.S_ISDIR(info.st_mode),
                f"snapshot contains unsafe directory: {path}",
            )
        for name in file_names:
            path = current / name
            info = os.stat(path, follow_symlinks=False)
            _require(
                not path.is_symlink() and stat.S_ISREG(info.st_mode),
                f"snapshot contains unsafe file: {path}",
            )
            _require(info.st_mode & 0o222 == 0, f"snapshot file is writable: {path}")
            discovered.add(str(path.relative_to(snapshot)))
    _require(
        discovered == expected_relatives | {str(STAGING_MANIFEST_RELATIVE)},
        "snapshot disk file set differs",
    )
    return manifest


def _logical_reliability_record(
    reliability_payload: Mapping[str, Any], *, scene: str
) -> dict[str, object]:
    rows = _mapping(reliability_payload.get("scenes"), "reliability scenes")
    row = _mapping(rows.get(scene), f"{scene}: reliability row")
    cache = _mapping(row.get("reliability_cache"), f"{scene}: cache record")
    report = _mapping(row.get("build_report"), f"{scene}: report record")
    _require(
        set(cache) == {"path", "bytes", "sha256"}
        and set(report) == {"path", "bytes", "sha256"},
        f"{scene}: reliability file records differ",
    )
    return {
        "path": cache["path"],
        "bytes": cache["bytes"],
        "sha256": cache["sha256"],
        "metadata_path": report["path"],
        "metadata_sha256": report["sha256"],
    }


def build_run_manifest_payload(
    *,
    snapshot_root: str | Path,
    source_root: str | Path,
    queue_plan: str | Path,
    benchmark_manifest: str | Path,
    radio_checkpoint: str | Path,
    parent_asset_manifest: str | Path,
    output_root: str | Path,
    runner: str | Path,
    thermal_guard: str | Path,
    gpu_authority: str | Path,
    runtime_closure: Mapping[str, Any],
    source_snapshot_authority: Mapping[str, Any],
    thermal_safety_contract: Mapping[str, Any],
    output_identity: Mapping[str, Any],
    reliability_cache_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Build one immutable v2 run manifest without reopening target data."""

    snapshot = Path(snapshot_root).resolve()
    staging = validate_snapshot(snapshot)
    source = Path(source_root).resolve()
    queue_path = Path(queue_plan).resolve()
    benchmark_path = Path(benchmark_manifest).resolve()
    checkpoint_path = Path(radio_checkpoint).resolve()
    parent_path = Path(parent_asset_manifest).resolve()
    runner_path = Path(runner).resolve()
    guard_path = Path(thermal_guard).resolve()
    authority_path = Path(gpu_authority).resolve()

    queue_bytes = _stable_bytes(queue_path, label="NVOS queue plan")
    benchmark_bytes = _stable_bytes(benchmark_path, label="NVOS benchmark manifest")
    queue = _decode_json(queue_bytes, label="NVOS queue plan")
    benchmark = _decode_json(benchmark_bytes, label="NVOS benchmark manifest")
    _require(queue.get("benchmark") == "nvos", "queue benchmark differs")
    _require(
        queue.get("protocol_hash") == benchmark.get("protocol_hash"),
        "queue and benchmark protocol hashes differ",
    )
    _require(
        [str(row.get("scene_id")) for row in queue.get("scenes", [])]
        == list(ORDERED_TASKS),
        "queue does not contain the fixed ordered eight-task cohort",
    )

    parent_bytes = _stable_bytes(parent_path, label="parent asset manifest")
    parent = _decode_json(parent_bytes, label="parent asset manifest")
    _require(parent.get("candidate") == "registered-region-v2", "parent candidate differs")
    _require(parent.get("scenes") == list(ORDERED_TASKS), "parent cohort differs")
    _require(
        Path(str(parent.get("source_root", ""))).resolve() == source,
        "parent source root differs",
    )
    _require(
        Path(str(parent.get("queue_plan", ""))).resolve() == queue_path
        and parent.get("queue_plan_sha256") == hashlib.sha256(queue_bytes).hexdigest(),
        "parent queue binding differs",
    )
    _require(
        Path(str(parent.get("benchmark_manifest", ""))).resolve() == benchmark_path
        and parent.get("benchmark_manifest_sha256")
        == hashlib.sha256(benchmark_bytes).hexdigest(),
        "parent benchmark binding differs",
    )
    checkpoint_record = _file_record(checkpoint_path, label="RADIO checkpoint")
    _require(
        Path(str(parent.get("radio_checkpoint", ""))).resolve() == checkpoint_path
        and parent.get("radio_checkpoint_sha256") == checkpoint_record["sha256"],
        "parent RADIO binding differs",
    )

    source_artifacts = json.loads(json.dumps(parent.get("source_artifacts")))
    queue_scene_inputs = json.loads(json.dumps(parent.get("queue_scene_inputs")))
    _require(isinstance(source_artifacts, dict), "parent source artifacts are absent")
    _require(isinstance(queue_scene_inputs, dict), "parent queue inputs are absent")
    for scene in ORDERED_TASKS:
        _require(scene in source_artifacts, f"{scene}: parent sources are absent")
        _require(scene in queue_scene_inputs, f"{scene}: parent queue inputs are absent")
        for name, raw_record in source_artifacts[scene].items():
            record = _mapping(raw_record, f"{scene} source {name}")
            artifact = Path(str(record.get("path", ""))).resolve()
            metadata = Path(str(record.get("metadata_path", ""))).resolve()
            actual = _file_record(artifact, label=f"{scene} source {name}")
            _require(
                actual["bytes"] == record.get("bytes")
                and actual["sha256"] == record.get("sha256"),
                f"{scene}: source record differs for {name}",
            )
            _require(
                _file_record(metadata, label=f"{scene} metadata {name}")["sha256"]
                == record.get("metadata_sha256"),
                f"{scene}: metadata digest differs for {name}",
            )
        for raw_path, raw_record in queue_scene_inputs[scene].items():
            record = _mapping(raw_record, f"{scene} queue input")
            actual = _file_record(Path(raw_path).resolve(), label=f"{scene} queue input")
            _require(actual["bytes"] == record.get("bytes"), f"{scene}: queue input size differs")
            if record.get("sha256") is not None:
                _require(
                    actual["sha256"] == record.get("sha256"),
                    f"{scene}: queue input digest differs",
                )

    origin_record = dict(
        _mapping(staging.get("reliability_cache_manifest"), "reliability manifest")
    )
    if reliability_cache_manifest is not None:
        requested_reliability = Path(reliability_cache_manifest).resolve()
        requested_record = _file_record(
            requested_reliability, label="requested reliability manifest"
        )
        _require(
            requested_record.get("bytes") == origin_record.get("bytes")
            and requested_record.get("sha256") == origin_record.get("sha256"),
            "requested reliability manifest bytes differ from staged authority",
        )
    origin_path = Path(str(origin_record.get("path", ""))).absolute()
    observed_origin = {
        "path": str(origin_path),
        **_file_record(origin_path, label="origin reliability manifest"),
    }
    _require(observed_origin == origin_record, "origin reliability manifest changed")
    staged_rel_path = snapshot / RELIABILITY_MANIFEST_RELATIVE
    staged_rel_bytes = _stable_bytes(staged_rel_path, label="staged reliability manifest")
    _require(
        hashlib.sha256(staged_rel_bytes).hexdigest() == origin_record.get("sha256")
        and len(staged_rel_bytes) == origin_record.get("bytes"),
        "staged/origin reliability manifest differs",
    )
    reliability_payload = _decode_json(
        staged_rel_bytes, label="staged reliability manifest"
    )
    validate_manifest_payload(reliability_payload, verify_files=False)
    for scene in ORDERED_TASKS:
        logical = _logical_reliability_record(reliability_payload, scene=scene)
        _require(
            "canonical_primitive_reliability_v1.pt"
            not in source_artifacts[scene],
            f"{scene}: parent unexpectedly supplies v2 reliability authority",
        )
        source_artifacts[scene][
            "canonical_primitive_reliability_v1.pt"
        ] = logical

    runtime_sources = _mapping(
        _mapping(runtime_closure.get("repository_sources"), "runtime sources").get(
            "files"
        ),
        "runtime source records",
    )
    implementation = {
        relative: str(_mapping(record, relative).get("sha256"))
        for relative, record in runtime_sources.items()
    }
    required_v2_sources = {
        "radio_gs/scripts/eval_nvos_gaussian_first.py",
        "radio_gs/querying/evidence_scorer.py",
        "radio_gs/rendering/contribution_compositor.py",
        "radio_gs/scripts/bind_nvos_forward_beta_protocol_authority.py",
        "radio_gs/scripts/bind_nvos_forward_beta_v2_protocol_authority.py",
        "radio_gs/scripts/bind_nvos_beta_v2_reliability_manifest.py",
        "radio_gs/scripts/stage_nvos_forward_beta_v2_snapshot.py",
        "radio_gs/scripts/nvos_forward_beta_v2_scene_authority.py",
        "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
        "radio_gs/scripts/validate_evaluation_protocol_freeze.py",
        "radio_gs/interfaces/capability_cache.py",
        "radio_gs/field/primitive_reliability.py",
        "radio_gs/scripts/build_canonical_reliability_cache.py",
    }
    _require(
        required_v2_sources <= set(implementation),
        "runtime closure lacks v2 implementation authority",
    )

    receipt_record = dict(
        _mapping(staging.get("protocol_authority_receipt"), "authority receipt")
    )
    candidate_record = dict(
        _mapping(staging.get("candidate_contract"), "candidate contract")
    )
    staging_path = snapshot / STAGING_MANIFEST_RELATIVE
    output = Path(output_root).resolve()
    payload: dict[str, Any] = {
        "schema_version": 3,
        "candidate": CANDIDATE_ID,
        "eligibility": ELIGIBILITY,
        "strict_unseen_protocol_exact_match": False,
        "frozen_diagnostic_eligible": False,
        "main_result_eligible": False,
        "target_metric_controls_continuation": False,
        "v1_result_or_receipt_reuse_permitted": False,
        "execution": "fixed_full_eight_without_metric_continuation",
        "maximum_concurrent_scene_evaluators": MAXIMUM_CONCURRENT_SCENE_EVALUATORS,
        "host_memory_policy": HOST_MEMORY_POLICY,
        "serial_scene_gpu_plan": json.loads(json.dumps(SERIAL_SCENE_GPU_PLAN)),
        "scenes": list(ORDERED_TASKS),
        "scene_gpu_assignment": json.loads(json.dumps(SCENE_GPU_ASSIGNMENT)),
        "physical_gpu_binding": PHYSICAL_GPU_BINDING,
        "source_snapshot_root": str(snapshot),
        "source_snapshot_import_root": runtime_closure.get("repository_import_root"),
        "source_snapshot_tree_sha256": _mapping(
            runtime_closure.get("repository_sources"), "runtime sources"
        ).get("digest"),
        "source_snapshot_permissions": dict(source_snapshot_authority),
        "runtime_closure_sha256": runtime_closure.get("digest"),
        "source_snapshot_staging_manifest": {
            "path": str(staging_path),
            **_file_record(staging_path, label="v2 staging manifest"),
        },
        "candidate_contract": {
            "path": str(snapshot / CANDIDATE_RELATIVE),
            "bytes": candidate_record.get("bytes"),
            "sha256": candidate_record.get("sha256"),
        },
        "method_contract": staging.get("method_contract"),
        "candidate_method_contract_sha256": staging.get(
            "candidate_method_contract_sha256"
        ),
        "method_contract_sha256": staging.get("candidate_method_contract_sha256"),
        "protocol_authority_receipt": {
            "path": str(snapshot / AUTHORITY_RECEIPT_RELATIVE),
            "bytes": receipt_record.get("bytes"),
            "sha256": receipt_record.get("sha256"),
        },
        "registered_forward_protocol_authority": staging.get("protocol_authority"),
        "registered_forward_protocol_authority_sha256": staging.get(
            "protocol_authority_canonical_json_sha256"
        ),
        "protocol_reuse": _decode_yaml(
            _stable_bytes(snapshot / CANDIDATE_RELATIVE, label="v2 candidate"),
            label="v2 candidate",
        ).get("protocol_reuse"),
        "reliability_cache_manifest": origin_record,
        "source_root": str(source),
        "source_artifacts": source_artifacts,
        "queue_scene_inputs": queue_scene_inputs,
        "queue_plan": str(queue_path),
        "queue_plan_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "radio_checkpoint": str(checkpoint_path),
        "radio_checkpoint_sha256": checkpoint_record["sha256"],
        "asset_manifest_parent": str(parent_path),
        "asset_manifest_parent_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "asset_reuse_contract": "parent_sha256_plus_path_size_and_sha256_preflight",
        "output_root": str(output),
        "output_identity": dict(output_identity),
        "runner": str(runner_path),
        "runner_sha256": _file_record(runner_path, label="v2 runner")["sha256"],
        "thermal_safety_contract": dict(thermal_safety_contract),
        "gpu_authority": {
            "path": str(authority_path),
            **_file_record(authority_path, label="GPU authority"),
        },
        "scene_receipt_contract": {
            "artifact_type": "nvos-forward-beta-v2-scene-receipt-v1",
            "receipt_root": str(output / "scene_receipts"),
            "attempt_root": str(output / "scene_attempts"),
            "skip_only_after_full_receipt_revalidation": True,
            "aggregate_requires_all_eight_scene_receipts": True,
            "metric_based_continuation": False,
            "v1_receipt_accepted": False,
        },
        "implementation_sources": implementation,
        "runtime_closure": dict(runtime_closure),
    }
    return payload


def write_run_manifest(output: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(output)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if os.path.lexists(destination):
        existing = _stable_bytes(destination, label="existing v2 run manifest")
        _require(existing == encoded, "existing immutable v2 run manifest differs")
        return destination
    _write_exclusive(destination, encoded)
    destination.chmod(0o444)
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--repo-root", type=Path, required=True)
    stage.add_argument("--snapshot-root", type=Path, required=True)
    stage.add_argument("--authority-receipt", type=Path, required=True)
    stage.add_argument("--reliability-manifest", type=Path, required=True)
    validate = commands.add_parser("validate-snapshot")
    validate.add_argument("--snapshot-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "stage":
        payload = stage_snapshot(
            repo_root=args.repo_root,
            snapshot_root=args.snapshot_root,
            authority_receipt=args.authority_receipt,
            reliability_manifest=args.reliability_manifest,
        )
    else:
        payload = validate_snapshot(args.snapshot_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
