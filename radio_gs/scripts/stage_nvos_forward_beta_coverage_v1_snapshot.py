#!/usr/bin/env python3
"""Stage and bind the source snapshot for the NVOS forward-Beta candidate.

This module is deliberately CPU-only.  It derives the candidate method
contract from the evaluator helper, creates the protocol-authority receipt in
the complete main repository, and only then copies that immutable receipt into
a new read-only source snapshot.  The snapshot and run manifests bind the
receipt by both its canonical payload digest and its exact file digest.

The dependency direction is therefore fixed and acyclic::

    evaluator method contract
      -> method-contract SHA256
      -> protocol-authority receipt
      -> read-only source snapshot
      -> run manifest

Target RGB, target masks, metrics, and GPU state are outside this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from radio_gs.scripts.bind_evaluation_protocol_freeze import (
    BindingError,
    write_binding_receipt,
)
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    AuthorityError,
    build_authority,
    canonical_json_sha256,
    validate_authority_payload,
    _read_stable_bytes,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _candidate_method_manifest_contract,
)


SCHEMA_VERSION = 1
CANDIDATE_ID = "nvos-forward-beta-coverage-v1"
ELIGIBILITY = "protocol_authority_bound_non_exact_diagnostic"
ORDERED_TASKS = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
SCENE_GPU_ASSIGNMENT = {
    "policy": "fixed_before_execution_no_target_metric_input",
    "gpu0": ["fern", "flower", "fortress", "horns_center"],
    "gpu1": ["horns_left", "leaves", "orchids", "trex"],
}
PHYSICAL_GPU_BINDING = (
    "runtime_inventory_index_uuid_pci_under_independent_flock"
)
MAXIMUM_CONCURRENT_SCENE_EVALUATORS = 1
HOST_MEMORY_POLICY = "fixed_mapping_single_scene_resident_v1"
SERIAL_SCENE_GPU_PLAN = (
    {"physical_gpu": 0, "scene": "fern"},
    {"physical_gpu": 1, "scene": "horns_left"},
    {"physical_gpu": 0, "scene": "flower"},
    {"physical_gpu": 1, "scene": "leaves"},
    {"physical_gpu": 0, "scene": "fortress"},
    {"physical_gpu": 1, "scene": "orchids"},
    {"physical_gpu": 0, "scene": "horns_center"},
    {"physical_gpu": 1, "scene": "trex"},
)
CANDIDATE_RELATIVE = Path(
    "paper/artifacts/nvos_forward_beta_coverage_v1_candidate_20260802.yaml"
)
AUTHORITY_RECEIPT_RELATIVE = Path(
    "paper/artifacts/nvos_forward_beta_coverage_v1_protocol_authority.json"
)
STAGING_MANIFEST_RELATIVE = Path(
    "paper/artifacts/nvos_forward_beta_coverage_v1_snapshot_staging.json"
)
# The present GPU closure authority still selects this historical non-package
# input.  Keeping it in the snapshot preserves closure compatibility without
# changing the old v3 runner or giving that contract any candidate authority.
LEGACY_CLOSURE_COMPATIBILITY_RELATIVE = Path(
    "paper/artifacts/nvos_registered_region_v3_candidate_20260731.yaml"
)
SOURCE_SELECTION = ("radio_gs/**/*.py", "radio_gs/**/*.sh")


class StagingError(ValueError):
    """Raised when a candidate, receipt, or source snapshot is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StagingError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StagingError(f"{label} must be a mapping")
    return value


def _sha256_bytes(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _stable_bytes(path: Path, *, label: str) -> bytes:
    try:
        return _read_stable_bytes(
            Path(os.path.abspath(os.fspath(path))), label=label
        )
    except AuthorityError as error:
        raise StagingError(str(error)) from error


def _decode_json(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagingError(f"{label} is not valid UTF-8 JSON") from error
    return _mapping(value, label)


def _decode_yaml(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(encoded.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise StagingError(f"{label} is not valid UTF-8 YAML") from error
    return _mapping(value, label)


def _file_record_from_bytes(encoded: bytes) -> dict[str, object]:
    return {"bytes": len(encoded), "sha256": _sha256_bytes(encoded)}


def _file_record(path: Path, *, label: str) -> dict[str, object]:
    return _file_record_from_bytes(_stable_bytes(path, label=label))


def _namespace_from_method_contract(
    method: Mapping[str, Any],
) -> argparse.Namespace:
    """Map the declarative candidate YAML to the evaluator's CLI namespace."""

    prompt = _mapping(method.get("prompt_registration"), "prompt registration")
    graph = _mapping(method.get("graph"), "graph contract")
    render = _mapping(method.get("score_render"), "score render contract")
    solver = _mapping(method.get("solver"), "solver contract")
    forward = _mapping(method.get("registered_forward_unary"), "forward unary")
    return argparse.Namespace(
        support_mode=method.get("support_mode"),
        region_space=method.get("region_space"),
        prompt_registration_mode=prompt.get("mode"),
        prompt_registration_scale=prompt.get("scale"),
        alpha_threshold=prompt.get("alpha_threshold"),
        depth_tolerance=prompt.get("depth_tolerance"),
        relative_depth_tolerance=prompt.get("relative_depth_tolerance"),
        registered_seed_construction=method.get("seed_construction"),
        registered_observation_fusion=method.get("observation_fusion"),
        registered_seed_unary_weight=method.get("registered_seed_unary_weight"),
        registered_observation_confidence=method.get("observation_confidence"),
        registered_observation_mass_scale=method.get("observation_mass_scale"),
        registered_observation_coverage_power=method.get(
            "observation_coverage_power"
        ),
        support_threshold=method.get("prompt_support_threshold"),
        prototype_count=method.get("prototype_count"),
        prototype_strategy=method.get("prototype_strategy"),
        appearance_weight=method.get("appearance_weight"),
        boundary_weight=method.get("boundary_weight"),
        prototype_temperature=method.get("prototype_temperature"),
        feature_calibration=method.get("feature_calibration"),
        background_centroids=method.get("background_centroids"),
        score_calibration=method.get("score_calibration"),
        negative_spatial_mode=method.get("negative_spatial_mode"),
        registered_selection_mode=method.get("diagnostic_selection_mode"),
        registered_readout_stage=method.get("final_readout"),
        registered_forward_unary=forward.get("mode"),
        graph_policy=graph.get("policy"),
        component_graph_policy=graph.get("component_policy"),
        graph_legacy_residual=graph.get("legacy_residual"),
        channel_confidence_mode=graph.get("channel_confidence_mode"),
        score_render_resolution=render.get("resolution"),
        score_render_scale=render.get("scale"),
        valid_support_normalization=render.get("valid_support_normalization"),
        valid_support_coverage_power=render.get("valid_support_coverage_power"),
        feature_contribution_gamma=render.get("feature_contribution_gamma"),
        score_chunk_size=render.get("score_chunk_size"),
        solver_type=solver.get("type"),
        solver_iterations=solver.get("iterations"),
        solver_residual=solver.get("residual"),
        solver_unary_temperature=solver.get("unary_temperature"),
        solver_support_threshold=solver.get("support_threshold"),
        laplacian_weight=solver.get("laplacian_weight"),
        cg_iterations=solver.get("cg_iterations"),
        cg_tolerance=solver.get("cg_tolerance"),
        hard_seed_threshold=solver.get("hard_seed_threshold"),
        hard_seed_conflict_policy=solver.get("hard_seed_conflict_policy"),
        hard_seed_conflict_margin=solver.get("hard_seed_conflict_margin"),
        component_edge_threshold=solver.get("component_edge_threshold"),
        seeded_component_min_weight=solver.get("seeded_component_min_weight"),
        canonical_reliability_cache=method.get("canonical_reliability_cache"),
        diagnostic_graph_affinity_override=method.get(
            "diagnostic_graph_affinity_override"
        ),
        require_asset_hashes=method.get("asset_hash_verification_required"),
    )


def _fixed_candidate_namespace() -> argparse.Namespace:
    """Return the immutable evaluator invocation baseline, independent of YAML."""

    return argparse.Namespace(
        support_mode="canonical_support",
        region_space="sam3",
        prompt_registration_mode="raster_adjoint",
        prompt_registration_scale=1.0,
        alpha_threshold=0.0,
        depth_tolerance=0.08,
        relative_depth_tolerance=0.02,
        registered_seed_construction="joint_signed",
        registered_observation_fusion="probability_mixture",
        registered_seed_unary_weight=0.0,
        registered_observation_confidence="poisson_mass_coverage",
        registered_observation_mass_scale=1.0,
        registered_observation_coverage_power=1.0,
        support_threshold=0.0,
        prototype_count=4,
        prototype_strategy="spherical_mean_fps",
        appearance_weight=1.0,
        boundary_weight=0.35,
        prototype_temperature=0.07,
        feature_calibration="none",
        background_centroids=0,
        score_calibration="none",
        negative_spatial_mode="none",
        registered_selection_mode="seeded_component",
        registered_readout_stage="propagated",
        registered_forward_unary="beta_coverage_v1",
        graph_policy="legacy",
        component_graph_policy="same",
        graph_legacy_residual=0.0,
        channel_confidence_mode="none",
        score_render_resolution="prompt_native",
        score_render_scale=1.0,
        valid_support_normalization=True,
        valid_support_coverage_power=1.0,
        feature_contribution_gamma=1.0,
        score_chunk_size=8192,
        solver_type="confidence_random_walker",
        solver_iterations=12,
        solver_residual=0.30,
        solver_unary_temperature=0.10,
        solver_support_threshold=0.50,
        laplacian_weight=1.0,
        cg_iterations=64,
        cg_tolerance=1e-5,
        hard_seed_threshold=0.20,
        hard_seed_conflict_policy="exclusive_relative",
        hard_seed_conflict_margin=0.0,
        component_edge_threshold=1e-5,
        seeded_component_min_weight=0.20,
        canonical_reliability_cache="",
        diagnostic_graph_affinity_override="",
        require_asset_hashes=True,
    )


def validate_candidate_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, object], str]:
    """Validate fixed eligibility/cohort and rederive the method contract."""

    _require(payload.get("schema_version") == 1, "candidate schema differs")
    _require(payload.get("candidate_id") == CANDIDATE_ID, "candidate id differs")
    _require(payload.get("status") == ELIGIBILITY, "candidate status differs")
    _require(payload.get("promoted") is False, "candidate cannot be promoted")

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
        "candidate must execute the fixed ordered eight-task cohort",
    )
    _require(
        cohort.get("execution") == "fixed_full_eight_without_metric_continuation",
        "candidate execution may not use metric continuation",
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

    declared_method = dict(_mapping(payload.get("method_contract"), "method contract"))
    derived = _candidate_method_manifest_contract(_fixed_candidate_namespace())
    _require(
        declared_method == derived,
        "candidate method contract differs from evaluator-derived contract",
    )
    forward = _mapping(derived.get("registered_forward_unary"), "derived forward unary")
    scoring = _mapping(forward.get("scoring_adapter"), "derived scoring adapter")
    _require(
        scoring
        == {
            "score_semantics": "beta_centered_posterior",
            "prediction_representation": "continuous_beta_centered_posterior",
            "threshold": {"comparison": "greater_or_equal", "value": 0.0},
            "resize": "nearest",
        },
        "derived forward-Beta scoring contract differs",
    )
    return derived, canonical_json_sha256(derived)


def load_and_validate_candidate(
    path: str | Path,
) -> tuple[Mapping[str, Any], dict[str, object], str, bytes]:
    candidate_path = Path(path)
    encoded = _stable_bytes(candidate_path, label="forward-Beta candidate YAML")
    payload = _decode_yaml(encoded, label="forward-Beta candidate YAML")
    method, method_sha256 = validate_candidate_payload(payload)
    return payload, method, method_sha256, encoded


def _validate_bound_authority(
    authority: Mapping[str, Any],
    *,
    method_sha256: str,
    scoring_contract: Mapping[str, Any],
) -> None:
    try:
        validate_authority_payload(authority)
    except AuthorityError as error:
        raise StagingError(str(error)) from error
    candidate = _mapping(authority.get("candidate"), "authority candidate")
    _require(
        candidate.get("method_contract_sha256") == method_sha256,
        "authority method SHA differs from the evaluator-derived method",
    )
    _require(
        authority.get("scoring_contract") == scoring_contract,
        "authority scoring contract differs from the evaluator-derived adapter",
    )
    _require(
        authority.get("strict_unseen_protocol_exact_match") is False,
        "forward-Beta authority must remain strict-unseen non-exact",
    )
    _require(
        authority.get("strict_unseen_exact_match_blockers")
        == ["score_semantics_differs", "prediction_representation_differs"],
        "forward-Beta non-exact blockers differ",
    )
    comparator = _mapping(
        authority.get("external_comparator_provenance"), "comparator provenance"
    )
    _require(
        comparator.get("candidate_binding")
        == {
            "canonical_task_id": None,
            "registry_row": None,
            "promptable_registry_row": None,
        },
        "external LUDVIG comparator cannot become the candidate binding",
    )


def publish_authority_receipt(
    *,
    repo_root: str | Path,
    output: str | Path,
    method_contract: Mapping[str, Any],
    method_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Build in the complete repo and immutably publish/reuse one receipt."""

    forward = _mapping(
        method_contract.get("registered_forward_unary"), "forward unary contract"
    )
    scoring = _mapping(forward.get("scoring_adapter"), "forward scoring adapter")
    try:
        authority = build_authority(
            candidate_method_sha256=method_sha256,
            scoring_contract=scoring,
            repo_root=repo_root,
        )
    except AuthorityError as error:
        raise StagingError(str(error)) from error
    _validate_bound_authority(
        authority, method_sha256=method_sha256, scoring_contract=scoring
    )

    destination = Path(output)
    if os.path.lexists(destination):
        encoded = _stable_bytes(destination, label="existing authority receipt")
        existing = _decode_json(encoded, label="existing authority receipt")
        _require(existing == authority, "existing immutable authority receipt differs")
        return dict(existing), encoded
    try:
        write_binding_receipt(destination, authority)
    except BindingError as error:
        raise StagingError(str(error)) from error
    encoded = _stable_bytes(destination, label="published authority receipt")
    published = _decode_json(encoded, label="published authority receipt")
    _require(published == authority, "published authority receipt changed")
    return authority, encoded


def _selected_repository_sources(repo_root: Path) -> list[Path]:
    package_root = repo_root / "radio_gs"
    _require(package_root.is_dir() and not package_root.is_symlink(), "radio_gs source root is unsafe")
    selected: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        package_root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        for name in list(directory_names):
            child = current / name
            if child.is_symlink():
                raise StagingError(f"source selection refuses symlink directory: {child}")
        for name in file_names:
            path = current / name
            if path.suffix not in {".py", ".sh"}:
                continue
            if path.is_symlink():
                raise StagingError(f"source selection refuses symlink file: {path}")
            selected.append(path)
    return sorted(selected, key=lambda path: str(path.relative_to(repo_root)))


def _write_exclusive(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _make_readonly(snapshot_root: Path) -> None:
    for directory, directory_names, file_names in os.walk(
        snapshot_root, topdown=False, followlinks=False
    ):
        current = Path(directory)
        for name in file_names:
            path = current / name
            _require(not path.is_symlink(), f"snapshot contains symlink: {path}")
            path.chmod(0o444)
        for name in directory_names:
            path = current / name
            _require(not path.is_symlink(), f"snapshot contains symlink: {path}")
            path.chmod(0o555)
        current.chmod(0o555)


def stage_snapshot(
    *,
    repo_root: str | Path,
    snapshot_root: str | Path,
    authority_receipt: str | Path,
) -> dict[str, Any]:
    """Create one new, complete, immutable source snapshot."""

    root = Path(os.path.abspath(os.fspath(repo_root)))
    snapshot = Path(os.path.abspath(os.fspath(snapshot_root)))
    _require(root.is_dir() and not root.is_symlink(), "repository root is unsafe")
    _require(not os.path.lexists(snapshot), "snapshot root must not already exist")
    _require(snapshot != root and root not in snapshot.parents, "snapshot must be outside the live repository")

    candidate_path = root / CANDIDATE_RELATIVE
    candidate_payload, method, method_sha256, candidate_bytes = (
        load_and_validate_candidate(candidate_path)
    )
    authority, authority_bytes = publish_authority_receipt(
        repo_root=root,
        output=authority_receipt,
        method_contract=method,
        method_sha256=method_sha256,
    )

    sources = _selected_repository_sources(root)
    compatibility = root / LEGACY_CLOSURE_COMPATIBILITY_RELATIVE
    compatibility_bytes = _stable_bytes(
        compatibility, label="legacy closure compatibility contract"
    )
    snapshot.mkdir(parents=False, mode=0o700)
    source_records: dict[str, dict[str, object]] = {}
    for source in sources:
        relative = source.relative_to(root)
        encoded = _stable_bytes(source, label=f"repository source {relative}")
        _write_exclusive(snapshot / relative, encoded)
        source_records[str(relative)] = _file_record_from_bytes(encoded)

    extra_bytes = {
        CANDIDATE_RELATIVE: candidate_bytes,
        AUTHORITY_RECEIPT_RELATIVE: authority_bytes,
        LEGACY_CLOSURE_COMPATIBILITY_RELATIVE: compatibility_bytes,
    }
    for relative, encoded in extra_bytes.items():
        _write_exclusive(snapshot / relative, encoded)

    all_snapshot_records = dict(source_records)
    for relative, encoded in extra_bytes.items():
        all_snapshot_records[str(relative)] = _file_record_from_bytes(encoded)
    snapshot_files_payload: dict[str, object] = {
        "selection": [
            *SOURCE_SELECTION,
            str(CANDIDATE_RELATIVE),
            str(AUTHORITY_RECEIPT_RELATIVE),
            str(LEGACY_CLOSURE_COMPATIBILITY_RELATIVE),
        ],
        "files": all_snapshot_records,
    }
    snapshot_files_payload["digest"] = canonical_json_sha256(snapshot_files_payload)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "nvos_forward_beta_source_snapshot_staging",
        "candidate": CANDIDATE_ID,
        "eligibility": ELIGIBILITY,
        "strict_unseen_protocol_exact_match": False,
        "main_result_eligible": False,
        "snapshot_root": str(snapshot),
        "source_repository_root": str(root),
        "dependency_dag": [
            "evaluator_derived_method_contract",
            "candidate_method_contract_sha256",
            "protocol_authority_receipt",
            "readonly_source_snapshot",
            "run_manifest",
        ],
        "target_data_read": False,
        "gpu_state_read": False,
        "ordered_tasks": list(ORDERED_TASKS),
        "execution": "fixed_full_eight_without_metric_continuation",
        "maximum_concurrent_scene_evaluators": (
            MAXIMUM_CONCURRENT_SCENE_EVALUATORS
        ),
        "host_memory_policy": HOST_MEMORY_POLICY,
        "serial_scene_gpu_plan": json.loads(json.dumps(SERIAL_SCENE_GPU_PLAN)),
        "candidate_contract": {
            "path": str(CANDIDATE_RELATIVE),
            **_file_record_from_bytes(candidate_bytes),
        },
        "method_contract": method,
        "candidate_method_contract_sha256": method_sha256,
        "protocol_authority_receipt": {
            "path": str(AUTHORITY_RECEIPT_RELATIVE),
            **_file_record_from_bytes(authority_bytes),
        },
        "protocol_authority": authority,
        "protocol_authority_canonical_json_sha256": canonical_json_sha256(
            authority
        ),
        "snapshot_files": snapshot_files_payload,
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_exclusive(snapshot / STAGING_MANIFEST_RELATIVE, encoded_manifest)
    _make_readonly(snapshot)
    validate_snapshot(snapshot)
    return manifest


def _verify_record(path: Path, record: Mapping[str, Any], *, label: str) -> bytes:
    encoded = _stable_bytes(path, label=label)
    _require(record == _file_record_from_bytes(encoded), f"{label} record differs")
    return encoded


def validate_snapshot(snapshot_root: str | Path) -> dict[str, Any]:
    """Validate the staged source set, embedded authority, and permissions."""

    snapshot = Path(os.path.abspath(os.fspath(snapshot_root)))
    root_info = os.stat(snapshot, follow_symlinks=False)
    _require(stat.S_ISDIR(root_info.st_mode), "snapshot root must be a real directory")
    manifest_bytes = _stable_bytes(
        snapshot / STAGING_MANIFEST_RELATIVE, label="snapshot staging manifest"
    )
    manifest = dict(_decode_json(manifest_bytes, label="snapshot staging manifest"))
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "staging schema differs")
    _require(
        manifest.get("artifact_type") == "nvos_forward_beta_source_snapshot_staging",
        "staging artifact type differs",
    )
    _require(manifest.get("candidate") == CANDIDATE_ID, "staging candidate differs")
    _require(manifest.get("snapshot_root") == str(snapshot), "staging root differs")
    _require(manifest.get("ordered_tasks") == list(ORDERED_TASKS), "staging cohort differs")
    _require(
        manifest.get("execution") == "fixed_full_eight_without_metric_continuation",
        "staging execution differs",
    )
    _require(
        manifest.get("maximum_concurrent_scene_evaluators")
        == MAXIMUM_CONCURRENT_SCENE_EVALUATORS,
        "staging scene-evaluator concurrency differs",
    )
    _require(
        manifest.get("host_memory_policy") == HOST_MEMORY_POLICY,
        "staging host-memory policy differs",
    )
    _require(
        manifest.get("serial_scene_gpu_plan")
        == json.loads(json.dumps(SERIAL_SCENE_GPU_PLAN)),
        "staging serial GPU plan differs",
    )

    candidate_record = _mapping(
        manifest.get("candidate_contract"), "staged candidate record"
    )
    candidate_bytes = _verify_record(
        snapshot / CANDIDATE_RELATIVE,
        {key: candidate_record[key] for key in ("bytes", "sha256")},
        label="staged candidate contract",
    )
    candidate_payload = _decode_yaml(candidate_bytes, label="staged candidate contract")
    method, method_sha256 = validate_candidate_payload(candidate_payload)
    _require(manifest.get("method_contract") == method, "staged method contract differs")
    _require(
        manifest.get("candidate_method_contract_sha256") == method_sha256,
        "staged method SHA differs",
    )

    receipt_record = _mapping(
        manifest.get("protocol_authority_receipt"), "staged authority record"
    )
    receipt_bytes = _verify_record(
        snapshot / AUTHORITY_RECEIPT_RELATIVE,
        {key: receipt_record[key] for key in ("bytes", "sha256")},
        label="staged authority receipt",
    )
    authority = _decode_json(receipt_bytes, label="staged authority receipt")
    forward = _mapping(method.get("registered_forward_unary"), "staged forward unary")
    scoring = _mapping(forward.get("scoring_adapter"), "staged scoring adapter")
    _validate_bound_authority(
        authority, method_sha256=method_sha256, scoring_contract=scoring
    )
    _require(manifest.get("protocol_authority") == authority, "inline authority differs")
    _require(
        manifest.get("protocol_authority_canonical_json_sha256")
        == canonical_json_sha256(authority),
        "inline authority canonical digest differs",
    )

    files_payload = _mapping(manifest.get("snapshot_files"), "snapshot files")
    declared_files = _mapping(files_payload.get("files"), "snapshot file records")
    expected_paths = _selected_repository_sources(snapshot)
    expected_relatives = {str(path.relative_to(snapshot)) for path in expected_paths}
    expected_relatives.update(
        {
            str(CANDIDATE_RELATIVE),
            str(AUTHORITY_RECEIPT_RELATIVE),
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

    discovered_regular_files: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        snapshot, topdown=True, followlinks=False
    ):
        current = Path(directory)
        _require(not current.is_symlink(), f"snapshot directory is symlinked: {current}")
        _require(current.stat().st_mode & 0o222 == 0, f"snapshot directory is writable: {current}")
        for name in directory_names:
            path = current / name
            _require(not path.is_symlink(), f"snapshot directory is symlinked: {path}")
            info = os.stat(path, follow_symlinks=False)
            _require(stat.S_ISDIR(info.st_mode), f"snapshot contains a special entry: {path}")
        for name in file_names:
            path = current / name
            _require(not path.is_symlink(), f"snapshot file is symlinked: {path}")
            info = os.stat(path, follow_symlinks=False)
            _require(stat.S_ISREG(info.st_mode), f"snapshot contains a special entry: {path}")
            _require(info.st_mode & 0o222 == 0, f"snapshot file is writable: {path}")
            discovered_regular_files.add(str(path.relative_to(snapshot)))
    _require(
        discovered_regular_files
        == expected_relatives | {str(STAGING_MANIFEST_RELATIVE)},
        "snapshot disk file set differs from declared files plus staging manifest",
    )
    return manifest


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
) -> dict[str, Any]:
    """Build the immutable full-eight run manifest without target reads."""

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
    _require(parent.get("candidate") == "registered-region-v2", "parent asset candidate differs")
    _require(parent.get("scenes") == list(ORDERED_TASKS), "parent asset cohort differs")
    _require(Path(str(parent.get("source_root", ""))).resolve() == source, "parent source root differs")
    _require(Path(str(parent.get("queue_plan", ""))).resolve() == queue_path, "parent queue path differs")
    _require(parent.get("queue_plan_sha256") == _sha256_bytes(queue_bytes), "parent queue digest differs")
    _require(Path(str(parent.get("benchmark_manifest", ""))).resolve() == benchmark_path, "parent benchmark path differs")
    _require(parent.get("benchmark_manifest_sha256") == _sha256_bytes(benchmark_bytes), "parent benchmark digest differs")
    _require(Path(str(parent.get("radio_checkpoint", ""))).resolve() == checkpoint_path, "parent RADIO path differs")
    checkpoint_record = _file_record(checkpoint_path, label="RADIO checkpoint")
    _require(parent.get("radio_checkpoint_sha256") == checkpoint_record["sha256"], "parent RADIO digest differs")

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
            _require(actual["bytes"] == record.get("bytes"), f"{scene}: source size differs for {name}")
            _require(actual["sha256"] == record.get("sha256"), f"{scene}: source digest differs for {name}")
            _require(_file_record(metadata, label=f"{scene} metadata {name}")["sha256"] == record.get("metadata_sha256"), f"{scene}: metadata digest differs for {name}")
        for raw_path, raw_record in queue_scene_inputs[scene].items():
            record = _mapping(raw_record, f"{scene} queue input")
            actual = _file_record(Path(raw_path).resolve(), label=f"{scene} queue input")
            _require(actual["bytes"] == record.get("bytes"), f"{scene}: queue input size differs")
            if record.get("sha256") is not None:
                _require(actual["sha256"] == record.get("sha256"), f"{scene}: queue input digest differs")

    files_payload = _mapping(staging.get("snapshot_files"), "staging files")
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
    receipt_record = dict(
        _mapping(staging.get("protocol_authority_receipt"), "authority receipt")
    )
    staging_path = snapshot / STAGING_MANIFEST_RELATIVE
    candidate_record = dict(_mapping(staging.get("candidate_contract"), "candidate contract"))
    payload: dict[str, Any] = {
        "schema_version": 3,
        "candidate": CANDIDATE_ID,
        "eligibility": ELIGIBILITY,
        "strict_unseen_protocol_exact_match": False,
        "frozen_diagnostic_eligible": False,
        "main_result_eligible": False,
        "target_metric_controls_continuation": False,
        "execution": "fixed_full_eight_without_metric_continuation",
        "maximum_concurrent_scene_evaluators": (
            MAXIMUM_CONCURRENT_SCENE_EVALUATORS
        ),
        "host_memory_policy": HOST_MEMORY_POLICY,
        "serial_scene_gpu_plan": json.loads(json.dumps(SERIAL_SCENE_GPU_PLAN)),
        "scenes": list(ORDERED_TASKS),
        "scene_gpu_assignment": json.loads(json.dumps(SCENE_GPU_ASSIGNMENT)),
        "physical_gpu_binding": PHYSICAL_GPU_BINDING,
        "source_snapshot_root": str(snapshot),
        "source_snapshot_import_root": runtime_closure.get(
            "repository_import_root"
        ),
        "source_snapshot_tree_sha256": _mapping(
            runtime_closure.get("repository_sources"), "runtime sources"
        ).get("digest"),
        "source_snapshot_permissions": dict(source_snapshot_authority),
        "runtime_closure_sha256": runtime_closure.get("digest"),
        "source_snapshot_staging_manifest": {
            "path": str(staging_path),
            **_file_record(staging_path, label="snapshot staging manifest"),
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
        "method_contract_sha256": staging.get(
            "candidate_method_contract_sha256"
        ),
        "protocol_authority_receipt": {
            "path": str(snapshot / AUTHORITY_RECEIPT_RELATIVE),
            "bytes": receipt_record.get("bytes"),
            "sha256": receipt_record.get("sha256"),
        },
        "registered_forward_protocol_authority": staging.get(
            "protocol_authority"
        ),
        "registered_forward_protocol_authority_sha256": staging.get(
            "protocol_authority_canonical_json_sha256"
        ),
        "protocol_reuse": _decode_yaml(
            _stable_bytes(snapshot / CANDIDATE_RELATIVE, label="candidate contract"),
            label="candidate contract",
        ).get("protocol_reuse"),
        "source_root": str(source),
        "source_artifacts": source_artifacts,
        "queue_scene_inputs": queue_scene_inputs,
        "queue_plan": str(queue_path),
        "queue_plan_sha256": _sha256_bytes(queue_bytes),
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": _sha256_bytes(benchmark_bytes),
        "radio_checkpoint": str(checkpoint_path),
        "radio_checkpoint_sha256": checkpoint_record["sha256"],
        "asset_manifest_parent": str(parent_path),
        "asset_manifest_parent_sha256": _sha256_bytes(parent_bytes),
        "asset_reuse_contract": "parent_sha256_plus_path_size_and_sha256_preflight",
        "output_root": str(Path(output_root).resolve()),
        "output_identity": dict(output_identity),
        "runner": str(runner_path),
        "runner_sha256": _file_record(runner_path, label="beta runner")["sha256"],
        "thermal_safety_contract": dict(thermal_safety_contract),
        "gpu_authority": {
            "path": str(authority_path),
            **_file_record(authority_path, label="GPU authority"),
        },
        "scene_receipt_contract": {
            "artifact_type": "nvos-forward-beta-scene-receipt-v1",
            "receipt_root": str(Path(output_root).resolve() / "scene_receipts"),
            "attempt_root": str(Path(output_root).resolve() / "scene_attempts"),
            "skip_only_after_full_receipt_revalidation": True,
            "aggregate_requires_all_eight_scene_receipts": True,
            "metric_based_continuation": False,
        },
        "implementation_sources": implementation,
        "runtime_closure": dict(runtime_closure),
    }
    return payload


def write_run_manifest(output: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(output)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if os.path.lexists(destination):
        existing = _stable_bytes(destination, label="existing run manifest")
        _require(existing == encoded, "output root belongs to another immutable run")
        return destination
    _write_exclusive(destination, encoded)
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage-snapshot")
    stage.add_argument("--repo-root", type=Path, required=True)
    stage.add_argument("--snapshot-root", type=Path, required=True)
    stage.add_argument("--authority-receipt", type=Path, required=True)
    validate = subparsers.add_parser("validate-snapshot")
    validate.add_argument("--snapshot-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "stage-snapshot":
            result = stage_snapshot(
                repo_root=args.repo_root,
                snapshot_root=args.snapshot_root,
                authority_receipt=args.authority_receipt,
            )
        else:
            result = validate_snapshot(args.snapshot_root)
    except StagingError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
