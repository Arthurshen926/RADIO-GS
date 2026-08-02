#!/usr/bin/env python3
"""Aggregate the authority-bound non-exact NVOS forward-Beta full-8 run.

This CPU-only closeout consumes immutable run/report JSON artifacts and their
emitted score arrays.  It does not open target masks, images, model
checkpoints, evaluator inputs, or GPU authority.  The output is deliberately
non-promotional: strict, frozen, and main-result eligibility remain false,
propagated is the only main output, connected selection is diagnostic, and
LUDVIG remains an external comparator fence rather than candidate authority.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    LUDVIG_TASK_ID,
    STRICT_TASKS,
    validate_authority_payload,
)
from radio_gs.scripts.nvos_forward_beta_scene_authority import (
    validate_scene_receipt,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    sha256_file,
    stable_descriptor_load,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "nvos_forward_beta_full8_authority_bound_non_exact_diagnostic"
ELIGIBILITY = "protocol_authority_bound_non_exact_diagnostic"
FIXED_BLOCKERS = [
    "score_semantics_differs",
    "prediction_representation_differs",
]
EXPECTED_SCORING_CONTRACT = {
    "score_semantics": "beta_centered_posterior",
    "prediction_representation": "continuous_beta_centered_posterior",
    "threshold": {"comparison": "greater_or_equal", "value": 0.0},
    "resize": "nearest",
}
REPORT_SUBDIRECTORY = "eval_full_mask_random_walker"
SCENE_RECEIPT_FILENAME = "scene_receipt.json"
SCORE_STAGES = ("unary_prior", "propagated", "connected")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RELIABILITY_SOURCE_KEY = "canonical_primitive_reliability_v1.pt"


class ForwardBetaAggregationError(ValueError):
    """Raised when full-8 authority or diagnostic semantics drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ForwardBetaAggregationError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForwardBetaAggregationError(f"{label} must be a mapping")
    return value


def _unit_metric(value: object, label: str) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError) as error:
        raise ForwardBetaAggregationError(f"{label} must be numeric") from error
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise ForwardBetaAggregationError(f"{label} must be finite and in [0,1]")
    return metric


def _equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _validate_stage(
    stage: object,
    *,
    scene: str,
    name: str,
) -> dict[str, Any]:
    value = _mapping(stage, f"{scene}: stage {name}")
    foreground_iou = _unit_metric(
        value.get("foreground_iou"), f"{scene}: {name} foreground_iou"
    )
    pixel_accuracy = _unit_metric(
        value.get("pixel_accuracy"), f"{scene}: {name} pixel_accuracy"
    )
    raw_frames = value.get("frames")
    _require(
        isinstance(raw_frames, list) and bool(raw_frames),
        f"{scene}: {name} frames must be non-empty",
    )
    frames: list[dict[str, Any]] = []
    frame_ids: set[str] = set()
    for raw_frame in raw_frames:
        frame = _mapping(raw_frame, f"{scene}: {name} frame")
        frame_id = str(frame.get("frame_id", ""))
        _require(
            bool(frame_id) and frame_id not in frame_ids,
            f"{scene}: {name} frame IDs are empty or duplicated",
        )
        frame_ids.add(frame_id)
        frames.append(
            {
                "frame_id": frame_id,
                "foreground_iou": _unit_metric(
                    frame.get("foreground_iou"),
                    f"{scene}: {name}/{frame_id} foreground_iou",
                ),
                "pixel_accuracy": _unit_metric(
                    frame.get("pixel_accuracy"),
                    f"{scene}: {name}/{frame_id} pixel_accuracy",
                ),
            }
        )
    for label, aggregate in (
        ("foreground_iou", foreground_iou),
        ("pixel_accuracy", pixel_accuracy),
    ):
        frame_mean = math.fsum(frame[label] for frame in frames) / len(frames)
        _require(
            _equal(aggregate, frame_mean),
            f"{scene}: {name} {label} is not its frame macro",
        )
    return {
        "foreground_iou": foreground_iou,
        "pixel_accuracy": pixel_accuracy,
        "frames": frames,
    }


def _canonical_declared_artifact_path(value: object, *, label: str) -> Path:
    """Canonicalize parents without following the final path component.

    The final component is intentionally left unresolved so ``sha256_file``
    can reject a symlink with ``O_NOFOLLOW``.  Resolving the whole declared
    path here would silently turn a final-component symlink into its target
    before the immutable-artifact reader sees it.
    """

    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    raw = Path(os.path.abspath(os.path.expanduser(value)))
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as error:
        raise ForwardBetaAggregationError(
            f"{label} parent cannot be resolved"
        ) from error
    return parent / raw.name


def _declared_sha256(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _validate_score_artifacts(
    report: Mapping[str, Any],
    *,
    scene: str,
    report_path: Path,
    frame_ids: Sequence[str],
) -> None:
    """Bind evaluator main/stage score copies by layout and stable content.

    The evaluator deliberately persists the registered readout twice: the
    main copy under ``scores/`` and each diagnostic copy under
    ``stage_scores/<stage>/``.  Consequently the main and propagated paths
    must be distinct, while their per-frame bytes (and therefore SHA-256)
    must be identical.
    """

    expected_frames = set(frame_ids)
    _require(
        len(expected_frames) == len(frame_ids),
        f"{scene}: score artifact frame IDs are duplicated",
    )
    for frame_id in frame_ids:
        _require(
            bool(frame_id)
            and Path(frame_id).name == frame_id
            and frame_id not in {".", ".."},
            f"{scene}: score artifact frame ID is not a safe path component",
        )

    main_paths = _mapping(report.get("score_paths"), f"{scene}: score paths")
    main_hashes = _mapping(report.get("score_sha256"), f"{scene}: score SHA256")
    stage_paths = _mapping(
        report.get("stage_score_paths"), f"{scene}: stage score paths"
    )
    stage_hashes = _mapping(
        report.get("stage_score_sha256"), f"{scene}: stage score SHA256"
    )
    _require(
        set(main_paths) == expected_frames and set(main_hashes) == expected_frames,
        f"{scene}: main score artifact frame set differs",
    )
    _require(
        set(stage_paths) == set(SCORE_STAGES)
        and set(stage_hashes) == set(SCORE_STAGES),
        f"{scene}: stage score artifact set differs",
    )

    per_stage_paths: dict[str, Mapping[str, Any]] = {}
    per_stage_hashes: dict[str, Mapping[str, Any]] = {}
    for stage_name in SCORE_STAGES:
        paths = _mapping(
            stage_paths.get(stage_name),
            f"{scene}: {stage_name} score paths",
        )
        hashes = _mapping(
            stage_hashes.get(stage_name),
            f"{scene}: {stage_name} score SHA256",
        )
        _require(
            set(paths) == expected_frames and set(hashes) == expected_frames,
            f"{scene}: {stage_name} score artifact frame set differs",
        )
        per_stage_paths[stage_name] = paths
        per_stage_hashes[stage_name] = hashes

    artifact_root = report_path.parent
    for frame_id in frame_ids:
        expected_main_path = artifact_root / "scores" / scene / f"{frame_id}.npy"
        main_path = _canonical_declared_artifact_path(
            main_paths[frame_id], label=f"{scene}: main/{frame_id} score artifact"
        )
        _require(
            main_path == expected_main_path,
            f"{scene}: main/{frame_id} score artifact path differs",
        )
        main_digest = _declared_sha256(
            main_hashes[frame_id],
            label=f"{scene}: main/{frame_id} score artifact SHA256",
        )
        try:
            actual_main_digest = sha256_file(main_path)
        except (OSError, ValueError) as error:
            raise ForwardBetaAggregationError(
                f"{scene}: main/{frame_id} score artifact is not a stable regular file"
            ) from error
        _require(
            actual_main_digest == main_digest,
            f"{scene}: main/{frame_id} score artifact SHA256 differs",
        )

        for stage_name in SCORE_STAGES:
            expected_stage_path = (
                artifact_root
                / "stage_scores"
                / stage_name
                / scene
                / f"{frame_id}.npy"
            )
            stage_path = _canonical_declared_artifact_path(
                per_stage_paths[stage_name][frame_id],
                label=f"{scene}: {stage_name}/{frame_id} score artifact",
            )
            _require(
                stage_path == expected_stage_path,
                f"{scene}: {stage_name}/{frame_id} score artifact path differs",
            )
            stage_digest = _declared_sha256(
                per_stage_hashes[stage_name][frame_id],
                label=f"{scene}: {stage_name}/{frame_id} score artifact SHA256",
            )
            try:
                actual_stage_digest = sha256_file(stage_path)
            except (OSError, ValueError) as error:
                raise ForwardBetaAggregationError(
                    f"{scene}: {stage_name}/{frame_id} score artifact is not a "
                    "stable regular file"
                ) from error
            _require(
                actual_stage_digest == stage_digest,
                f"{scene}: {stage_name}/{frame_id} score artifact SHA256 differs",
            )
            if stage_name == "propagated":
                _require(
                    stage_path != main_path
                    and stage_digest == main_digest
                    and actual_stage_digest == actual_main_digest,
                    f"{scene}: propagated/{frame_id} score content is not the "
                    "main artifact content",
                )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_candidate: str | None = None,
    expected_forward_mode: str | None = None,
    require_reliability_bindings: bool = False,
) -> tuple[Mapping[str, Any], str, str, str, dict[str, Any] | None]:
    _require(
        manifest.get("scenes") == list(STRICT_TASKS),
        "run manifest must contain exactly the ordered frozen full-8 tasks",
    )
    _require(
        manifest.get("eligibility") == ELIGIBILITY,
        "run manifest eligibility must remain authority-bound non-exact diagnostic",
    )
    candidate = str(manifest.get("candidate", ""))
    _require(bool(candidate), "run manifest candidate is empty")
    if expected_candidate is not None:
        _require(
            candidate == expected_candidate,
            "run manifest candidate differs from the aggregation profile",
        )
    method_contract = _mapping(
        manifest.get("method_contract"), "run manifest method contract"
    )
    method_sha256 = canonical_json_sha256(method_contract)
    _require(
        method_contract.get("final_readout") == "propagated"
        and method_contract.get("selection_applied_to_main_output") is False,
        "run manifest must bind propagated main output and diagnostic selection",
    )
    forward = _mapping(
        method_contract.get("registered_forward_unary"),
        "run manifest forward-Beta contract",
    )
    _require(
        forward.get("status") == ELIGIBILITY
        and forward.get("strict_unseen_eligible") is False
        and forward.get("selection_applied_to_main_output") is False
        and forward.get("required_final_readout") == "propagated"
        and forward.get("scoring_adapter") == EXPECTED_SCORING_CONTRACT,
        "run manifest forward-Beta non-exact contract differs",
    )
    if expected_forward_mode is not None:
        _require(
            forward.get("mode") == expected_forward_mode,
            "run manifest forward-Beta mode differs from the aggregation profile",
        )

    authority = _mapping(
        manifest.get("registered_forward_protocol_authority"),
        "run manifest forward-Beta authority",
    )
    authority_sha256 = canonical_json_sha256(authority)
    _require(
        manifest.get("registered_forward_protocol_authority_sha256")
        == authority_sha256,
        "run manifest forward-Beta authority SHA256 differs",
    )
    try:
        validate_authority_payload(authority)
    except ValueError as error:
        raise ForwardBetaAggregationError(
            f"run manifest forward-Beta authority is invalid: {error}"
        ) from error
    _require(
        authority.get("scoring_contract") == EXPECTED_SCORING_CONTRACT
        and authority.get("strict_unseen_protocol_exact_match") is False
        and authority.get("strict_unseen_exact_match_blockers") == FIXED_BLOCKERS,
        "run manifest authority must remain non-exact with the fixed blockers",
    )
    authority_candidate = _mapping(
        authority.get("candidate"), "run manifest authority candidate"
    )
    _require(
        authority_candidate.get("method_contract_sha256") == method_sha256,
        "run manifest method contract SHA256 differs from authority",
    )
    comparator = _mapping(
        authority.get("external_comparator_provenance"),
        "LUDVIG comparator provenance",
    )
    candidate_binding = _mapping(
        comparator.get("candidate_binding"), "LUDVIG candidate fence"
    )
    excluded = _mapping(
        comparator.get("excluded_from_candidate_authority"),
        "excluded LUDVIG authority",
    )
    _require(
        comparator.get("binding_role") == "external_method_comparator_only"
        and set(candidate_binding.values()) == {None}
        and excluded.get("canonical_task_id") == LUDVIG_TASK_ID,
        "LUDVIG must remain a comparator-only negative fence",
    )
    reliability_binding = None
    if require_reliability_bindings:
        _require(
            manifest.get("v1_result_or_receipt_reuse_permitted") is False,
            "v2 run manifest must explicitly forbid v1 result/receipt reuse",
        )
        reliability_binding = _validate_reliability_bindings(manifest)
    return (
        authority,
        authority_sha256,
        candidate,
        method_sha256,
        reliability_binding,
    )


def _validate_reliability_bindings(
    manifest: Mapping[str, Any],
    *,
    reliability_manifest_validator: Callable[[str | Path], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Validate the v2 reliability authority once and bind every scene projection."""

    record = _mapping(
        manifest.get("reliability_cache_manifest"),
        "v2 reliability cache manifest record",
    )
    _require(
        set(record) == {"path", "bytes", "sha256"},
        "v2 reliability cache manifest file-record fields differ",
    )
    declared_bytes = record.get("bytes")
    declared_sha256 = record.get("sha256")
    _require(
        isinstance(declared_bytes, int)
        and not isinstance(declared_bytes, bool)
        and declared_bytes >= 0,
        "v2 reliability cache manifest byte count differs",
    )
    _declared_sha256(
        declared_sha256,
        label="v2 reliability cache manifest SHA256",
    )
    try:
        observed_bytes, observed_sha256, manifest_path = stable_descriptor_load(
            str(record.get("path", "")),
            lambda handle: int(os.fstat(handle.fileno()).st_size),
            expected_sha256=str(declared_sha256),
            label="v2 reliability cache manifest",
        )
    except (OSError, ValueError) as error:
        raise ForwardBetaAggregationError(
            f"v2 reliability cache manifest is invalid: {error}"
        ) from error
    _require(
        dict(record)
        == {
            "path": str(manifest_path),
            "bytes": observed_bytes,
            "sha256": observed_sha256,
        },
        "v2 reliability cache manifest immutable record differs",
    )

    try:
        stable_manifest, stable_sha256, stable_path = load_json_object(
            manifest_path,
            expected_sha256=observed_sha256,
            label="v2 reliability cache manifest",
        )
        if reliability_manifest_validator is None:
            from radio_gs.scripts.bind_nvos_beta_v2_reliability_manifest import (
                validate_manifest_payload,
            )

            # Full cohort validation is a one-time snapshot-publication gate.
            # Aggregation rechecks the immutable manifest bytes and all eight
            # logical projections, but must not hash ~17GB a second time.
            validate_manifest_payload(stable_manifest, verify_files=False)
            reliability_manifest = dict(stable_manifest)
        else:
            reliability_manifest = dict(
                reliability_manifest_validator(manifest_path)
            )
    except (OSError, ValueError) as error:
        raise ForwardBetaAggregationError(
            f"v2 reliability cache authority is invalid: {error}"
        ) from error
    _require(
        stable_path == manifest_path
        and stable_sha256 == observed_sha256
        and reliability_manifest == stable_manifest,
        "v2 reliability cache manifest changed during validation",
    )
    _require(
        reliability_manifest.get("ordered_scenes") == list(STRICT_TASKS),
        "v2 reliability cache manifest cohort differs",
    )
    cache_scenes = _mapping(
        reliability_manifest.get("scenes"),
        "v2 reliability cache manifest scenes",
    )
    _require(
        list(cache_scenes) == list(STRICT_TASKS),
        "v2 reliability cache manifest scene order differs",
    )
    source_artifacts = _mapping(
        manifest.get("source_artifacts"),
        "v2 run manifest source artifacts",
    )
    _require(
        set(source_artifacts) == set(STRICT_TASKS),
        "v2 run manifest source-artifact cohort differs",
    )

    scene_bindings: dict[str, dict[str, Any]] = {}
    cache_paths: set[str] = set()
    metadata_paths: set[str] = set()
    for scene in STRICT_TASKS:
        scene_sources = _mapping(
            source_artifacts.get(scene),
            f"{scene}: v2 source artifacts",
        )
        source = _mapping(
            scene_sources.get(RELIABILITY_SOURCE_KEY),
            f"{scene}: v2 reliability source",
        )
        _require(
            set(source)
            == {
                "path",
                "bytes",
                "sha256",
                "metadata_path",
                "metadata_sha256",
            },
            f"{scene}: v2 reliability logical source fields differ",
        )
        cache_row = _mapping(
            cache_scenes.get(scene),
            f"{scene}: v2 reliability cache row",
        )
        cache_record = _mapping(
            cache_row.get("reliability_cache"),
            f"{scene}: v2 reliability cache file record",
        )
        report_record = _mapping(
            cache_row.get("build_report"),
            f"{scene}: v2 reliability build-report file record",
        )
        _require(
            set(cache_record) == {"path", "bytes", "sha256"}
            and set(report_record) == {"path", "bytes", "sha256"},
            f"{scene}: v2 reliability cache/report record fields differ",
        )
        expected_source = {
            "path": cache_record.get("path"),
            "bytes": cache_record.get("bytes"),
            "sha256": cache_record.get("sha256"),
            "metadata_path": report_record.get("path"),
            "metadata_sha256": report_record.get("sha256"),
        }
        _require(
            dict(source) == expected_source,
            f"{scene}: v2 reliability logical source differs from cache authority",
        )
        cache_path = str(source["path"])
        metadata_path = str(source["metadata_path"])
        _require(
            bool(cache_path)
            and bool(metadata_path)
            and cache_path not in cache_paths
            and metadata_path not in metadata_paths,
            f"{scene}: v2 reliability paths are empty or reused across scenes",
        )
        cache_paths.add(cache_path)
        metadata_paths.add(metadata_path)
        scene_bindings[scene] = dict(source)

    return {
        "manifest": dict(record),
        "logical_source_key": RELIABILITY_SOURCE_KEY,
        "scene_bindings": scene_bindings,
        "all_scene_cache_and_metadata_records_match_authority": True,
    }


def _validate_scene_report(
    report: Mapping[str, Any],
    *,
    scene: str,
    report_path: Path,
    run_manifest_sha256: str,
    run_manifest_method_sha256: str,
    eligibility: str,
    authority: Mapping[str, Any],
    authority_sha256: str,
    expected_forward_mode: str | None = None,
) -> dict[str, Any]:
    _require(report.get("scene_id") == scene, f"{scene}: scene_id differs")
    _require(
        report.get("registered_forward_protocol_authority") == authority
        and report.get("registered_forward_protocol_authority_sha256")
        == authority_sha256,
        f"{scene}: top-level authority payload/SHA differs from run manifest",
    )
    _require(
        report.get("run_manifest_sha256") == run_manifest_sha256,
        f"{scene}: run manifest SHA256 differs",
    )
    _require(
        "beta_centered_posterior" in str(report.get("method", "")),
        f"{scene}: method is not the forward-Beta diagnostic",
    )

    method = _mapping(report.get("method_contract"), f"{scene}: method contract")
    _require(
        report.get("method_config_sha256") == canonical_json_sha256(method),
        f"{scene}: method_config_sha256 differs",
    )
    _require(
        method.get("candidate_run_manifest_sha256") == run_manifest_sha256
        and method.get("candidate_method_contract_sha256") == run_manifest_method_sha256
        and method.get("candidate_eligibility") == eligibility,
        f"{scene}: method/run-manifest binding differs",
    )
    _require(
        method.get("registered_forward_protocol_authority") == authority
        and method.get("registered_forward_protocol_authority_sha256")
        == authority_sha256,
        f"{scene}: method authority payload/SHA differs",
    )
    method_solver = _mapping(
        method.get("shared_solver"), f"{scene}: method shared solver"
    )
    report_solver = _mapping(
        report.get("shared_solver"), f"{scene}: report shared solver"
    )
    method_forward = _mapping(
        method_solver.get("registered_forward_unary"),
        f"{scene}: method forward-Beta contract",
    )
    report_forward = _mapping(
        report_solver.get("registered_forward_unary"),
        f"{scene}: report forward-Beta contract",
    )
    _require(
        method_solver.get("registered_readout_stage") == "propagated"
        and report_solver.get("registered_readout_stage") == "propagated"
        and method_forward.get("status") == ELIGIBILITY
        and method_forward.get("strict_unseen_eligible") is False
        and method_forward.get("selection_applied_to_main_output") is False
        and method_forward.get("required_final_readout") == "propagated",
        f"{scene}: propagated/diagnostic forward-Beta solver contract differs",
    )
    if expected_forward_mode is not None:
        _require(
            method_forward.get("mode") == expected_forward_mode
            and report_forward.get("mode") == expected_forward_mode
            and report_forward == method_forward,
            f"{scene}: forward-Beta report mode differs from the aggregation profile",
        )

    safety = _mapping(report.get("safety"), f"{scene}: safety")
    _require(
        safety.get("candidate_eligibility") == eligibility
        and safety.get("frozen_diagnostic_eligible") is False
        and safety.get("main_result_eligible") is False
        and safety.get("strict_unseen_eligible") is False
        and safety.get("strict_unseen_protocol_exact_match") is False
        and safety.get("registered_forward_protocol_authority_sha256")
        == authority_sha256,
        f"{scene}: non-exact/frozen/main safety labels differ",
    )
    for forbidden in (
        "target_ground_truth_opened_before_prediction_write",
        "target_rgb_opened",
        "target_camera_used_as_support",
        "test_calibration",
    ):
        _require(
            safety.get(forbidden) is False,
            f"{scene}: unsafe report flag {forbidden}",
        )

    evaluation = _mapping(
        report.get("evaluation_protocol_contract"),
        f"{scene}: evaluation protocol contract",
    )
    _require(
        report.get("evaluation_protocol_sha256") == canonical_json_sha256(evaluation),
        f"{scene}: evaluation protocol SHA256 differs",
    )
    _require(
        evaluation.get("method_config_sha256") == report.get("method_config_sha256")
        and evaluation.get("dataset_protocol_sha256")
        == report.get("dataset_protocol_sha256")
        and evaluation.get("final_readout") == "propagated"
        and evaluation.get("registered_forward_protocol_authority_sha256")
        == authority_sha256
        and evaluation.get("strict_unseen_protocol_exact_match") is False
        and evaluation.get("score_semantics")
        == EXPECTED_SCORING_CONTRACT["score_semantics"]
        and evaluation.get("prediction_representation")
        == EXPECTED_SCORING_CONTRACT["prediction_representation"]
        and evaluation.get("pixel_threshold") == EXPECTED_SCORING_CONTRACT["threshold"]
        and evaluation.get("resize_to_ground_truth") == "cv2.INTER_NEAREST",
        f"{scene}: evaluation non-exact propagated contract differs",
    )

    stages = _mapping(report.get("stage_metrics"), f"{scene}: stage metrics")
    _require(
        {"unary_prior", "propagated", "connected"}.issubset(stages),
        f"{scene}: required stage metrics are missing",
    )
    unary = _validate_stage(stages["unary_prior"], scene=scene, name="unary_prior")
    propagated = _validate_stage(stages["propagated"], scene=scene, name="propagated")
    connected = _validate_stage(stages["connected"], scene=scene, name="connected")
    final_iou = _unit_metric(
        report.get("foreground_iou"), f"{scene}: final foreground_iou"
    )
    final_accuracy = _unit_metric(
        report.get("pixel_accuracy"), f"{scene}: final pixel_accuracy"
    )
    _require(
        _equal(final_iou, propagated["foreground_iou"])
        and _equal(final_accuracy, propagated["pixel_accuracy"])
        and report.get("frames") == stages["propagated"].get("frames"),
        f"{scene}: propagated stage is not the reported main output",
    )
    _validate_score_artifacts(
        report,
        scene=scene,
        report_path=report_path,
        frame_ids=[frame["frame_id"] for frame in propagated["frames"]],
    )

    return {
        "task_id": scene,
        "main_output": {
            "stage": "propagated",
            "foreground_iou": final_iou,
            "pixel_accuracy": final_accuracy,
        },
        "diagnostics": {
            "unary_prior": {
                "foreground_iou": unary["foreground_iou"],
                "pixel_accuracy": unary["pixel_accuracy"],
            },
            "connected": {
                "role": "diagnostic_only_not_applied_to_main_output",
                "foreground_iou": connected["foreground_iou"],
                "pixel_accuracy": connected["pixel_accuracy"],
            },
        },
        "report": file_record(report_path),
        "method_config_sha256": str(report["method_config_sha256"]),
        "evaluation_protocol_sha256": str(report["evaluation_protocol_sha256"]),
        "dataset_protocol_sha256": str(report["dataset_protocol_sha256"]),
        "registered_forward_protocol_authority_sha256": authority_sha256,
    }


def _macro(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in path:
            current = current[key]
        values.append(float(current))
    return math.fsum(values) / len(values)


def aggregate_forward_beta_full8(
    *,
    run_manifest_path: str | Path,
    result_root: str | Path,
    receipt_root: str | Path,
    expected_candidate: str | None = None,
    expected_forward_mode: str | None = None,
    require_reliability_bindings: bool = False,
    receipt_validator: Callable[..., Mapping[str, Any]] | None = None,
    artifact_type: str | None = None,
) -> dict[str, Any]:
    manifest, run_manifest_sha256, canonical_manifest_path = load_json_object(
        run_manifest_path, label="forward-Beta run manifest"
    )
    (
        authority,
        authority_sha256,
        candidate,
        method_sha256,
        reliability_binding,
    ) = _validate_manifest(
        manifest,
        expected_candidate=expected_candidate,
        expected_forward_mode=expected_forward_mode,
        require_reliability_bindings=require_reliability_bindings,
    )
    selected_receipt_validator = receipt_validator or validate_scene_receipt
    root = Path(result_root).expanduser().resolve()
    _require(root.is_dir(), "forward-Beta result root is absent")
    receipts_root = Path(receipt_root).expanduser().resolve()
    _require(receipts_root.is_dir(), "forward-Beta receipt root is absent")
    expected_paths = {
        (
            root / scene / REPORT_SUBDIRECTORY / f"{scene}_evaluation.json"
        ).resolve(): scene
        for scene in STRICT_TASKS
    }
    discovered = {
        path.resolve()
        for path in root.rglob("*_evaluation.json")
        if path.is_file() or path.is_symlink()
    }
    missing = [
        scene for path, scene in expected_paths.items() if path not in discovered
    ]
    extra = sorted(str(path) for path in discovered if path not in expected_paths)
    _require(
        not missing and not extra and discovered == set(expected_paths),
        "full-8 report set differs; " f"missing={missing!r}, extra={extra!r}",
    )
    expected_receipts = {
        (receipts_root / scene / SCENE_RECEIPT_FILENAME).resolve(): scene
        for scene in STRICT_TASKS
    }
    discovered_receipts = {
        path.resolve()
        for path in receipts_root.rglob(SCENE_RECEIPT_FILENAME)
        if path.is_file() or path.is_symlink()
    }
    missing_receipts = [
        scene
        for path, scene in expected_receipts.items()
        if path not in discovered_receipts
    ]
    extra_receipts = sorted(
        str(path) for path in discovered_receipts if path not in expected_receipts
    )
    _require(
        not missing_receipts
        and not extra_receipts
        and discovered_receipts == set(expected_receipts),
        "full-8 scene receipt set differs; "
        f"missing={missing_receipts!r}, extra={extra_receipts!r}",
    )

    rows: list[dict[str, Any]] = []
    for scene in STRICT_TASKS:
        path = (
            root / scene / REPORT_SUBDIRECTORY / f"{scene}_evaluation.json"
        ).resolve()
        report, _report_sha256, loaded_path = load_json_object(
            path, label=f"forward-Beta {scene} report"
        )
        row = _validate_scene_report(
            report,
            scene=scene,
            report_path=loaded_path,
            run_manifest_sha256=run_manifest_sha256,
            run_manifest_method_sha256=method_sha256,
            eligibility=ELIGIBILITY,
            authority=authority,
            authority_sha256=authority_sha256,
            expected_forward_mode=expected_forward_mode,
        )
        receipt_path = (receipts_root / scene / SCENE_RECEIPT_FILENAME).resolve()
        try:
            validated_receipt = selected_receipt_validator(
                receipt_path,
                run_manifest=canonical_manifest_path,
                scene=scene,
                result=loaded_path,
            )
        except ValueError as error:
            raise ForwardBetaAggregationError(
                f"{scene}: scene GPU authority receipt is invalid: {error}"
            ) from error
        receipt_payload = _mapping(
            validated_receipt.get("payload"),
            f"{scene}: validated scene receipt payload",
        )
        promotion = _mapping(
            receipt_payload.get("promotion"),
            f"{scene}: scene receipt promotion",
        )
        _require(
            promotion
            == {
                "main_result_eligible": False,
                "frozen_diagnostic_eligible": False,
                "strict_unseen_protocol_exact_match": False,
            }
            and receipt_payload.get("result") == file_record(loaded_path),
            f"{scene}: scene receipt promotion/result binding differs",
        )
        row["scene_gpu_authority"] = {
            "receipt": validated_receipt["receipt"],
            "status": receipt_payload.get("status"),
            "gpu_identity": receipt_payload.get("gpu_identity"),
            "owner_audit": receipt_payload.get("owner_audit"),
            "cuda_attestation": receipt_payload.get("cuda_attestation"),
            "postcheck": receipt_payload.get("postcheck"),
            "promotion": dict(promotion),
        }
        rows.append(row)

    for digest_name in (
        "method_config_sha256",
        "evaluation_protocol_sha256",
        "dataset_protocol_sha256",
        "registered_forward_protocol_authority_sha256",
    ):
        _require(
            len({row[digest_name] for row in rows}) == 1,
            f"full-8 scene reports disagree on {digest_name}",
        )

    comparator = authority["external_comparator_provenance"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type or ARTIFACT_TYPE,
        "status": ELIGIBILITY,
        "claim_scope": "authority_bound_non_exact_diagnostic_only",
        "candidate": candidate,
        "strict_unseen_protocol_exact_match": False,
        "strict_unseen_exact_match_blockers": list(FIXED_BLOCKERS),
        "frozen_diagnostic_eligible": False,
        "main_result_eligible": False,
        "complete": True,
        "expected_task_count": len(STRICT_TASKS),
        "completed_task_count": len(rows),
        "expected_tasks": list(STRICT_TASKS),
        "aggregation": {
            "unit": "scene_task",
            "weighting": "equal_weight",
            "formula": "arithmetic_mean_of_eight_scene_level_metrics",
            "frame_or_pixel_weighting": False,
        },
        "main_output": {
            "stage": "propagated",
            "role": "only_reported_main_output",
            "macro": {
                "foreground_iou": _macro(rows, ("main_output", "foreground_iou")),
                "pixel_accuracy": _macro(rows, ("main_output", "pixel_accuracy")),
            },
        },
        "diagnostics": {
            "unary_prior_macro": {
                "foreground_iou": _macro(
                    rows, ("diagnostics", "unary_prior", "foreground_iou")
                ),
                "pixel_accuracy": _macro(
                    rows, ("diagnostics", "unary_prior", "pixel_accuracy")
                ),
            },
            "connected_macro": {
                "role": "diagnostic_only_not_applied_to_main_output",
                "foreground_iou": _macro(
                    rows, ("diagnostics", "connected", "foreground_iou")
                ),
                "pixel_accuracy": _macro(
                    rows, ("diagnostics", "connected", "pixel_accuracy")
                ),
            },
        },
        "tasks": rows,
        "run_manifest": file_record(canonical_manifest_path),
        "run_manifest_method_contract_sha256": method_sha256,
        "registered_forward_protocol_authority": authority,
        "registered_forward_protocol_authority_sha256": authority_sha256,
        "authority_binding": {
            "payload_and_sha_match_run_manifest_and_all_eight_reports": True,
            "strict_exactness_derived_not_caller_supplied": True,
        },
        "scene_gpu_authority": {
            "receipt_root": str(receipts_root),
            "receipt_relative_path": f"<scene>/{SCENE_RECEIPT_FILENAME}",
            "expected_receipt_count": len(STRICT_TASKS),
            "validated_receipt_count": len(rows),
            "all_receipts_bind_current_manifest_and_report": True,
            "gpu_owner_attestation_postcheck_chain_validated": True,
        },
        "external_comparator_fence": {
            "binding_role": comparator["binding_role"],
            "candidate_binding": comparator["candidate_binding"],
            "excluded_from_candidate_authority": comparator[
                "excluded_from_candidate_authority"
            ],
            "use": "comparison_context_only_never_candidate_authority",
        },
    }
    if expected_forward_mode is not None:
        summary["registered_forward_unary_mode"] = expected_forward_mode
    if reliability_binding is not None:
        summary["v1_result_or_receipt_reuse_permitted"] = False
        summary["reliability_cache_authority"] = reliability_binding
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(
            f"immutable aggregation output already exists: {args.output}"
        )
    summary = aggregate_forward_beta_full8(
        run_manifest_path=args.run_manifest,
        result_root=args.result_root,
        receipt_root=args.receipt_root,
    )
    write_frozen_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
