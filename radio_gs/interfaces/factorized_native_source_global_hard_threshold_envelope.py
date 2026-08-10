"""Formal source-only hard-threshold PR/F1 envelope diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as source_formal
from radio_gs.interfaces import factorized_native_source_global_margin_calibration_v2 as v2_formal
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    validate_file_record,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_source_global_hard_threshold_envelope_"
    "execution_authority.v1"
)
RESULT_SCHEMA = "radio_gs.factorized_native_source_global_hard_threshold_envelope.v1"
SCHEMA_VERSION = 1
IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/diagnose_factorized_native_contrast_v21_hard_threshold_envelope.py"
)
IMPLEMENTATION_DEPENDENCIES = {
    "envelope_formal": Path(__file__).resolve(),
    "envelope_tests": (
        Path(__file__).resolve().parents[2]
        / "tests/test_factorized_native_source_global_hard_threshold_envelope.py"
    ),
    "class_balanced_calibration_formal_v2": Path(v2_formal.__file__).resolve(),
    "immutable_margin_extractor_v1": (
        Path(__file__).resolve().parents[1]
        / "scripts/calibrate_factorized_native_contrast_v21_global_margin.py"
    ),
    "source_result_validator": Path(source_formal.__file__).resolve(),
}
TRAIN_SCENES = v2_formal.TRAIN_SCENES
VALIDATION_SCENES = v2_formal.VALIDATION_SCENES
FIT_QUERY_ROWS = v2_formal.FIT_QUERY_ROWS
CANONICAL_NEGATIVE_ROWS = v2_formal.CANONICAL_NEGATIVE_ROWS
REGIONS_PER_SCENE = v2_formal.REGIONS_PER_SCENE
IDENTITY_THRESHOLD = 0.0
HARD_BOUNDARY = 0.5
METRIC_STRICT_IMPROVEMENT = 1e-7
MINIMUM_PRECISION = 0.25
MINIMUM_IDENTITY_PRECISION_RETENTION = 0.75
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_access() -> dict[str, bool]:
    access = dict(v2_formal.source_access())
    access["class_balanced_v2_source_diagnostic_opened"] = True
    access["source_validation_labels_opened_for_no_promotion_oracle"] = True
    return access


def envelope_contract() -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "margin_and_teacher": {
            "source": "identical_frozen_contrast_v2_1_student_and_teacher_definition",
            "student_margin": "cosine_positive_minus_max_four_canonical_negative",
            "teacher_soft_probability": (
                "valid_view_mean_sigmoid_10_times_teacher_margin"
            ),
            "teacher_hard_boundary": HARD_BOUNDARY,
            "identity_prediction": "student_margin_greater_than_or_equal_to_zero",
        },
        "axes": {
            "train_scenes": list(TRAIN_SCENES),
            "validation_scenes": list(VALIDATION_SCENES),
            "regions_per_scene": REGIONS_PER_SCENE,
            "fit_query_rows": FIT_QUERY_ROWS,
            "all_pairs_used": True,
            "random_sampling": False,
        },
        "train_selected_candidate": {
            "threshold_axis": "all_unique_train_student_margins_exact_descending_sweep",
            "precision_floor": (
                "max_0.25_and_0.75_times_train_identity_precision"
            ),
            "eligibility": (
                "precision_floor_and_recall_strictly_improves_identity_and_"
                "f1_strictly_improves_identity"
            ),
            "objective": "maximum_train_hard_f1",
            "tie_break": "first_descending_unique_margin_equal_maximum",
            "validation_contribution": False,
            "one_global_threshold": True,
            "scene_parameters": False,
            "query_parameters": False,
        },
        "validation_gate": {
            "same_frozen_train_threshold_on_both_scenes": True,
            "hard_recall_strict_improvement": METRIC_STRICT_IMPROVEMENT,
            "hard_f1_strict_improvement": METRIC_STRICT_IMPROVEMENT,
            "minimum_precision": MINIMUM_PRECISION,
            "minimum_identity_precision_retention": (
                MINIMUM_IDENTITY_PRECISION_RETENTION
            ),
            "all_checks_required_on_every_scene": True,
        },
        "validation_oracle": {
            "role": "diagnostic_ranking_upper_bound_only",
            "threshold_axis": (
                "all_unique_joint_validation_student_margins_exact_descending_sweep"
            ),
            "question": (
                "exists_one_global_threshold_satisfying_all_four_checks_on_both_scenes"
            ),
            "representative": "maximum_mean_validation_f1_among_feasible_thresholds",
            "promotion_authorized": False,
            "parameter_export_authorized": False,
        },
        "target_or_query_execution_authorized": False,
        "access": source_access(),
    }


ENVELOPE_CONTRACT_SHA256 = canonical_json_sha256(envelope_contract())


def record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError("hard-threshold envelope output must be canonical absolute")
    return resolved


def _validate_v2_diagnostic(value: Mapping[str, str]) -> dict[str, Any]:
    raw, _, _ = load_json_object(
        value["path"],
        expected_sha256=value["sha256"],
        label="class-balanced source calibration V2 result",
    )
    checked = v2_formal.validate_calibration_result(raw, require_promotion=False)
    if (
        checked["status"] != "source_only_complete_no_promotion"
        or checked["query_calibration_authorized"] is not False
    ):
        raise ValueError("hard-threshold envelope requires rejected V2 diagnostic")
    return checked


def validate_execution_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source global hard-threshold envelope execution authority",
    )
    required = {
        "schema", "schema_version", "status", "implementation",
        "implementation_dependencies", "envelope_contract_sha256",
        "source_contrast_v21_result", "class_balanced_v2_result",
        "fit_text_bank", "canonical_negative_bank", "benchmark_exclusion_manifest",
        "envelope_output", "diagnostic_authorized", "target_execution_authorized",
        "query_execution_authorized", "metric_execution_authorized", "source_access",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("hard-threshold envelope authority fields differ")
    authority = dict(raw)
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status") != "authorized_source_only_hard_threshold_envelope"
        or authority.get("envelope_contract_sha256") != ENVELOPE_CONTRACT_SHA256
        or authority.get("diagnostic_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("hard-threshold envelope authority header differs")
    # The complete promoted source chain remains the first external gate.
    source_gate = source_formal.validate_source_contrast_v21_result(
        authority["source_contrast_v21_result"]
    )
    implementation = validate_file_record(
        authority["implementation"], label="hard-threshold envelope implementation"
    )
    if implementation != IMPLEMENTATION_PATH:
        raise ValueError("hard-threshold envelope implementation differs")
    dependencies = authority["implementation_dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("hard-threshold envelope dependencies differ")
    verified_dependencies: dict[str, dict[str, str]] = {}
    for name, expected in IMPLEMENTATION_DEPENDENCIES.items():
        observed = validate_file_record(
            dependencies[name], label=f"hard-threshold envelope dependency {name}"
        )
        if observed != expected:
            raise ValueError(f"hard-threshold envelope dependency differs: {name}")
        verified_dependencies[name] = record(
            dependencies[name], label=f"hard-threshold envelope dependency {name}"
        )
    inputs: dict[str, dict[str, str]] = {}
    for name in (
        "class_balanced_v2_result", "fit_text_bank", "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    ):
        shaped = record(authority[name], label=f"hard-threshold envelope {name}")
        observed = validate_file_record(shaped, label=f"hard-threshold envelope {name}")
        if str(observed) != shaped["path"]:
            raise ValueError(f"hard-threshold envelope {name} path differs")
        inputs[name] = shaped
    v2_result = _validate_v2_diagnostic(inputs["class_balanced_v2_result"])
    source_record = record(
        authority["source_contrast_v21_result"], label="contrast V2.1 source result"
    )
    if v2_result["input_authority"]["source_contrast_v21_result"] != source_record:
        raise ValueError("hard-threshold envelope and V2 source result differ")
    for name in ("fit_text_bank", "canonical_negative_bank", "benchmark_exclusion_manifest"):
        if v2_result["input_authority"][name] != inputs[name]:
            raise ValueError(f"hard-threshold envelope and V2 {name} differ")
    output = _canonical_output(authority["envelope_output"])
    if expected_output is not None and output != str(Path(expected_output).expanduser().resolve()):
        raise ValueError("hard-threshold envelope output differs")
    authority["source_contrast_v21_result"] = source_record
    authority["implementation"] = record(
        authority["implementation"], label="hard-threshold envelope implementation"
    )
    authority["implementation_dependencies"] = verified_dependencies
    authority.update(inputs)
    authority["envelope_output"] = output
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    authority["verified_v2_diagnostic"] = v2_result
    return authority


def expected_checks(
    identity: Mapping[str, float | int], candidate: Mapping[str, float | int]
) -> dict[str, bool]:
    return {
        "hard_recall_strictly_improved": (
            float(candidate["teacher_positive_recall"])
            >= float(identity["teacher_positive_recall"])
            + METRIC_STRICT_IMPROVEMENT
        ),
        "hard_f1_strictly_improved": (
            float(candidate["teacher_positive_f1"])
            >= float(identity["teacher_positive_f1"])
            + METRIC_STRICT_IMPROVEMENT
        ),
        "precision_absolute_floor": (
            float(candidate["teacher_positive_precision"]) >= MINIMUM_PRECISION
        ),
        "precision_identity_retention": (
            float(candidate["teacher_positive_precision"])
            >= MINIMUM_IDENTITY_PRECISION_RETENTION
            * float(identity["teacher_positive_precision"])
        ),
    }


_METRIC_NAMES = {
    "pairs", "teacher_positive_count", "predicted_positive_count",
    "true_positive_count", "teacher_positive_rate", "predicted_positive_rate",
    "teacher_positive_precision", "teacher_positive_recall", "teacher_positive_f1",
}


def _validate_metrics(value: object, *, label: str, expected_pairs: int) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != _METRIC_NAMES:
        raise ValueError(f"{label} hard-threshold metrics fields differ")
    metrics = dict(value)
    counts = {
        "pairs", "teacher_positive_count", "predicted_positive_count", "true_positive_count"
    }
    rates = _METRIC_NAMES - counts
    if (
        any(not isinstance(metrics[name], int) or metrics[name] < 0 for name in counts)
        or metrics["pairs"] != expected_pairs
        or metrics["teacher_positive_count"] > expected_pairs
        or metrics["predicted_positive_count"] > expected_pairs
        or metrics["true_positive_count"] > min(
            metrics["teacher_positive_count"], metrics["predicted_positive_count"]
        )
        or any(
            not isinstance(metrics[name], (int, float))
            or isinstance(metrics[name], bool)
            or not math.isfinite(float(metrics[name]))
            or not 0.0 <= float(metrics[name]) <= 1.0
            for name in rates
        )
    ):
        raise ValueError(f"{label} hard-threshold metrics differ")
    return metrics


def validate_result(value: object) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "status", "contract", "contract_sha256",
        "execution_authority", "input_authority", "thresholds", "sampling_audit",
        "train_selection", "validation_audit", "promotion_checks",
        "no_promotion_oracle", "global_threshold_authorized", "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("hard-threshold envelope result fields differ")
    result = dict(value)
    if (
        result["schema"] != RESULT_SCHEMA
        or result["schema_version"] != SCHEMA_VERSION
        or result["status"] not in {"source_only_promoted", "source_only_complete_no_promotion"}
        or result["contract"] != envelope_contract()
        or result["contract_sha256"] != ENVELOPE_CONTRACT_SHA256
        or result["source_access"] != source_access()
    ):
        raise ValueError("hard-threshold envelope result header differs")
    result["execution_authority"] = record(
        result["execution_authority"], label="hard-threshold execution authority"
    )
    names = {
        "source_contrast_v21_result", "source_contrast_v21_checkpoint",
        "class_balanced_v2_result", "fit_text_bank", "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    }
    inputs = result["input_authority"]
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("hard-threshold envelope inputs differ")
    result["input_authority"] = {
        name: record(inputs[name], label=f"hard-threshold input {name}")
        for name in sorted(names)
    }
    thresholds = result["thresholds"]
    if (
        not isinstance(thresholds, Mapping)
        or set(thresholds) != {"identity", "train_selected_candidate"}
        or thresholds["identity"] != IDENTITY_THRESHOLD
        or not isinstance(thresholds["train_selected_candidate"], (int, float))
        or isinstance(thresholds["train_selected_candidate"], bool)
        or not math.isfinite(float(thresholds["train_selected_candidate"]))
    ):
        raise ValueError("hard-threshold envelope thresholds differ")
    expected_sampling = {
        "train_scenes": list(TRAIN_SCENES),
        "validation_scenes": list(VALIDATION_SCENES),
        "regions_per_scene": REGIONS_PER_SCENE,
        "query_rows": FIT_QUERY_ROWS,
        "canonical_negative_rows": CANONICAL_NEGATIVE_ROWS,
        "train_pairs": len(TRAIN_SCENES) * REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        "validation_pairs_per_scene": REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        "all_rows_and_queries_used": True,
        "validation_contribution_to_train_selection": False,
    }
    if result["sampling_audit"] != expected_sampling:
        raise ValueError("hard-threshold envelope sampling differs")
    train = result["train_selection"]
    train_fields = {
        "unique_thresholds_swept", "precision_floor", "identity", "candidate",
        "candidate_checks", "selection_objective", "validation_contribution",
    }
    if not isinstance(train, Mapping) or set(train) != train_fields:
        raise ValueError("hard-threshold train selection fields differ")
    train_pairs = expected_sampling["train_pairs"]
    identity_train = _validate_metrics(train["identity"], label="train identity", expected_pairs=train_pairs)
    candidate_train = _validate_metrics(train["candidate"], label="train candidate", expected_pairs=train_pairs)
    if (
        not isinstance(train["unique_thresholds_swept"], int)
        or train["unique_thresholds_swept"] <= 0
        or abs(float(train["precision_floor"]) - max(
            MINIMUM_PRECISION,
            MINIMUM_IDENTITY_PRECISION_RETENTION * float(identity_train["teacher_positive_precision"]),
        )) > 1e-12
        or train["candidate_checks"] != expected_checks(identity_train, candidate_train)
        or train["selection_objective"] != "maximum_train_hard_f1_among_eligible_thresholds"
        or train["validation_contribution"] is not False
        or not all(train["candidate_checks"].values())
    ):
        raise ValueError("hard-threshold train selection differs")
    validation = result["validation_audit"]
    if not isinstance(validation, Mapping) or tuple(validation) != VALIDATION_SCENES:
        raise ValueError("hard-threshold validation scenes differ")
    checked_validation: dict[str, Any] = {}
    scene_checks: dict[str, dict[str, bool]] = {}
    val_pairs = expected_sampling["validation_pairs_per_scene"]
    for scene in VALIDATION_SCENES:
        row = validation[scene]
        if not isinstance(row, Mapping) or set(row) != {"identity", "candidate", "checks"}:
            raise ValueError(f"hard-threshold {scene} validation fields differ")
        identity = _validate_metrics(row["identity"], label=f"{scene} identity", expected_pairs=val_pairs)
        candidate = _validate_metrics(row["candidate"], label=f"{scene} candidate", expected_pairs=val_pairs)
        checks = expected_checks(identity, candidate)
        if row["checks"] != checks:
            raise ValueError(f"hard-threshold {scene} checks differ")
        checked_validation[scene] = {"identity": identity, "candidate": candidate, "checks": checks}
        scene_checks[scene] = checks
    result["validation_audit"] = checked_validation
    expected_promotion = {
        "train_candidate_passed": all(train["candidate_checks"].values()),
        "every_validation_scene_passed": all(
            all(checks.values()) for checks in scene_checks.values()
        ),
        "candidate_is_one_global_train_selected_threshold": True,
    }
    promoted = all(expected_promotion.values())
    if (
        result["promotion_checks"] != expected_promotion
        or result["global_threshold_authorized"] is not promoted
        or result["status"] != ("source_only_promoted" if promoted else "source_only_complete_no_promotion")
    ):
        raise ValueError("hard-threshold promotion decision differs")
    oracle = result["no_promotion_oracle"]
    oracle_fields = {
        "role", "joint_unique_thresholds_swept", "unified_feasible_threshold_exists",
        "unified_feasible_threshold_count", "representative_threshold",
        "representative_metrics", "promotion_authorized", "parameter_export_authorized",
    }
    if not isinstance(oracle, Mapping) or set(oracle) != oracle_fields:
        raise ValueError("hard-threshold oracle fields differ")
    exists = oracle["unified_feasible_threshold_exists"]
    count = oracle["unified_feasible_threshold_count"]
    if (
        oracle["role"] != "validation_label_oracle_ranking_diagnostic_only"
        or not isinstance(oracle["joint_unique_thresholds_swept"], int)
        or oracle["joint_unique_thresholds_swept"] <= 0
        or not isinstance(exists, bool)
        or not isinstance(count, int)
        or count < 0
        or exists is not (count > 0)
        or oracle["promotion_authorized"] is not False
        or oracle["parameter_export_authorized"] is not False
    ):
        raise ValueError("hard-threshold oracle decision differs")
    if exists:
        threshold = oracle["representative_threshold"]
        metrics = oracle["representative_metrics"]
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not isinstance(metrics, Mapping)
            or tuple(metrics) != VALIDATION_SCENES
        ):
            raise ValueError("hard-threshold oracle representative differs")
        for scene in VALIDATION_SCENES:
            row = metrics[scene]
            if not isinstance(row, Mapping) or set(row) != {"metrics", "checks"}:
                raise ValueError("hard-threshold oracle scene fields differ")
            checked = _validate_metrics(row["metrics"], label=f"oracle {scene}", expected_pairs=val_pairs)
            checks = expected_checks(checked_validation[scene]["identity"], checked)
            if row["checks"] != checks or not all(checks.values()):
                raise ValueError("hard-threshold oracle representative is not feasible")
    elif oracle["representative_threshold"] is not None or oracle["representative_metrics"] is not None:
        raise ValueError("hard-threshold infeasible oracle must not export a representative")
    return result


__all__ = [
    "ENVELOPE_CONTRACT_SHA256", "EXECUTION_AUTHORITY_SCHEMA", "FIT_QUERY_ROWS",
    "HARD_BOUNDARY", "IDENTITY_THRESHOLD", "IMPLEMENTATION_DEPENDENCIES",
    "IMPLEMENTATION_PATH", "RESULT_SCHEMA", "SCHEMA_VERSION", "TRAIN_SCENES",
    "VALIDATION_SCENES", "envelope_contract", "expected_checks", "record",
    "source_access", "validate_execution_authority", "validate_result",
]
