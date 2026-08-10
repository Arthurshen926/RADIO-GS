"""Formal source-only global absolute-margin calibration for contrast V2.1."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as source_formal
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance_loss
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.scripts import train_surface_region_typed_context_response_listwise_v2 as text_loader
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    validate_file_record,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_source_global_margin_calibration_"
    "execution_authority.v1"
)
RESULT_SCHEMA = "radio_gs.factorized_native_source_global_margin_calibration.v1"
SCHEMA_VERSION = 1
IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/calibrate_factorized_native_contrast_v21_global_margin.py"
)
IMPLEMENTATION_DEPENDENCIES = {
    "calibration_formal": Path(__file__).resolve(),
    "source_result_validator": Path(source_formal.__file__).resolve(),
    "fit_text_bank_loader": Path(text_loader.__file__).resolve(),
    "canonical_negative_loader": Path(relevance_loss.__file__).resolve(),
}
TRAIN_SCENES = ("scene0001_00", "scene0002_00", "scene0003_00", "scene0005_00")
VALIDATION_SCENES = ("scene0004_00", "scene0008_00")
FIT_QUERY_ROWS = 806
CANONICAL_NEGATIVE_ROWS = 4
REGIONS_PER_SCENE = 4096
IDENTITY_A = 10.0
IDENTITY_B = 0.0
TEACHER_LOGIT_SCALE = 10.0
MAXIMUM_NEWTON_ITERATIONS = 64
MAXIMUM_BACKTRACK_STEPS = 24
MINIMUM_POSITIVE_SLOPE = 1e-8
OPTIMIZATION_GRADIENT_TOLERANCE = 1e-10
METRIC_STRICT_IMPROVEMENT = 1e-7
RANK_NON_REGRESSION_TOLERANCE = 1e-12
MINIMUM_CLASSIFICATION_METRIC = 0.25
MINIMUM_BASELINE_CLASSIFICATION_RETENTION = 0.75
RANK_SAMPLE_CAP = 262_144
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_access() -> dict[str, bool]:
    return {
        "query_independent": True,
        "source_train_opened": True,
        "source_validation_opened_for_promotion_only": True,
        "generic_target_blind_text_bank_opened": True,
        "canonical_generic_negative_bank_opened": True,
        "benchmark_exclusion_manifest_opened": True,
        "target_heldout_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "runtime_query_strings_consumed": False,
        "target_descriptor_opened": False,
        "target_relevance_opened": False,
        "target_metrics_computed": False,
        "scene_identifiers_consumed_by_calibrator": False,
        "per_scene_parameters": False,
    }


def calibration_contract() -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_candidate": {
            "result": "promoted_contrast_v2_1_direction_only",
            "checkpoint": "schema21_selected_source_step",
            "cohort": {
                "train": list(TRAIN_SCENES),
                "validation": list(VALIDATION_SCENES),
                "regions_per_scene": REGIONS_PER_SCENE,
                "scene_and_physical_space_disjoint": True,
            },
        },
        "text": {
            "positive": "frozen_target_blind_806_fit_embedding_rows",
            "negative": "frozen_four_canonical_official_siglip2_rows",
            "benchmark_vocabulary": False,
            "runtime_strings": False,
        },
        "sampling": {
            "fit_and_metric_region_axis": "all_4096_eligible_canonical_rows",
            "fit_and_metric_query_axis": "all_806_frozen_fit_rows",
            "rank_correlation_axis": (
                "fixed_evenly_spaced_flat_region_query_pairs_cap262144"
            ),
            "random_sampling": False,
        },
        "margin": {
            "student": "cosine_positive_minus_max_four_canonical_negative",
            "teacher_per_view": (
                "cosine_positive_minus_max_four_canonical_negative"
            ),
            "soft_teacher_probability": (
                "valid_view_arithmetic_mean_of_sigmoid_10_times_view_margin"
            ),
            "identity": {"a": IDENTITY_A, "b": IDENTITY_B},
            "candidate": "sigmoid(a_times_student_margin_plus_b)",
            "constraint": "one_global_a_strictly_positive_and_one_global_b",
            "scene_parameters": False,
            "query_parameters": False,
        },
        "fit": {
            "split": "source_train_four_only",
            "objective": "mean_soft_binary_cross_entropy",
            "solver": "deterministic_strictly_convex_two_parameter_newton_irls",
            "maximum_iterations": MAXIMUM_NEWTON_ITERATIONS,
            "maximum_backtrack_steps": MAXIMUM_BACKTRACK_STEPS,
            "minimum_positive_slope": MINIMUM_POSITIVE_SLOPE,
            "gradient_tolerance": OPTIMIZATION_GRADIENT_TOLERANCE,
            "initialization": {"a": IDENTITY_A, "b": IDENTITY_B},
            "validation_contribution": False,
        },
        "promotion": {
            "validation_only": True,
            "required_scenes": list(VALIDATION_SCENES),
            "every_scene_strict_improvement": {
                "brier": METRIC_STRICT_IMPROVEMENT,
                "soft_binary_cross_entropy": METRIC_STRICT_IMPROVEMENT,
                "mean_absolute_error": METRIC_STRICT_IMPROVEMENT,
            },
            "rank_correlation_non_regression_tolerance": (
                RANK_NON_REGRESSION_TOLERANCE
            ),
            "teacher_positive_boundary": 0.5,
            "prediction_boundary": 0.5,
            "precision_and_recall": {
                "minimum_absolute": MINIMUM_CLASSIFICATION_METRIC,
                "minimum_identity_retention": (
                    MINIMUM_BASELINE_CLASSIFICATION_RETENTION
                ),
            },
            "all_checks_required": True,
        },
        "target_or_query_execution_authorized": False,
        "access": source_access(),
    }


CALIBRATION_CONTRACT_SHA256 = canonical_json_sha256(calibration_contract())


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
        raise ValueError("global margin calibration output must be canonical absolute")
    return resolved


def validate_execution_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source global margin calibration execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "implementation_dependencies",
        "calibration_contract_sha256",
        "source_contrast_v21_result",
        "fit_text_bank",
        "canonical_negative_bank",
        "benchmark_exclusion_manifest",
        "calibration_output",
        "calibration_authorized",
        "target_execution_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "source_access",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("source global margin calibration authority fields differ")
    authority = dict(raw)
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_global_margin_fit_4train_2validation"
        or authority.get("calibration_contract_sha256")
        != CALIBRATION_CONTRACT_SHA256
        or authority.get("calibration_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("source global margin calibration authority header differs")

    # The promoted source result and full nested code/checkpoint chain are
    # validated before any text bank is opened.
    source_gate = source_formal.validate_source_contrast_v21_result(
        authority["source_contrast_v21_result"]
    )
    implementation = validate_file_record(
        authority["implementation"], label="global margin implementation"
    )
    if implementation != IMPLEMENTATION_PATH:
        raise ValueError("global margin calibration implementation differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("global margin calibration dependencies differ")
    verified_dependencies: dict[str, dict[str, str]] = {}
    for name, expected in IMPLEMENTATION_DEPENDENCIES.items():
        observed = validate_file_record(
            dependencies[name], label=f"global margin dependency {name}"
        )
        if observed != expected:
            raise ValueError(f"global margin dependency differs: {name}")
        verified_dependencies[name] = record(
            dependencies[name], label=f"global margin dependency {name}"
        )
    inputs: dict[str, dict[str, str]] = {}
    for name in (
        "fit_text_bank",
        "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    ):
        shaped = record(authority[name], label=f"global margin {name}")
        observed = validate_file_record(shaped, label=f"global margin {name}")
        if str(observed) != shaped["path"]:
            raise ValueError(f"global margin {name} path differs")
        inputs[name] = shaped
    output = _canonical_output(authority["calibration_output"])
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("global margin calibration output differs")
    authority["source_contrast_v21_result"] = record(
        authority["source_contrast_v21_result"], label="contrast V2.1 source result"
    )
    authority["implementation"] = record(
        authority["implementation"], label="global margin implementation"
    )
    authority["implementation_dependencies"] = verified_dependencies
    authority.update(inputs)
    authority["calibration_output"] = output
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    return authority


def expected_validation_checks(
    identity: Mapping[str, float | int], candidate: Mapping[str, float | int]
) -> dict[str, bool]:
    return {
        "brier_strictly_improved": (
            float(candidate["brier"]) + METRIC_STRICT_IMPROVEMENT
            <= float(identity["brier"])
        ),
        "soft_binary_cross_entropy_strictly_improved": (
            float(candidate["soft_binary_cross_entropy"])
            + METRIC_STRICT_IMPROVEMENT
            <= float(identity["soft_binary_cross_entropy"])
        ),
        "mean_absolute_error_strictly_improved": (
            float(candidate["mean_absolute_error"]) + METRIC_STRICT_IMPROVEMENT
            <= float(identity["mean_absolute_error"])
        ),
        "rank_correlation_non_regression": (
            float(candidate["rank_correlation"])
            + RANK_NON_REGRESSION_TOLERANCE
            >= float(identity["rank_correlation"])
        ),
        "teacher_positive_precision_not_catastrophic": (
            float(candidate["teacher_positive_precision"])
            >= max(
                MINIMUM_CLASSIFICATION_METRIC,
                MINIMUM_BASELINE_CLASSIFICATION_RETENTION
                * float(identity["teacher_positive_precision"]),
            )
        ),
        "teacher_positive_recall_not_catastrophic": (
            float(candidate["teacher_positive_recall"])
            >= max(
                MINIMUM_CLASSIFICATION_METRIC,
                MINIMUM_BASELINE_CLASSIFICATION_RETENTION
                * float(identity["teacher_positive_recall"]),
            )
        ),
    }


_METRIC_NAMES = {
    "pairs",
    "rank_samples",
    "brier",
    "soft_binary_cross_entropy",
    "mean_absolute_error",
    "rank_correlation",
    "teacher_positive_rate",
    "predicted_positive_rate",
    "teacher_positive_precision",
    "teacher_positive_recall",
}


def _validate_metrics(
    value: object, *, label: str, expected_pairs: int
) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != _METRIC_NAMES:
        raise ValueError(f"{label} calibration metrics fields differ")
    metrics = dict(value)
    if (
        not isinstance(metrics["pairs"], int)
        or metrics["pairs"] != int(expected_pairs)
        or not isinstance(metrics["rank_samples"], int)
        or metrics["rank_samples"] != RANK_SAMPLE_CAP
        or any(
            not isinstance(metrics[name], (int, float))
            or isinstance(metrics[name], bool)
            or not math.isfinite(float(metrics[name]))
            for name in _METRIC_NAMES - {"pairs", "rank_samples"}
        )
        or any(
            not 0.0 <= float(metrics[name]) <= 1.0
            for name in (
                "brier",
                "mean_absolute_error",
                "teacher_positive_rate",
                "predicted_positive_rate",
                "teacher_positive_precision",
                "teacher_positive_recall",
            )
        )
        or not -1.000001 <= float(metrics["rank_correlation"]) <= 1.000001
        or float(metrics["soft_binary_cross_entropy"]) < 0.0
    ):
        raise ValueError(f"{label} calibration metrics differ")
    return metrics


def validate_calibration_result(value: object, *, require_promotion: bool = False) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "contract",
        "contract_sha256",
        "execution_authority",
        "input_authority",
        "parameters",
        "optimization_audit",
        "sampling_audit",
        "fit_metrics",
        "validation_metrics",
        "promotion_checks",
        "query_calibration_authorized",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source global margin calibration result fields differ")
    result = dict(value)
    if (
        result["schema"] != RESULT_SCHEMA
        or result["schema_version"] != SCHEMA_VERSION
        or result["status"]
        not in {"source_only_promoted", "source_only_complete_no_promotion"}
        or result["contract"] != calibration_contract()
        or result["contract_sha256"] != CALIBRATION_CONTRACT_SHA256
        or result["source_access"] != source_access()
    ):
        raise ValueError("source global margin calibration result header differs")
    result["execution_authority"] = record(
        result["execution_authority"], label="calibration execution authority"
    )
    inputs = result["input_authority"]
    names = {
        "source_contrast_v21_result",
        "source_contrast_v21_checkpoint",
        "fit_text_bank",
        "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("source global margin calibration inputs differ")
    result["input_authority"] = {
        name: record(inputs[name], label=f"calibration input {name}")
        for name in sorted(names)
    }
    parameters = result["parameters"]
    if (
        not isinstance(parameters, Mapping)
        or set(parameters) != {"identity", "candidate"}
        or parameters["identity"] != {"a": IDENTITY_A, "b": IDENTITY_B}
        or not isinstance(parameters["candidate"], Mapping)
        or set(parameters["candidate"]) != {"a", "b"}
        or not all(
            isinstance(parameters["candidate"][name], (int, float))
            and not isinstance(parameters["candidate"][name], bool)
            and math.isfinite(float(parameters["candidate"][name]))
            for name in ("a", "b")
        )
        or float(parameters["candidate"]["a"]) <= MINIMUM_POSITIVE_SLOPE
    ):
        raise ValueError("source global margin calibration parameters differ")
    optimization = result["optimization_audit"]
    optimization_fields = {
        "solver",
        "iterations",
        "converged",
        "initial_soft_binary_cross_entropy",
        "final_soft_binary_cross_entropy",
        "final_max_absolute_gradient",
        "minimum_observed_hessian_eigenvalue",
        "backtrack_reductions",
    }
    if (
        not isinstance(optimization, Mapping)
        or set(optimization) != optimization_fields
        or optimization.get("solver")
        != "deterministic_strictly_convex_two_parameter_newton_irls"
        or not isinstance(optimization.get("iterations"), int)
        or not 1 <= optimization["iterations"] <= MAXIMUM_NEWTON_ITERATIONS
        or optimization.get("converged") is not True
        or not isinstance(optimization.get("backtrack_reductions"), int)
        or optimization["backtrack_reductions"] < 0
        or any(
            not isinstance(optimization[name], (int, float))
            or isinstance(optimization[name], bool)
            or not math.isfinite(float(optimization[name]))
            for name in (
                "initial_soft_binary_cross_entropy",
                "final_soft_binary_cross_entropy",
                "final_max_absolute_gradient",
                "minimum_observed_hessian_eigenvalue",
            )
        )
        or float(optimization["final_soft_binary_cross_entropy"])
        >= float(optimization["initial_soft_binary_cross_entropy"])
        or float(optimization["final_max_absolute_gradient"])
        > OPTIMIZATION_GRADIENT_TOLERANCE
        or float(optimization["minimum_observed_hessian_eigenvalue"]) <= 1e-12
    ):
        raise ValueError("source global margin optimization audit differs")
    sampling = result["sampling_audit"]
    if sampling != {
        "train_scenes": list(TRAIN_SCENES),
        "validation_scenes": list(VALIDATION_SCENES),
        "regions_per_scene": REGIONS_PER_SCENE,
        "query_rows": FIT_QUERY_ROWS,
        "canonical_negative_rows": CANONICAL_NEGATIVE_ROWS,
        "train_pairs": len(TRAIN_SCENES) * REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        "validation_pairs_per_scene": REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        "validation_contribution_to_fit": False,
        "all_rows_and_queries_used": True,
    }:
        raise ValueError("source global margin calibration sampling differs")
    fit_metrics = result["fit_metrics"]
    if not isinstance(fit_metrics, Mapping) or set(fit_metrics) != {
        "identity",
        "candidate",
    }:
        raise ValueError("source global margin fit metrics differ")
    result["fit_metrics"] = {
        name: _validate_metrics(
            fit_metrics[name],
            label=f"fit {name}",
            expected_pairs=len(TRAIN_SCENES) * REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        )
        for name in ("identity", "candidate")
    }
    validation = result["validation_metrics"]
    if not isinstance(validation, Mapping) or tuple(validation) != VALIDATION_SCENES:
        raise ValueError("source global margin validation scenes differ")
    checked_validation: dict[str, Any] = {}
    expected_scene_checks: dict[str, dict[str, bool]] = {}
    for scene in VALIDATION_SCENES:
        row = validation[scene]
        if not isinstance(row, Mapping) or set(row) != {
            "identity",
            "candidate",
            "checks",
        }:
            raise ValueError(f"source global margin {scene} validation fields differ")
        identity = _validate_metrics(
            row["identity"],
            label=f"{scene} identity",
            expected_pairs=REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        )
        candidate = _validate_metrics(
            row["candidate"],
            label=f"{scene} candidate",
            expected_pairs=REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        )
        checks = expected_validation_checks(identity, candidate)
        if row["checks"] != checks:
            raise ValueError(f"source global margin {scene} checks differ")
        checked_validation[scene] = {
            "identity": identity,
            "candidate": candidate,
            "checks": checks,
        }
        expected_scene_checks[scene] = checks
    result["validation_metrics"] = checked_validation
    expected_promotion = {
        "optimization_converged": (
            isinstance(result["optimization_audit"], Mapping)
            and result["optimization_audit"].get("converged") is True
        ),
        "candidate_slope_strictly_positive": (
            float(parameters["candidate"]["a"]) > MINIMUM_POSITIVE_SLOPE
        ),
        "every_validation_scene_passed": all(
            all(checks.values()) for checks in expected_scene_checks.values()
        ),
    }
    promoted = all(expected_promotion.values())
    if (
        result["promotion_checks"] != expected_promotion
        or result["query_calibration_authorized"] is not promoted
        or result["status"]
        != ("source_only_promoted" if promoted else "source_only_complete_no_promotion")
    ):
        raise ValueError("source global margin promotion decision differs")
    if require_promotion and not promoted:
        raise ValueError("source global margin calibration did not pass promotion")
    return result


__all__ = [
    "CALIBRATION_CONTRACT_SHA256",
    "CANONICAL_NEGATIVE_ROWS",
    "EXECUTION_AUTHORITY_SCHEMA",
    "FIT_QUERY_ROWS",
    "IDENTITY_A",
    "IDENTITY_B",
    "IMPLEMENTATION_DEPENDENCIES",
    "IMPLEMENTATION_PATH",
    "RESULT_SCHEMA",
    "RANK_SAMPLE_CAP",
    "REGIONS_PER_SCENE",
    "SCHEMA_VERSION",
    "TEACHER_LOGIT_SCALE",
    "TRAIN_SCENES",
    "VALIDATION_SCENES",
    "calibration_contract",
    "expected_validation_checks",
    "record",
    "source_access",
    "validate_calibration_result",
    "validate_execution_authority",
]
