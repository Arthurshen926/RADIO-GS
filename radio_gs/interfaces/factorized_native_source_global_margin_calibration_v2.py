"""Formal source-only class-balanced global margin calibration V2."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as source_formal
from radio_gs.interfaces import factorized_native_source_global_margin_calibration as v1_formal
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance_loss
from radio_gs.scripts import train_surface_region_typed_context_response_listwise_v2 as text_loader
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    validate_file_record,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_source_global_margin_calibration_v2_"
    "execution_authority.v1"
)
RESULT_SCHEMA = "radio_gs.factorized_native_source_global_margin_calibration.v2"
SCHEMA_VERSION = 2
IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/calibrate_factorized_native_contrast_v21_global_margin_v2.py"
)
IMPLEMENTATION_DEPENDENCIES = {
    "calibration_formal_v2": Path(__file__).resolve(),
    "calibration_v2_tests": (
        Path(__file__).resolve().parents[2]
        / "tests/test_factorized_native_source_global_margin_calibration_v2.py"
    ),
    "unweighted_calibration_formal_v1": Path(v1_formal.__file__).resolve(),
    "immutable_margin_extractor_v1": (
        Path(__file__).resolve().parents[1]
        / "scripts/calibrate_factorized_native_contrast_v21_global_margin.py"
    ),
    "source_result_validator": Path(source_formal.__file__).resolve(),
    "fit_text_bank_loader": Path(text_loader.__file__).resolve(),
    "canonical_negative_loader": Path(relevance_loss.__file__).resolve(),
}
TRAIN_SCENES = v1_formal.TRAIN_SCENES
VALIDATION_SCENES = v1_formal.VALIDATION_SCENES
FIT_QUERY_ROWS = v1_formal.FIT_QUERY_ROWS
CANONICAL_NEGATIVE_ROWS = v1_formal.CANONICAL_NEGATIVE_ROWS
REGIONS_PER_SCENE = v1_formal.REGIONS_PER_SCENE
IDENTITY_A = v1_formal.IDENTITY_A
IDENTITY_B = v1_formal.IDENTITY_B
MAXIMUM_NEWTON_ITERATIONS = v1_formal.MAXIMUM_NEWTON_ITERATIONS
MAXIMUM_BACKTRACK_STEPS = v1_formal.MAXIMUM_BACKTRACK_STEPS
MINIMUM_POSITIVE_SLOPE = v1_formal.MINIMUM_POSITIVE_SLOPE
OPTIMIZATION_GRADIENT_TOLERANCE = v1_formal.OPTIMIZATION_GRADIENT_TOLERANCE
METRIC_STRICT_IMPROVEMENT = 1e-7
RANK_NON_REGRESSION_TOLERANCE = 1e-12
MINIMUM_PRECISION = 0.25
MINIMUM_IDENTITY_PRECISION_RETENTION = 0.75
RANK_SAMPLE_CAP = v1_formal.RANK_SAMPLE_CAP
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_access() -> dict[str, bool]:
    access = dict(v1_formal.source_access())
    access["unweighted_source_only_v1_diagnostic_opened"] = True
    return access


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
        "text_and_margin": {
            "positive": "frozen_target_blind_806_fit_embedding_rows",
            "negative": "frozen_four_canonical_official_siglip2_rows",
            "student_margin": "cosine_positive_minus_max_four_canonical_negative",
            "teacher_soft_probability": (
                "valid_view_mean_sigmoid_10_times_teacher_margin"
            ),
            "benchmark_vocabulary": False,
            "runtime_strings": False,
        },
        "fit": {
            "split": "source_train_four_only",
            "axes": "all_4096_regions_times_all_806_fit_rows_per_scene",
            "train_soft_positive_rate": "mean_of_train_teacher_soft_probability",
            "positive_weight": "0.5_divided_by_train_soft_positive_rate",
            "negative_weight": "0.5_divided_by_one_minus_train_soft_positive_rate",
            "objective": (
                "mean_train_positive_weight_y_softplus_minus_z_plus_"
                "negative_weight_one_minus_y_softplus_z"
            ),
            "positive_and_negative_soft_mass_each_total_fraction": 0.5,
            "validation_contribution": False,
            "validation_reuses_fixed_train_weights": True,
            "candidate": "sigmoid(a_times_student_margin_plus_b)",
            "constraint": "one_global_a_strictly_positive_and_one_global_b",
            "scene_parameters": False,
            "query_parameters": False,
            "solver": "deterministic_strictly_convex_two_parameter_newton_irls",
            "maximum_iterations": MAXIMUM_NEWTON_ITERATIONS,
            "maximum_backtrack_steps": MAXIMUM_BACKTRACK_STEPS,
            "gradient_tolerance": OPTIMIZATION_GRADIENT_TOLERANCE,
            "initialization": {"a": IDENTITY_A, "b": IDENTITY_B},
        },
        "promotion": {
            "required_scenes": list(VALIDATION_SCENES),
            "fixed_train_weight_balanced_bce_strict_improvement": (
                METRIC_STRICT_IMPROVEMENT
            ),
            "fixed_train_weight_balanced_brier_strict_improvement": (
                METRIC_STRICT_IMPROVEMENT
            ),
            "hard_teacher_and_prediction_boundary": 0.5,
            "hard_f1_strict_improvement": METRIC_STRICT_IMPROVEMENT,
            "hard_recall_strict_improvement": METRIC_STRICT_IMPROVEMENT,
            "minimum_candidate_precision": MINIMUM_PRECISION,
            "minimum_identity_precision_retention": (
                MINIMUM_IDENTITY_PRECISION_RETENTION
            ),
            "rank_correlation_non_regression_tolerance": (
                RANK_NON_REGRESSION_TOLERANCE
            ),
            "all_checks_required_on_every_validation_scene": True,
        },
        "unweighted_v1_role": "immutable_source_only_rejected_diagnostic_only",
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
        raise ValueError("class-balanced global margin output must be canonical absolute")
    return resolved


def _validate_unweighted_v1(value: Mapping[str, str]) -> dict[str, Any]:
    raw, _, _ = load_json_object(
        value["path"],
        expected_sha256=value["sha256"],
        label="unweighted source global margin V1 result",
    )
    checked = v1_formal.validate_calibration_result(raw, require_promotion=False)
    if (
        checked["status"] != "source_only_complete_no_promotion"
        or checked["query_calibration_authorized"] is not False
    ):
        raise ValueError("unweighted V1 must be the rejected source-only diagnostic")
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
        label="class-balanced source global margin V2 execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "implementation_dependencies",
        "calibration_contract_sha256",
        "source_contrast_v21_result",
        "unweighted_v1_result",
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
        raise ValueError("class-balanced global margin authority fields differ")
    authority = dict(raw)
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_class_balanced_global_margin_fit"
        or authority.get("calibration_contract_sha256")
        != CALIBRATION_CONTRACT_SHA256
        or authority.get("calibration_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("class-balanced global margin authority header differs")

    # This source gate is deliberately first: no text, diagnostic, or implementation
    # record may be opened before the complete promoted source chain passes.
    source_gate = source_formal.validate_source_contrast_v21_result(
        authority["source_contrast_v21_result"]
    )
    implementation = validate_file_record(
        authority["implementation"], label="class-balanced margin implementation"
    )
    if implementation != IMPLEMENTATION_PATH:
        raise ValueError("class-balanced global margin implementation differs")
    dependencies = authority["implementation_dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("class-balanced global margin dependencies differ")
    verified_dependencies: dict[str, dict[str, str]] = {}
    for name, expected in IMPLEMENTATION_DEPENDENCIES.items():
        observed = validate_file_record(
            dependencies[name], label=f"class-balanced margin dependency {name}"
        )
        if observed != expected:
            raise ValueError(f"class-balanced global margin dependency differs: {name}")
        verified_dependencies[name] = record(
            dependencies[name], label=f"class-balanced margin dependency {name}"
        )
    inputs: dict[str, dict[str, str]] = {}
    for name in (
        "unweighted_v1_result",
        "fit_text_bank",
        "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    ):
        shaped = record(authority[name], label=f"class-balanced margin {name}")
        observed = validate_file_record(shaped, label=f"class-balanced margin {name}")
        if str(observed) != shaped["path"]:
            raise ValueError(f"class-balanced margin {name} path differs")
        inputs[name] = shaped
    v1_result = _validate_unweighted_v1(inputs["unweighted_v1_result"])
    source_record = record(
        authority["source_contrast_v21_result"], label="contrast V2.1 source result"
    )
    if v1_result["input_authority"]["source_contrast_v21_result"] != source_record:
        raise ValueError("class-balanced V2 and unweighted V1 source result differ")
    for name in ("fit_text_bank", "canonical_negative_bank", "benchmark_exclusion_manifest"):
        if v1_result["input_authority"][name] != inputs[name]:
            raise ValueError(f"class-balanced V2 and unweighted V1 {name} differ")
    output = _canonical_output(authority["calibration_output"])
    if expected_output is not None and output != str(Path(expected_output).expanduser().resolve()):
        raise ValueError("class-balanced global margin calibration output differs")
    authority["source_contrast_v21_result"] = source_record
    authority["implementation"] = record(
        authority["implementation"], label="class-balanced margin implementation"
    )
    authority["implementation_dependencies"] = verified_dependencies
    authority.update(inputs)
    authority["calibration_output"] = output
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    authority["verified_unweighted_v1"] = v1_result
    return authority


def expected_validation_checks(
    identity: Mapping[str, float | int], candidate: Mapping[str, float | int]
) -> dict[str, bool]:
    return {
        "balanced_bce_strictly_improved": (
            float(candidate["balanced_soft_binary_cross_entropy"])
            + METRIC_STRICT_IMPROVEMENT
            <= float(identity["balanced_soft_binary_cross_entropy"])
        ),
        "balanced_brier_strictly_improved": (
            float(candidate["balanced_brier"])
            + METRIC_STRICT_IMPROVEMENT
            <= float(identity["balanced_brier"])
        ),
        "hard_f1_strictly_improved": (
            float(candidate["teacher_positive_f1"])
            >= float(identity["teacher_positive_f1"])
            + METRIC_STRICT_IMPROVEMENT
        ),
        "hard_recall_strictly_improved": (
            float(candidate["teacher_positive_recall"])
            >= float(identity["teacher_positive_recall"])
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
        "rank_correlation_invariant": (
            abs(
                float(candidate["rank_correlation"])
                - float(identity["rank_correlation"])
            )
            <= RANK_NON_REGRESSION_TOLERANCE
        ),
    }


_METRIC_NAMES = {
    "pairs",
    "rank_samples",
    "brier",
    "soft_binary_cross_entropy",
    "mean_absolute_error",
    "balanced_brier",
    "balanced_soft_binary_cross_entropy",
    "rank_correlation",
    "teacher_positive_rate",
    "predicted_positive_rate",
    "teacher_positive_precision",
    "teacher_positive_recall",
    "teacher_positive_f1",
}


def _validate_metrics(
    value: object, *, label: str, expected_pairs: int
) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != _METRIC_NAMES:
        raise ValueError(f"{label} class-balanced metrics fields differ")
    metrics = dict(value)
    floats = _METRIC_NAMES - {"pairs", "rank_samples"}
    unit = {
        "brier", "mean_absolute_error", "balanced_brier",
        "teacher_positive_rate", "predicted_positive_rate",
        "teacher_positive_precision", "teacher_positive_recall",
        "teacher_positive_f1",
    }
    if (
        not isinstance(metrics["pairs"], int)
        or metrics["pairs"] != expected_pairs
        or not isinstance(metrics["rank_samples"], int)
        or metrics["rank_samples"] != RANK_SAMPLE_CAP
        or any(
            not isinstance(metrics[name], (int, float))
            or isinstance(metrics[name], bool)
            or not math.isfinite(float(metrics[name]))
            for name in floats
        )
        or any(not 0.0 <= float(metrics[name]) <= 1.0 for name in unit)
        or float(metrics["soft_binary_cross_entropy"]) < 0.0
        or float(metrics["balanced_soft_binary_cross_entropy"]) < 0.0
        or not -1.000001 <= float(metrics["rank_correlation"]) <= 1.000001
    ):
        raise ValueError(f"{label} class-balanced metrics differ")
    return metrics


def validate_calibration_result(
    value: object, *, require_promotion: bool = False
) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "status", "contract", "contract_sha256",
        "execution_authority", "input_authority", "parameters", "train_mass_weights",
        "optimization_audit", "sampling_audit", "fit_metrics", "validation_metrics",
        "promotion_checks", "query_calibration_authorized", "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("class-balanced global margin result fields differ")
    result = dict(value)
    if (
        result["schema"] != RESULT_SCHEMA
        or result["schema_version"] != SCHEMA_VERSION
        or result["status"] not in {"source_only_promoted", "source_only_complete_no_promotion"}
        or result["contract"] != calibration_contract()
        or result["contract_sha256"] != CALIBRATION_CONTRACT_SHA256
        or result["source_access"] != source_access()
    ):
        raise ValueError("class-balanced global margin result header differs")
    result["execution_authority"] = record(
        result["execution_authority"], label="class-balanced execution authority"
    )
    names = {
        "source_contrast_v21_result", "source_contrast_v21_checkpoint",
        "unweighted_v1_result", "fit_text_bank", "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    }
    inputs = result["input_authority"]
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("class-balanced global margin inputs differ")
    result["input_authority"] = {
        name: record(inputs[name], label=f"class-balanced input {name}")
        for name in sorted(names)
    }
    parameters = result["parameters"]
    if (
        not isinstance(parameters, Mapping)
        or set(parameters) != {"identity", "candidate"}
        or parameters["identity"] != {"a": IDENTITY_A, "b": IDENTITY_B}
        or not isinstance(parameters["candidate"], Mapping)
        or set(parameters["candidate"]) != {"a", "b"}
        or any(
            not isinstance(parameters["candidate"][name], (int, float))
            or isinstance(parameters["candidate"][name], bool)
            or not math.isfinite(float(parameters["candidate"][name]))
            for name in ("a", "b")
        )
        or float(parameters["candidate"]["a"]) <= MINIMUM_POSITIVE_SLOPE
    ):
        raise ValueError("class-balanced global margin parameters differ")
    weights = result["train_mass_weights"]
    weight_fields = {
        "teacher_soft_positive_rate", "teacher_soft_negative_rate",
        "positive_weight", "negative_weight",
        "weighted_positive_mass_fraction", "weighted_negative_mass_fraction",
        "derived_from_train_only", "validation_reuses_fixed_train_weights",
    }
    if not isinstance(weights, Mapping) or set(weights) != weight_fields:
        raise ValueError("class-balanced train mass weights fields differ")
    positive_rate = float(weights.get("teacher_soft_positive_rate", math.nan))
    negative_rate = float(weights.get("teacher_soft_negative_rate", math.nan))
    if (
        not 0.0 < positive_rate < 1.0
        or abs(positive_rate + negative_rate - 1.0) > 1e-12
        or abs(float(weights.get("positive_weight", math.nan)) - 0.5 / positive_rate) > 1e-10
        or abs(float(weights.get("negative_weight", math.nan)) - 0.5 / negative_rate) > 1e-10
        or abs(float(weights.get("weighted_positive_mass_fraction", math.nan)) - 0.5) > 1e-10
        or abs(float(weights.get("weighted_negative_mass_fraction", math.nan)) - 0.5) > 1e-10
        or weights.get("derived_from_train_only") is not True
        or weights.get("validation_reuses_fixed_train_weights") is not True
    ):
        raise ValueError("class-balanced train mass weights differ")
    optimization = result["optimization_audit"]
    optimization_fields = {
        "solver", "iterations", "converged", "initial_balanced_soft_binary_cross_entropy",
        "final_balanced_soft_binary_cross_entropy", "final_max_absolute_gradient",
        "minimum_observed_hessian_eigenvalue", "backtrack_reductions",
    }
    if (
        not isinstance(optimization, Mapping)
        or set(optimization) != optimization_fields
        or optimization.get("solver") != "deterministic_strictly_convex_two_parameter_newton_irls"
        or not isinstance(optimization.get("iterations"), int)
        or not 1 <= optimization["iterations"] <= MAXIMUM_NEWTON_ITERATIONS
        or optimization.get("converged") is not True
        or not isinstance(optimization.get("backtrack_reductions"), int)
        or optimization["backtrack_reductions"] < 0
        or any(
            not isinstance(optimization[name], (int, float))
            or isinstance(optimization[name], bool)
            or not math.isfinite(float(optimization[name]))
            for name in optimization_fields - {"solver", "iterations", "converged", "backtrack_reductions"}
        )
        or float(optimization["final_balanced_soft_binary_cross_entropy"])
        >= float(optimization["initial_balanced_soft_binary_cross_entropy"])
        or float(optimization["final_max_absolute_gradient"]) > OPTIMIZATION_GRADIENT_TOLERANCE
        or float(optimization["minimum_observed_hessian_eigenvalue"]) <= 1e-12
    ):
        raise ValueError("class-balanced optimization audit differs")
    sampling_expected = {
        "train_scenes": list(TRAIN_SCENES),
        "validation_scenes": list(VALIDATION_SCENES),
        "regions_per_scene": REGIONS_PER_SCENE,
        "query_rows": FIT_QUERY_ROWS,
        "canonical_negative_rows": CANONICAL_NEGATIVE_ROWS,
        "train_pairs": len(TRAIN_SCENES) * REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        "validation_pairs_per_scene": REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        "validation_contribution_to_fit_or_weights": False,
        "all_rows_and_queries_used": True,
    }
    if result["sampling_audit"] != sampling_expected:
        raise ValueError("class-balanced sampling audit differs")
    fit_metrics = result["fit_metrics"]
    if not isinstance(fit_metrics, Mapping) or set(fit_metrics) != {"identity", "candidate"}:
        raise ValueError("class-balanced fit metrics differ")
    train_pairs = len(TRAIN_SCENES) * REGIONS_PER_SCENE * FIT_QUERY_ROWS
    result["fit_metrics"] = {
        name: _validate_metrics(fit_metrics[name], label=f"fit {name}", expected_pairs=train_pairs)
        for name in ("identity", "candidate")
    }
    validation = result["validation_metrics"]
    if not isinstance(validation, Mapping) or tuple(validation) != VALIDATION_SCENES:
        raise ValueError("class-balanced validation scenes differ")
    checked_validation: dict[str, Any] = {}
    expected_scene_checks: dict[str, dict[str, bool]] = {}
    for scene in VALIDATION_SCENES:
        row = validation[scene]
        if not isinstance(row, Mapping) or set(row) != {"identity", "candidate", "checks"}:
            raise ValueError(f"class-balanced {scene} validation fields differ")
        identity = _validate_metrics(
            row["identity"], label=f"{scene} identity",
            expected_pairs=REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        )
        candidate = _validate_metrics(
            row["candidate"], label=f"{scene} candidate",
            expected_pairs=REGIONS_PER_SCENE * FIT_QUERY_ROWS,
        )
        checks = expected_validation_checks(identity, candidate)
        if row["checks"] != checks:
            raise ValueError(f"class-balanced {scene} checks differ")
        checked_validation[scene] = {"identity": identity, "candidate": candidate, "checks": checks}
        expected_scene_checks[scene] = checks
    result["validation_metrics"] = checked_validation
    expected_promotion = {
        "optimization_converged": optimization["converged"] is True,
        "candidate_slope_strictly_positive": (
            float(parameters["candidate"]["a"]) > MINIMUM_POSITIVE_SLOPE
        ),
        "train_soft_mass_exactly_class_balanced": (
            abs(float(weights["weighted_positive_mass_fraction"]) - 0.5) <= 1e-10
            and abs(float(weights["weighted_negative_mass_fraction"]) - 0.5) <= 1e-10
        ),
        "every_validation_scene_passed": all(
            all(checks.values()) for checks in expected_scene_checks.values()
        ),
    }
    promoted = all(expected_promotion.values())
    if (
        result["promotion_checks"] != expected_promotion
        or result["query_calibration_authorized"] is not promoted
        or result["status"] != ("source_only_promoted" if promoted else "source_only_complete_no_promotion")
    ):
        raise ValueError("class-balanced promotion decision differs")
    if require_promotion and not promoted:
        raise ValueError("class-balanced source calibration did not pass promotion")
    return result


__all__ = [
    "CALIBRATION_CONTRACT_SHA256", "CANONICAL_NEGATIVE_ROWS",
    "EXECUTION_AUTHORITY_SCHEMA", "FIT_QUERY_ROWS", "IDENTITY_A", "IDENTITY_B",
    "IMPLEMENTATION_DEPENDENCIES", "IMPLEMENTATION_PATH", "RESULT_SCHEMA",
    "RANK_SAMPLE_CAP", "REGIONS_PER_SCENE", "SCHEMA_VERSION", "TRAIN_SCENES",
    "VALIDATION_SCENES", "calibration_contract", "expected_validation_checks",
    "record", "source_access", "validate_calibration_result",
    "validate_execution_authority",
]
