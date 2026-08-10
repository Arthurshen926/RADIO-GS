"""Preregistered source-student-envelope health gate for contrast V2.1.

V2 and V3 remain immutable diagnostics.  V4 uses the deployed, source-promoted
step-60 student as its source-only dispersion reference and mechanically binds
the frozen preregistration, result, and checkpoint singletons.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as descriptor_formal
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, load_json_object


HEALTH_AUDIT_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_target_health_audit.v4"
)
HEALTH_AUDIT_SCHEMA_VERSION = 4
HEALTH_AUDIT_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/audit_factorized_native_contrast_v21_target_descriptor_health_v4.py"
)
PREREGISTRATION_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_source_student_envelope_"
    "health_preregistration.v1"
)
PREREGISTRATION_RECORD = {
    "path": str(
        Path(__file__).resolve().parents[2]
        / "paper/artifacts/factorized_native_contrast_v21_"
        "source_student_envelope_health_v4_preregistration_20260807.json"
    ),
    "sha256": "a1b04ce679e45388c2d2addde2f9a1f4cc62a4a7b1def1b1d3c5d3e7f0d8fc0f",
}
SOURCE_EXECUTION_AUTHORITY_RECORD = {
    "path": (
        "/root/RADIO-GS/paper/artifacts/"
        "factorized_native_gauge_state_readout_exact4x2_"
        "contrast_v21_execution_authority_20260807.json"
    ),
    "sha256": "f24008d976067a5b0ee42eb2b9cf3e3276c11199c82b68754f5c000a5166c171",
}
SOURCE_RESULT_RECORD = {
    "path": (
        "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/"
        "factorized_native_gauge_state_exact4x2/contrast_v21_direction_only/"
        "model.pt.json"
    ),
    "sha256": "bc264f50c33a20199201875eba638ee36c654b888d270d32d3259beeebd5debc",
}
SOURCE_CHECKPOINT_RECORD = {
    "path": (
        "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/"
        "factorized_native_gauge_state_exact4x2/contrast_v21_direction_only/"
        "model.pt"
    ),
    "sha256": "952e1d841969c8d2d48586aa8d08a59d4e28fadea10b56d4c17557df1c434185",
}
MAXIMUM_CENTROID_SQUARED_NORM = 0.9507060856165268
MAXIMUM_PAIR_COSINE_MEAN = 0.9499539540409823
MAXIMUM_PAIR_COSINE_P90 = 1.0
MINIMUM_CENTERED_MEAN_SQUARED_RADIUS = 0.044470436018765095
MINIMUM_CENTERED_GRAM_EFFECTIVE_RANK = 1.9454990424664538
PAIR_SAMPLE_CAP = 65_536
GRAM_REGION_SAMPLE_CAP = 512
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def frozen_gate() -> dict[str, Any]:
    return {
        "maximum_centroid_squared_norm": MAXIMUM_CENTROID_SQUARED_NORM,
        "maximum_pair_cosine_mean": MAXIMUM_PAIR_COSINE_MEAN,
        "maximum_pair_cosine_p90": MAXIMUM_PAIR_COSINE_P90,
        "minimum_centered_mean_squared_radius": (
            MINIMUM_CENTERED_MEAN_SQUARED_RADIUS
        ),
        "minimum_centered_gram_effective_rank": (
            MINIMUM_CENTERED_GRAM_EFFECTIVE_RANK
        ),
        "derivation": {
            "centroid_and_pair_mean": (
                "worst_source_validation_student_plus_fixed_0p01_absolute_slack"
            ),
            "pair_p90": (
                "min_1p0_of_worst_source_validation_student_plus_fixed_0p01_absolute_slack"
            ),
            "spread": "0p75_times_minimum_source_validation_student_spread",
            "effective_rank": (
                "0p75_times_minimum_source_validation_student_effective_rank"
            ),
        },
        "all_five_checks_required": True,
        "scene_specific_parameters": False,
        "target_statistics_used_to_define_thresholds": False,
    }


def validate_preregistration(
    path: str | Path = PREREGISTRATION_RECORD["path"],
    *,
    expected_sha256: str = PREREGISTRATION_RECORD["sha256"],
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="contrast V2.1 health V4 preregistration",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "decision_rationale",
        "source_only_inputs",
        "deterministic_statistics_contract",
        "source_student_statistics",
        "frozen_global_gate",
        "required_alignment_invariants",
        "execution_policy",
        "contamination_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("health V4 preregistration fields differ")
    prereg = dict(raw)
    source_inputs = prereg.get("source_only_inputs")
    statistics = prereg.get("deterministic_statistics_contract")
    expected_student_statistics = {
        "scene0004_00": {
            "centroid_squared_norm": 0.9407060856165268,
            "pair_cosine_mean": 0.939670051078352,
            "pair_cosine_p90": 0.9900809915592366,
            "centered_mean_squared_radius": 0.059293914691686794,
            "centered_gram_effective_rank": 2.593998723288605,
        },
        "scene0008_00": {
            "centroid_squared_norm": 0.9397602369878126,
            "pair_cosine_mean": 0.9399539540409823,
            "pair_cosine_p90": 0.9851797821955808,
            "centered_mean_squared_radius": 0.0602397613450751,
            "centered_gram_effective_rank": 2.7787211303060215,
        },
    }
    if (
        {"path": str(source), "sha256": digest} != PREREGISTRATION_RECORD
        or prereg.get("schema") != PREREGISTRATION_SCHEMA
        or prereg.get("schema_version") != 1
        or prereg.get("status")
        != "preregistered_after_teacher_envelope_rank_mismatch_audit_before_any_query_or_metric_access"
        or prereg.get("decision_rationale")
        != {
            "deployed_candidate_is_the_source_promoted_student_not_the_official_teacher": True,
            "teacher_envelope_effective_rank_rejected_the_source_promoted_student_itself": True,
            "deployment_health_must_use_the_source_promoted_student_dispersion_envelope": True,
            "teacher_fidelity_remains_enforced_by_the_unchanged_source_v21_promotion_gate": True,
            "target_values_are_not_used_in_any_threshold_derivation": True,
        }
        or not isinstance(source_inputs, Mapping)
        or set(source_inputs)
        != {
            "contrast_v21_execution_authority",
            "contrast_v21_source_result",
            "contrast_v21_checkpoint",
            "source_validation_scenes",
            "target_threshold_fitting",
            "query_or_text_opened",
            "benchmark_metric_opened",
        }
        or source_inputs.get("contrast_v21_execution_authority")
        != SOURCE_EXECUTION_AUTHORITY_RECORD
        or source_inputs.get("contrast_v21_source_result") != SOURCE_RESULT_RECORD
        or source_inputs.get("contrast_v21_checkpoint")
        != SOURCE_CHECKPOINT_RECORD
        or source_inputs.get("source_validation_scenes")
        != ["scene0004_00", "scene0008_00"]
        or source_inputs.get("target_threshold_fitting") is not False
        or source_inputs.get("query_or_text_opened") is not False
        or source_inputs.get("benchmark_metric_opened") is not False
        or not isinstance(statistics, Mapping)
        or statistics
        != {
            "descriptor": (
                "source_promoted_step60_student_descriptor_per_eligible_canonical_region"
            ),
            "regions_per_scene": 4096,
            "pair_axis": "fixed_modular_ordered_distinct_region_pairs_cap65536",
            "gram_axis": "fixed_evenly_spaced_region_rows_cap512",
            "centered_spread": "full_mean_squared_radius_about_full_centroid",
            "effective_rank": "trace_squared_over_frobenius_squared",
        }
        or prereg.get("source_student_statistics")
        != expected_student_statistics
        or prereg.get("frozen_global_gate") != frozen_gate()
        or prereg.get("required_alignment_invariants")
        != {
            "float32_cpu_finite_unit_l2": True,
            "canonical_region_identity_equal_to_accepted_v2": True,
            "exact_active_fallback_masks_consistent": True,
            "fallback_descriptor_bitwise_equal_to_accepted_v2": True,
        }
        or prereg.get("execution_policy")
        != {
            "accepted_relative_health_v2_and_teacher_envelope_health_v3_remain_immutable_diagnostics": True,
            "source_student_envelope_health_v4_is_the_query_authority_hard_gate": True,
            "query_manifest_or_text_must_not_open_before_v4_pass": True,
            "benchmark_metric_must_not_run_before_v4_pass": True,
            "single_target_candidate_no_threshold_sweep": True,
        }
        or prereg.get("contamination_audit")
        != {
            "figurines_health_v3_statistics_had_been_observed_before_this_preregistration": True,
            "figurines_statistics_used_in_threshold_formula": False,
            "all_numeric_thresholds_are_mechanical_functions_of_the_two_source_validation_student_statistics": True,
            "no_query_ground_truth_or_metric_had_been_opened": True,
        }
    ):
        raise ValueError("health V4 preregistration contract differs")
    prereg["verified_record"] = dict(PREREGISTRATION_RECORD)
    return prereg


def health_access_audit() -> dict[str, bool]:
    return {
        "query_independent": True,
        "preregistration_opened": True,
        "contrast_v21_source_result_and_checkpoint_opened": True,
        "target_descriptor_opened": True,
        "accepted_v2_baseline_opened": True,
        "health_v2_and_v3_diagnostics_opened": True,
        "benchmark_queries_opened": False,
        "text_queries_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def health_contract() -> dict[str, Any]:
    return {
        "schema": HEALTH_AUDIT_SCHEMA,
        "schema_version": HEALTH_AUDIT_SCHEMA_VERSION,
        "candidate_schema": descriptor_formal.TARGET_DESCRIPTOR_SCHEMA,
        "preregistration": dict(PREREGISTRATION_RECORD),
        "source_result": dict(SOURCE_RESULT_RECORD),
        "source_checkpoint": dict(SOURCE_CHECKPOINT_RECORD),
        "health_v2_and_v3": "mandatory_immutable_diagnostics_not_hard_gates",
        "statistics": {
            "regions": 4096,
            "pair_axis": "fixed_modular_ordered_distinct_region_pairs_cap65536",
            "gram_axis": "fixed_evenly_spaced_region_rows_cap512",
            "centered_spread": "full_mean_squared_radius_about_full_centroid",
            "effective_rank": "trace_squared_over_frobenius_squared",
        },
        "frozen_global_gate": frozen_gate(),
        "all_five_checks_required": True,
        "scene_parameters": False,
        "target_statistics_used_to_define_thresholds": False,
        "query_authority_requires_pass": True,
        "access_audit": health_access_audit(),
    }


HEALTH_CONTRACT_SHA256 = canonical_json_sha256(health_contract())


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def validate_statistics(value: object, *, label: str) -> dict[str, float | int]:
    required = {
        "regions",
        "pair_samples",
        "gram_region_samples",
        "centroid_squared_norm",
        "pair_cosine_mean",
        "pair_cosine_p90",
        "centered_mean_squared_radius",
        "centered_gram_effective_rank",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} health V4 statistics fields differ")
    result = dict(value)
    if (
        result["regions"] != 4096
        or result["pair_samples"] != PAIR_SAMPLE_CAP
        or result["gram_region_samples"] != GRAM_REGION_SAMPLE_CAP
        or any(
            not _finite_number(result[name])
            for name in required
            if name not in {"regions", "pair_samples", "gram_region_samples"}
        )
        or not 0.0 <= float(result["centroid_squared_norm"]) <= 1.0005
        or not -1.0005 <= float(result["pair_cosine_mean"]) <= 1.0005
        or not -1.0005 <= float(result["pair_cosine_p90"]) <= 1.0005
        or float(result["centered_mean_squared_radius"]) < 0.0
        or not 1.0
        <= float(result["centered_gram_effective_rank"])
        <= GRAM_REGION_SAMPLE_CAP + 1e-3
    ):
        raise ValueError(f"{label} health V4 statistics differ")
    return result


def expected_checks(candidate: Mapping[str, float | int]) -> dict[str, bool]:
    return {
        "centroid_squared_norm_within_source_student_envelope": (
            float(candidate["centroid_squared_norm"])
            <= MAXIMUM_CENTROID_SQUARED_NORM
        ),
        "pair_cosine_mean_within_source_student_envelope": (
            float(candidate["pair_cosine_mean"]) <= MAXIMUM_PAIR_COSINE_MEAN
        ),
        "pair_cosine_p90_within_source_student_envelope": (
            float(candidate["pair_cosine_p90"]) <= MAXIMUM_PAIR_COSINE_P90
        ),
        "centered_spread_within_source_student_envelope": (
            float(candidate["centered_mean_squared_radius"])
            >= MINIMUM_CENTERED_MEAN_SQUARED_RADIUS
        ),
        "centered_gram_effective_rank_within_source_student_envelope": (
            float(candidate["centered_gram_effective_rank"])
            >= MINIMUM_CENTERED_GRAM_EFFECTIVE_RANK
        ),
    }


def validate_health_audit(value: object, *, require_pass: bool = False) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "status",
        "scene_id",
        "physical_space_id",
        "producer",
        "input_authority",
        "descriptor_channel_sha256",
        "accepted_v2_channel_sha256",
        "alignment_audit",
        "accepted_relative_health_v2_status",
        "teacher_envelope_health_v3_status",
        "candidate_statistics",
        "accepted_v2_diagnostic_statistics",
        "frozen_global_gate",
        "checks",
        "query_authority_eligible",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("contrast V2.1 health V4 audit fields differ")
    audit = dict(value)
    if (
        audit["schema"] != HEALTH_AUDIT_SCHEMA
        or audit["schema_version"] != HEALTH_AUDIT_SCHEMA_VERSION
        or audit["contract"] != health_contract()
        or audit["contract_sha256"] != HEALTH_CONTRACT_SHA256
        or audit["status"]
        not in {"pass", "reject_outside_source_student_envelope"}
        or not isinstance(audit["scene_id"], str)
        or not audit["scene_id"]
        or not isinstance(audit["physical_space_id"], str)
        or not audit["physical_space_id"]
        or audit["accepted_relative_health_v2_status"]
        not in {"pass", "reject_more_collapsed_than_accepted"}
        or audit["teacher_envelope_health_v3_status"]
        not in {"pass", "reject_outside_source_teacher_envelope"}
        or audit["frozen_global_gate"] != frozen_gate()
        or audit["access_audit"] != health_access_audit()
    ):
        raise ValueError("contrast V2.1 health V4 audit header differs")
    audit["producer"] = record(audit["producer"], label="health V4 producer")
    inputs = audit.get("input_authority")
    names = {
        "preregistration",
        "contrast_v21_source_result",
        "contrast_v21_source_checkpoint",
        "target_descriptor",
        "accepted_v2_baseline",
        "accepted_relative_health_v2_diagnostic",
        "teacher_envelope_health_v3_diagnostic",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("contrast V2.1 health V4 inputs differ")
    audit["input_authority"] = {
        name: record(inputs[name], label=f"health V4 {name}")
        for name in sorted(names)
    }
    if (
        audit["input_authority"]["preregistration"] != PREREGISTRATION_RECORD
        or audit["input_authority"]["contrast_v21_source_result"]
        != SOURCE_RESULT_RECORD
        or audit["input_authority"]["contrast_v21_source_checkpoint"]
        != SOURCE_CHECKPOINT_RECORD
    ):
        raise ValueError("contrast V2.1 health V4 frozen lineage differs")
    for name in ("descriptor_channel_sha256", "accepted_v2_channel_sha256"):
        channels = audit[name]
        if (
            not isinstance(channels, Mapping)
            or not channels
            or any(
                not isinstance(key, str)
                or not key
                or _SHA256.fullmatch(str(digest)) is None
                for key, digest in channels.items()
            )
        ):
            raise ValueError(f"contrast V2.1 health V4 {name} differs")
    expected_alignment = {
        "scene_and_physical_space_equal": True,
        "canonical_region_indices_equal": True,
        "region_fingerprints_equal": True,
        "accepted_input_record_equal": True,
        "source_result_and_checkpoint_records_equal": True,
        "health_v2_and_v3_descriptor_baseline_records_equal": True,
        "health_v2_and_v3_candidate_statistics_equal": True,
        "exact_active_masks_equal": True,
        "fallback_mask_complement_active": True,
        "fallback_descriptor_bitwise_equal": True,
        "candidate_and_accepted_unit_l2_finite": True,
    }
    if audit["alignment_audit"] != expected_alignment:
        raise ValueError("contrast V2.1 health V4 alignment differs")
    candidate = validate_statistics(audit["candidate_statistics"], label="candidate")
    baseline = validate_statistics(
        audit["accepted_v2_diagnostic_statistics"], label="AcceptedV2 diagnostic"
    )
    checks = expected_checks(candidate)
    eligible = all(checks.values())
    if (
        audit["checks"] != checks
        or audit["query_authority_eligible"] is not eligible
        or audit["status"]
        != ("pass" if eligible else "reject_outside_source_student_envelope")
    ):
        raise ValueError("contrast V2.1 health V4 decision differs")
    if require_pass and not eligible:
        raise ValueError(
            "contrast V2.1 descriptor failed source-student-envelope health gate"
        )
    audit["candidate_statistics"] = candidate
    audit["accepted_v2_diagnostic_statistics"] = baseline
    return audit


def validate_query_gate_binding(
    audit_value: object,
    descriptor_record: object,
    descriptor_value: object,
    source_result_record: object,
    source_checkpoint_record: object,
    health_preregistration_record: object,
) -> dict[str, Any]:
    """Validate the exact V4 PASS lineage required before opening query text."""

    audit = validate_health_audit(audit_value, require_pass=True)
    descriptor = descriptor_formal.validate_target_descriptor_authority(
        descriptor_value
    )
    shaped_descriptor = record(descriptor_record, label="query-gate descriptor")
    shaped_source = record(source_result_record, label="query-gate source result")
    shaped_checkpoint = record(
        source_checkpoint_record, label="query-gate source checkpoint"
    )
    shaped_prereg = record(
        health_preregistration_record, label="query-gate preregistration"
    )
    if (
        audit["input_authority"]["target_descriptor"] != shaped_descriptor
        or audit["input_authority"]["contrast_v21_source_result"] != shaped_source
        or audit["input_authority"]["contrast_v21_source_checkpoint"]
        != shaped_checkpoint
        or audit["input_authority"]["preregistration"] != shaped_prereg
        or shaped_source != SOURCE_RESULT_RECORD
        or shaped_checkpoint != SOURCE_CHECKPOINT_RECORD
        or shaped_prereg != PREREGISTRATION_RECORD
        or audit["scene_id"] != descriptor["scene_id"]
        or audit["physical_space_id"] != descriptor["physical_space_id"]
        or audit["descriptor_channel_sha256"] != descriptor["channel_sha256"]
        or descriptor["input_authority"]["source_contrast_v21_result"]
        != SOURCE_RESULT_RECORD
        or descriptor["input_authority"]["source_contrast_v21_checkpoint"]
        != SOURCE_CHECKPOINT_RECORD
    ):
        raise ValueError("contrast V2.1 query health-V4 lineage differs")
    return audit


__all__ = [
    "GRAM_REGION_SAMPLE_CAP",
    "HEALTH_AUDIT_IMPLEMENTATION_PATH",
    "HEALTH_AUDIT_SCHEMA",
    "HEALTH_AUDIT_SCHEMA_VERSION",
    "HEALTH_CONTRACT_SHA256",
    "PAIR_SAMPLE_CAP",
    "PREREGISTRATION_RECORD",
    "SOURCE_CHECKPOINT_RECORD",
    "SOURCE_EXECUTION_AUTHORITY_RECORD",
    "SOURCE_RESULT_RECORD",
    "expected_checks",
    "frozen_gate",
    "health_access_audit",
    "health_contract",
    "record",
    "validate_health_audit",
    "validate_preregistration",
    "validate_query_gate_binding",
    "validate_statistics",
]
