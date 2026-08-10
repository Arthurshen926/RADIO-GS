"""Query-free anti-collapse audit for factorized-native target descriptors."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

from radio_gs.interfaces import factorized_native_target_descriptor as descriptor_formal
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


HEALTH_AUDIT_SCHEMA = "radio_gs.factorized_native_target_health_audit.v1"
HEALTH_AUDIT_SCHEMA_VERSION = 1
HEALTH_AUDIT_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/audit_factorized_native_target_descriptor_health.py"
)
PAIR_SAMPLE_CAP = 65_536
GRAM_REGION_SAMPLE_CAP = 512
PAIR_MEAN_ABSOLUTE_SLACK = 0.005
PAIR_P90_ABSOLUTE_SLACK = 0.01
CENTROID_SQUARED_NORM_ABSOLUTE_SLACK = 0.005
CENTERED_SPREAD_ABSOLUTE_SLACK = 0.005
MINIMUM_EFFECTIVE_RANK_RATIO = 0.95
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def health_access_audit() -> dict[str, bool]:
    return {
        "query_independent": True,
        "factorized_native_target_descriptor_opened": True,
        "accepted_v2_baseline_opened": True,
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
        "candidate": descriptor_formal.TARGET_DESCRIPTOR_SCHEMA,
        "baseline": "bitwise_accepted_v2_e0_for_same_canonical_regions",
        "required_input_checks": {
            "float32_cpu_finite_unit_l2": True,
            "canonical_region_identity_equal": True,
            "exact_active_fallback_masks_consistent": True,
            "fallback_descriptor_bitwise_equal_to_accepted": True,
        },
        "statistics": {
            "centroid_squared_norm": "full_region_axis",
            "pair_cosine": {
                "sample": "fixed_modular_ordered_distinct_region_pairs",
                "cap": PAIR_SAMPLE_CAP,
                "reducers": ["mean", "p90"],
            },
            "centered_spread": "full_mean_squared_radius_about_full_centroid",
            "centered_gram": {
                "sample": "fixed_evenly_spaced_region_rows",
                "cap": GRAM_REGION_SAMPLE_CAP,
                "effective_rank": "trace_squared_over_frobenius_squared",
            },
        },
        "pass_definition": {
            "candidate_centroid_squared_norm_at_most_baseline_plus": (
                CENTROID_SQUARED_NORM_ABSOLUTE_SLACK
            ),
            "candidate_pair_mean_at_most_baseline_plus": PAIR_MEAN_ABSOLUTE_SLACK,
            "candidate_pair_p90_at_most_baseline_plus": PAIR_P90_ABSOLUTE_SLACK,
            "candidate_centered_spread_plus_at_least_baseline": (
                CENTERED_SPREAD_ABSOLUTE_SLACK
            ),
            "candidate_centered_gram_effective_rank_at_least_baseline_ratio": (
                MINIMUM_EFFECTIVE_RANK_RATIO
            ),
            "all_checks_required": True,
        },
        "threshold_origin": (
            "global_method_level_fixed_before_target_audit;"
            "not_fit_from_scene_or_benchmark_values"
        ),
        "scene_parameters": False,
        "query_authority_requires_pass": True,
        "legacy_evaluator_changed": False,
        "access_audit": health_access_audit(),
    }


HEALTH_CONTRACT_SHA256 = canonical_json_sha256(health_contract())


def record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _validate_statistics(value: object, *, label: str) -> dict[str, float | int]:
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
        raise ValueError(f"{label} health statistics fields differ")
    result = dict(value)
    if (
        not isinstance(result["regions"], int)
        or result["regions"] < 2
        or not isinstance(result["pair_samples"], int)
        or result["pair_samples"] <= 0
        or result["pair_samples"] > PAIR_SAMPLE_CAP
        or not isinstance(result["gram_region_samples"], int)
        or result["gram_region_samples"] <= 1
        or result["gram_region_samples"] > GRAM_REGION_SAMPLE_CAP
        or any(
            not _finite_number(result[name])
            for name in required
            if name not in {"regions", "pair_samples", "gram_region_samples"}
        )
        or float(result["centroid_squared_norm"]) < 0.0
        or float(result["centroid_squared_norm"]) > 1.0005
        or float(result["pair_cosine_mean"]) < -1.0005
        or float(result["pair_cosine_mean"]) > 1.0005
        or float(result["pair_cosine_p90"]) < -1.0005
        or float(result["pair_cosine_p90"]) > 1.0005
        or float(result["centered_mean_squared_radius"]) < 0.0
        or float(result["centered_gram_effective_rank"]) < 1.0
        or float(result["centered_gram_effective_rank"])
        > float(result["gram_region_samples"]) + 1e-3
    ):
        raise ValueError(f"{label} health statistics differ")
    return result


def expected_checks(
    candidate: Mapping[str, float | int], baseline: Mapping[str, float | int]
) -> dict[str, bool]:
    return {
        "centroid_squared_norm_not_more_collapsed": (
            float(candidate["centroid_squared_norm"])
            <= float(baseline["centroid_squared_norm"])
            + CENTROID_SQUARED_NORM_ABSOLUTE_SLACK
        ),
        "sampled_pair_mean_not_more_collapsed": (
            float(candidate["pair_cosine_mean"])
            <= float(baseline["pair_cosine_mean"])
            + PAIR_MEAN_ABSOLUTE_SLACK
        ),
        "sampled_pair_p90_not_more_collapsed": (
            float(candidate["pair_cosine_p90"])
            <= float(baseline["pair_cosine_p90"])
            + PAIR_P90_ABSOLUTE_SLACK
        ),
        "centered_spread_not_smaller": (
            float(candidate["centered_mean_squared_radius"])
            + CENTERED_SPREAD_ABSOLUTE_SLACK
            >= float(baseline["centered_mean_squared_radius"])
        ),
        "centered_gram_effective_rank_not_materially_smaller": (
            float(candidate["centered_gram_effective_rank"])
            >= MINIMUM_EFFECTIVE_RANK_RATIO
            * float(baseline["centered_gram_effective_rank"])
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
        "baseline_channel_sha256",
        "alignment_audit",
        "candidate_statistics",
        "baseline_statistics",
        "checks",
        "query_authority_eligible",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("factorized-native health audit fields differ")
    audit = dict(value)
    if (
        audit["schema"] != HEALTH_AUDIT_SCHEMA
        or audit["schema_version"] != HEALTH_AUDIT_SCHEMA_VERSION
        or audit["contract"] != health_contract()
        or audit["contract_sha256"] != HEALTH_CONTRACT_SHA256
        or audit["status"] not in {"pass", "reject_more_collapsed_than_accepted"}
        or not isinstance(audit["scene_id"], str)
        or not audit["scene_id"]
        or not isinstance(audit["physical_space_id"], str)
        or not audit["physical_space_id"]
        or audit["access_audit"] != health_access_audit()
    ):
        raise ValueError("factorized-native health audit header differs")
    audit["producer"] = record(audit["producer"], label="health audit producer")
    inputs = audit["input_authority"]
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "target_descriptor",
        "accepted_v2_baseline",
    }:
        raise ValueError("factorized-native health audit inputs differ")
    audit["input_authority"] = {
        name: record(inputs[name], label=f"health audit {name}")
        for name in sorted(inputs)
    }
    for name in ("descriptor_channel_sha256", "baseline_channel_sha256"):
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
            raise ValueError(f"factorized-native health {name} differs")
    alignment = audit["alignment_audit"]
    expected_alignment = {
        "scene_and_physical_space_equal": True,
        "canonical_region_indices_equal": True,
        "region_fingerprints_equal": True,
        "accepted_input_record_equal": True,
        "exact_active_masks_equal": True,
        "fallback_mask_complement_active": True,
        "fallback_descriptor_bitwise_equal": True,
        "candidate_and_baseline_unit_l2_finite": True,
    }
    if alignment != expected_alignment:
        raise ValueError("factorized-native health alignment audit differs")
    candidate = _validate_statistics(
        audit["candidate_statistics"], label="candidate"
    )
    baseline = _validate_statistics(audit["baseline_statistics"], label="baseline")
    if (
        candidate["regions"] != baseline["regions"]
        or candidate["pair_samples"] != baseline["pair_samples"]
        or candidate["gram_region_samples"] != baseline["gram_region_samples"]
    ):
        raise ValueError("factorized-native health sampling axes differ")
    checks = expected_checks(candidate, baseline)
    eligible = all(checks.values())
    if (
        audit["checks"] != checks
        or audit["query_authority_eligible"] is not eligible
        or audit["status"]
        != ("pass" if eligible else "reject_more_collapsed_than_accepted")
    ):
        raise ValueError("factorized-native health decision differs")
    if require_pass and not eligible:
        raise ValueError(
            "factorized-native descriptor failed query-free anti-collapse gate"
        )
    audit["candidate_statistics"] = candidate
    audit["baseline_statistics"] = baseline
    return audit


__all__ = [
    "CENTERED_SPREAD_ABSOLUTE_SLACK",
    "CENTROID_SQUARED_NORM_ABSOLUTE_SLACK",
    "GRAM_REGION_SAMPLE_CAP",
    "HEALTH_AUDIT_SCHEMA",
    "HEALTH_AUDIT_SCHEMA_VERSION",
    "HEALTH_AUDIT_IMPLEMENTATION_PATH",
    "HEALTH_CONTRACT_SHA256",
    "MINIMUM_EFFECTIVE_RANK_RATIO",
    "PAIR_MEAN_ABSOLUTE_SLACK",
    "PAIR_P90_ABSOLUTE_SLACK",
    "PAIR_SAMPLE_CAP",
    "expected_checks",
    "health_access_audit",
    "health_contract",
    "record",
    "validate_health_audit",
]
