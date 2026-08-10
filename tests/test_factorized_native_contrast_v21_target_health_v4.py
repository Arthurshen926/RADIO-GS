from __future__ import annotations

import inspect

import pytest

from radio_gs.interfaces import factorized_native_contrast_v21_target_health_v4 as formal


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def _statistics(*, effective_rank: float = 3.17) -> dict[str, float | int]:
    return {
        "regions": 4096,
        "pair_samples": 65536,
        "gram_region_samples": 512,
        "centroid_squared_norm": 0.917,
        "pair_cosine_mean": 0.917,
        "pair_cosine_p90": 0.982,
        "centered_mean_squared_radius": 0.083,
        "centered_gram_effective_rank": effective_rank,
    }


def _audit(candidate: dict) -> dict:
    checks = formal.expected_checks(candidate)
    eligible = all(checks.values())
    return {
        "schema": formal.HEALTH_AUDIT_SCHEMA,
        "schema_version": formal.HEALTH_AUDIT_SCHEMA_VERSION,
        "contract": formal.health_contract(),
        "contract_sha256": formal.HEALTH_CONTRACT_SHA256,
        "status": "pass" if eligible else "reject_outside_source_student_envelope",
        "scene_id": "figurines",
        "physical_space_id": "lerf3d:figurines:physical",
        "producer": _record("a"),
        "input_authority": {
            "preregistration": dict(formal.PREREGISTRATION_RECORD),
            "contrast_v21_source_result": dict(formal.SOURCE_RESULT_RECORD),
            "contrast_v21_source_checkpoint": dict(formal.SOURCE_CHECKPOINT_RECORD),
            "target_descriptor": _record("b"),
            "accepted_v2_baseline": _record("c"),
            "accepted_relative_health_v2_diagnostic": _record("d"),
            "teacher_envelope_health_v3_diagnostic": _record("e"),
        },
        "descriptor_channel_sha256": {"semantic_descriptor": "f" * 64},
        "accepted_v2_channel_sha256": {"accepted_v2_e0": "1" * 64},
        "alignment_audit": {
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
        },
        "accepted_relative_health_v2_status": "reject_more_collapsed_than_accepted",
        "teacher_envelope_health_v3_status": "reject_outside_source_teacher_envelope",
        "candidate_statistics": candidate,
        "accepted_v2_diagnostic_statistics": {
            **_statistics(effective_rank=12.0),
            "centroid_squared_norm": 0.69,
            "pair_cosine_mean": 0.69,
            "pair_cosine_p90": 0.86,
            "centered_mean_squared_radius": 0.31,
        },
        "frozen_global_gate": formal.frozen_gate(),
        "checks": checks,
        "query_authority_eligible": eligible,
        "access_audit": formal.health_access_audit(),
    }


def test_preregistration_exactly_binds_source_student_singletons() -> None:
    prereg = formal.validate_preregistration()
    assert prereg["verified_record"] == formal.PREREGISTRATION_RECORD
    assert prereg["source_only_inputs"]["contrast_v21_source_result"] == formal.SOURCE_RESULT_RECORD
    assert prereg["source_only_inputs"]["contrast_v21_checkpoint"] == formal.SOURCE_CHECKPOINT_RECORD
    assert prereg["contamination_audit"]["figurines_statistics_used_in_threshold_formula"] is False


def test_figurines_like_statistics_pass_all_source_student_checks() -> None:
    payload = _audit(_statistics())
    checked = formal.validate_health_audit(payload, require_pass=True)
    assert checked["status"] == "pass"
    assert all(checked["checks"].values())


def test_effective_rank_below_frozen_source_student_floor_rejects() -> None:
    payload = _audit(_statistics(effective_rank=1.5))
    checked = formal.validate_health_audit(payload)
    assert checked["status"] == "reject_outside_source_student_envelope"
    with pytest.raises(ValueError, match="source-student-envelope"):
        formal.validate_health_audit(payload, require_pass=True)


def test_query_gate_binding_exports_required_checkpoint_lineage_argument() -> None:
    assert list(inspect.signature(formal.validate_query_gate_binding).parameters) == [
        "audit_value",
        "descriptor_record",
        "descriptor_value",
        "source_result_record",
        "source_checkpoint_record",
        "health_preregistration_record",
    ]
