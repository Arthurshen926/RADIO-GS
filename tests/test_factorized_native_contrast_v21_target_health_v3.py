from __future__ import annotations

import pytest

from radio_gs.interfaces import factorized_native_contrast_v21_target_health_v3 as formal


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def _statistics(*, effective_rank: float = 8.0) -> dict[str, float | int]:
    return {
        "regions": 4096,
        "pair_samples": 65536,
        "gram_region_samples": 512,
        "centroid_squared_norm": 0.93,
        "pair_cosine_mean": 0.93,
        "pair_cosine_p90": 0.98,
        "centered_mean_squared_radius": 0.07,
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
        "status": "pass" if eligible else "reject_outside_source_teacher_envelope",
        "scene_id": "figurines",
        "physical_space_id": "lerf3d:figurines:physical",
        "producer": _record("a"),
        "input_authority": {
            "preregistration": dict(formal.PREREGISTRATION_RECORD),
            "contrast_v21_source_result": dict(formal.SOURCE_RESULT_RECORD),
            "target_descriptor": _record("b"),
            "accepted_v2_baseline": _record("c"),
            "accepted_relative_health_v2_diagnostic": _record("d"),
        },
        "descriptor_channel_sha256": {"semantic_descriptor": "e" * 64},
        "accepted_v2_channel_sha256": {"accepted_v2_e0": "f" * 64},
        "alignment_audit": {
            "scene_and_physical_space_equal": True,
            "canonical_region_indices_equal": True,
            "region_fingerprints_equal": True,
            "accepted_input_record_equal": True,
            "source_result_record_equal": True,
            "health_v2_descriptor_and_baseline_records_equal": True,
            "health_v2_candidate_statistics_equal": True,
            "exact_active_masks_equal": True,
            "fallback_mask_complement_active": True,
            "fallback_descriptor_bitwise_equal": True,
            "candidate_and_accepted_unit_l2_finite": True,
        },
        "accepted_relative_health_v2_status": "reject_more_collapsed_than_accepted",
        "candidate_statistics": candidate,
        "accepted_v2_diagnostic_statistics": _statistics(effective_rank=12.0),
        "frozen_global_gate": formal.frozen_gate(),
        "checks": checks,
        "query_authority_eligible": eligible,
        "access_audit": formal.health_access_audit(),
    }


def test_frozen_preregistration_is_exact_and_target_blind() -> None:
    prereg = formal.validate_preregistration()
    assert prereg["verified_record"] == formal.PREREGISTRATION_RECORD
    assert prereg["frozen_global_gate"] == formal.frozen_gate()
    assert prereg["execution_policy"]["single_target_candidate_no_threshold_sweep"]


def test_all_five_source_teacher_envelope_checks_are_required() -> None:
    candidate = _statistics(effective_rank=3.2)
    payload = _audit(candidate)
    checked = formal.validate_health_audit(payload)
    assert checked["status"] == "reject_outside_source_teacher_envelope"
    assert checked["checks"][
        "centered_gram_effective_rank_within_source_teacher_envelope"
    ] is False
    assert sum(checked["checks"].values()) == 4
    with pytest.raises(ValueError, match="source-teacher-envelope"):
        formal.validate_health_audit(payload, require_pass=True)


def test_perfect_source_teacher_like_candidate_passes() -> None:
    payload = _audit(_statistics())
    checked = formal.validate_health_audit(payload, require_pass=True)
    assert checked["status"] == "pass"
    assert all(checked["checks"].values())
