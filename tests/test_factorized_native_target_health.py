from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_target_health as formal
from radio_gs.scripts.audit_factorized_native_target_descriptor_health import (
    descriptor_statistics,
    deterministic_gram_axis,
    deterministic_pair_axis,
)


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def _audit_payload(candidate: dict, baseline: dict) -> dict:
    checks = formal.expected_checks(candidate, baseline)
    eligible = all(checks.values())
    return {
        "schema": formal.HEALTH_AUDIT_SCHEMA,
        "schema_version": formal.HEALTH_AUDIT_SCHEMA_VERSION,
        "contract": formal.health_contract(),
        "contract_sha256": formal.HEALTH_CONTRACT_SHA256,
        "status": "pass" if eligible else "reject_more_collapsed_than_accepted",
        "scene_id": "synthetic",
        "physical_space_id": "synthetic:physical",
        "producer": _record("a"),
        "input_authority": {
            "target_descriptor": _record("b"),
            "accepted_v2_baseline": _record("c"),
        },
        "descriptor_channel_sha256": {"semantic_descriptor": "d" * 64},
        "baseline_channel_sha256": {"accepted_v2_e0": "e" * 64},
        "alignment_audit": {
            "scene_and_physical_space_equal": True,
            "canonical_region_indices_equal": True,
            "region_fingerprints_equal": True,
            "accepted_input_record_equal": True,
            "exact_active_masks_equal": True,
            "fallback_mask_complement_active": True,
            "fallback_descriptor_bitwise_equal": True,
            "candidate_and_baseline_unit_l2_finite": True,
        },
        "candidate_statistics": candidate,
        "baseline_statistics": baseline,
        "checks": checks,
        "query_authority_eligible": eligible,
        "access_audit": formal.health_access_audit(),
    }


def test_deterministic_sampling_axes_are_fixed_and_self_pair_free() -> None:
    first_a, second_a = deterministic_pair_axis(97)
    first_b, second_b = deterministic_pair_axis(97)
    assert torch.equal(first_a, first_b)
    assert torch.equal(second_a, second_b)
    assert not bool((first_a == second_a).any())
    gram = deterministic_gram_axis(4096)
    assert gram.numel() == formal.GRAM_REGION_SAMPLE_CAP
    assert gram.unique().numel() == gram.numel()


def test_identical_candidate_passes_query_free_anti_collapse_gate() -> None:
    torch.manual_seed(31)
    baseline = F.normalize(torch.randn(96, 1536), dim=-1)
    baseline_stats = descriptor_statistics(baseline)
    payload = _audit_payload(dict(baseline_stats), dict(baseline_stats))
    checked = formal.validate_health_audit(payload, require_pass=True)
    assert checked["status"] == "pass"
    assert all(checked["checks"].values())


def test_common_direction_collapse_is_rejected_without_scene_thresholds() -> None:
    torch.manual_seed(37)
    baseline = F.normalize(torch.randn(128, 1536), dim=-1)
    common = F.normalize(torch.randn(1, 1536), dim=-1)
    collapse_axis = F.normalize(torch.randn(1, 1536), dim=-1)
    coefficients = torch.linspace(-0.01, 0.01, baseline.shape[0])[:, None]
    candidate = F.normalize(common + coefficients * collapse_axis, dim=-1)
    baseline_stats = descriptor_statistics(baseline)
    candidate_stats = descriptor_statistics(candidate)
    payload = _audit_payload(candidate_stats, baseline_stats)
    checked = formal.validate_health_audit(payload)
    assert checked["status"] == "reject_more_collapsed_than_accepted"
    assert checked["query_authority_eligible"] is False
    assert checked["checks"]["centroid_squared_norm_not_more_collapsed"] is False
    assert checked["checks"]["sampled_pair_mean_not_more_collapsed"] is False
    assert checked["checks"]["centered_gram_effective_rank_not_materially_smaller"] is False
    with pytest.raises(ValueError, match="anti-collapse gate"):
        formal.validate_health_audit(payload, require_pass=True)


def test_health_contract_has_no_query_or_scene_adaptation() -> None:
    contract = formal.health_contract()
    assert contract["scene_parameters"] is False
    assert contract["threshold_origin"].startswith("global_method_level_fixed")
    assert contract["query_authority_requires_pass"] is True
    assert formal.health_access_audit()["benchmark_queries_opened"] is False
    assert formal.health_access_audit()["target_metrics_computed"] is False
