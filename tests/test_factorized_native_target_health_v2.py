from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_target_health_v2 as formal
from radio_gs.scripts.audit_factorized_native_target_descriptor_health_v2 import (
    descriptor_statistics,
)


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def _payload(candidate: dict, baseline: dict, *, descriptor_schema: str) -> dict:
    checks = formal.expected_checks(candidate, baseline)
    eligible = all(checks.values())
    return {
        "schema": formal.HEALTH_AUDIT_SCHEMA,
        "schema_version": formal.HEALTH_AUDIT_SCHEMA_VERSION,
        "contract": formal.health_contract(),
        "contract_sha256": formal.HEALTH_CONTRACT_SHA256,
        "status": "pass" if eligible else "reject_more_collapsed_than_accepted",
        "candidate_descriptor_schema": descriptor_schema,
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


@pytest.mark.parametrize("schema", formal.SUPPORTED_DESCRIPTOR_SCHEMAS)
def test_exact_schema_union_dispatches_only_registered_validators(
    monkeypatch, schema: str
) -> None:
    expected = {"schema": schema, "checked": True}
    monkeypatch.setattr(
        formal.v1_formal,
        "validate_target_descriptor_authority",
        lambda value: expected,
    )
    monkeypatch.setattr(
        formal.contrast_formal,
        "validate_target_descriptor_authority",
        lambda value: expected,
    )
    assert formal.validate_supported_target_descriptor({"schema": schema}) == expected
    with pytest.raises(ValueError, match="not supported"):
        formal.validate_supported_target_descriptor({"schema": "unknown"})


def test_health_v2_passes_identical_descriptor_for_contrast_schema() -> None:
    torch.manual_seed(73)
    values = F.normalize(torch.randn(96, 1536), dim=-1)
    statistics = descriptor_statistics(values)
    payload = _payload(
        dict(statistics),
        dict(statistics),
        descriptor_schema=formal.contrast_formal.TARGET_DESCRIPTOR_SCHEMA,
    )
    checked = formal.validate_health_audit(payload, require_pass=True)
    assert checked["status"] == "pass"
    assert checked["candidate_descriptor_schema"] in formal.SUPPORTED_DESCRIPTOR_SCHEMAS


def test_health_v2_contract_preserves_v1_and_has_no_scene_adaptation() -> None:
    contract = formal.health_contract()
    assert contract["health_v1_changed"] is False
    assert contract["scene_parameters"] is False
    assert contract["candidate"]["supported_descriptor_schemas"] == list(
        formal.SUPPORTED_DESCRIPTOR_SCHEMAS
    )
    assert contract["candidate"]["dispatch"].startswith("exact_schema_union")
