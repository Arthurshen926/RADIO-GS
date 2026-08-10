from __future__ import annotations

import pytest
import torch

from radio_gs.interfaces import factorized_native_contrast_v21_lerf_exact as raw_formal
from radio_gs.interfaces import factorized_native_contrast_v21_source_threshold_aligned_relevance as formal
from radio_gs.scripts import materialize_factorized_native_contrast_v21_source_threshold_aligned_relevance as script


def _record(name: str, digit: str = "1") -> dict[str, str]:
    return {"path": f"/tmp/{name}", "sha256": digit * 64}


def _raw_payload() -> dict[str, object]:
    relevance = torch.tensor([[0.48, 0.51], [0.50, 0.62]], dtype=torch.float32)
    payload: dict[str, object] = {
        "schema": raw_formal.QUERY_RELEVANCE_SCHEMA,
        "schema_version": raw_formal.SCHEMA_VERSION,
        "contract": raw_formal.query_contract(),
        "contract_sha256": raw_formal.QUERY_CONTRACT_SHA256,
        "scene_id": "figurines",
        "physical_space_id": "space",
        "producer": _record("raw.py"),
        "query_execution_authority": _record("raw-authority.json", "2"),
        "input_authority": {
            "source_result": _record("source.json"),
            "target_descriptor": _record("descriptor.pt", "2"),
            "health_v4_audit": _record("health.json", "3"),
            "health_v4_preregistration": raw_formal.HEALTH_V4_PREREGISTRATION,
            "query_preregistration": raw_formal.QUERY_PREREGISTRATION,
            "exact_query_manifest": _record("manifest.json", "4"),
            "positive_text_cache": _record("positive.pt", "5"),
            "all_query_text_cache": raw_formal.FROZEN_ALL_QUERY_CACHE,
            "canonical_negative_bank": raw_formal.FROZEN_CANONICAL_NEGATIVE_BANK,
        },
        "region_row_ids": ["r0", "r1"],
        "canonical_region_indices": torch.tensor([3, 9], dtype=torch.int64),
        "region_fingerprints": ["a" * 64, "b" * 64],
        "query_ids": ["chair", "waldo"],
        "region_absolute_relevance": relevance,
        "access_audit": raw_formal.query_access_audit(),
    }
    payload["channel_sha256"] = raw_formal.query_channel_sha256(payload)
    return payload


def _aligned_payload(raw: dict[str, object], threshold: float) -> dict[str, object]:
    margin, aligned = formal.boundary_align(
        raw["region_absolute_relevance"], threshold=threshold
    )
    inputs = {
        "source_threshold_envelope": _record("threshold.json", "6"),
        "raw_query_relevance": _record("raw.pt", "7"),
        "raw_query_execution_authority": raw["query_execution_authority"],
        "source_result": raw["input_authority"]["source_result"],
        "source_checkpoint": _record("checkpoint.pt", "8"),
        "target_descriptor": raw["input_authority"]["target_descriptor"],
        "health_v4_audit": raw["input_authority"]["health_v4_audit"],
        "exact_query_manifest": raw["input_authority"]["exact_query_manifest"],
        "positive_text_cache": raw["input_authority"]["positive_text_cache"],
        "all_query_text_cache": raw["input_authority"]["all_query_text_cache"],
        "canonical_negative_bank": raw["input_authority"]["canonical_negative_bank"],
    }
    payload: dict[str, object] = {
        "schema": formal.RELEVANCE_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "contract": formal.relevance_contract(),
        "contract_sha256": formal.RELEVANCE_CONTRACT_SHA256,
        "scene_id": raw["scene_id"],
        "physical_space_id": raw["physical_space_id"],
        "producer": _record("aligned.py"),
        "execution_authority": _record("aligned-authority.json", "9"),
        "input_authority": inputs,
        "source_global_margin_threshold": threshold,
        "raw_probability_boundary": float(
            torch.sigmoid(torch.tensor(10.0 * threshold, dtype=torch.float64))
        ),
        "region_row_ids": raw["region_row_ids"],
        "canonical_region_indices": raw["canonical_region_indices"].clone(),
        "region_fingerprints": raw["region_fingerprints"],
        "query_ids": raw["query_ids"],
        "recovered_margin": margin,
        "region_boundary_aligned_relevance": aligned,
        "coverage_audit": formal.query_coverage_audit(
            query_ids=raw["query_ids"],
            raw_probability=raw["region_absolute_relevance"],
            aligned_probability=aligned,
        ),
        "rank_invariance_audit": {
            "per_query_strict_order_preserved": True,
            "queries_checked": 2,
            "regions_checked": 2,
            "ranking_normalization": False,
        },
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    return payload


def test_boundary_alignment_recovers_margin_and_moves_frozen_threshold_to_half() -> None:
    threshold = -0.003271527588367462
    raw_boundary = torch.sigmoid(torch.tensor([[10.0 * threshold]])).float()
    margin, aligned = formal.boundary_align(raw_boundary, threshold=threshold)
    assert margin.item() == pytest.approx(threshold, abs=1e-7)
    assert aligned.item() == pytest.approx(0.5, abs=1e-7)


def test_alignment_strictly_preserves_per_query_order() -> None:
    raw = torch.tensor([[0.2, 0.8], [0.4, 0.3], [0.7, 0.6]], dtype=torch.float32)
    _, aligned = formal.boundary_align(raw, threshold=-0.003)
    assert torch.equal(torch.argsort(raw, dim=0), torch.argsort(aligned, dim=0))


def test_strict_schema_recomputes_formula_and_rejects_tamper() -> None:
    raw = _raw_payload()
    threshold = -0.003271527588367462
    source = {
        "status": "source_only_promoted",
        "global_threshold_authorized": True,
        "thresholds": {"train_selected_candidate": threshold},
    }
    payload = _aligned_payload(raw, threshold)
    checked = formal.validate_relevance(
        payload, raw_payload=raw, source_threshold_result=source
    )
    assert checked["coverage_audit"]["aligned_queries_with_positive"] == 2
    tampered = dict(payload)
    tampered["region_boundary_aligned_relevance"] = payload[
        "region_boundary_aligned_relevance"
    ].clone()
    tampered["region_boundary_aligned_relevance"][0, 0] += 0.01
    tampered["channel_sha256"] = formal.channel_sha256(tampered)
    with pytest.raises(ValueError, match="tensor differs"):
        formal.validate_relevance(
            tampered, raw_payload=raw, source_threshold_result=source
        )


def test_source_promotion_is_checked_before_raw_target_open(monkeypatch) -> None:
    events: list[str] = []

    def reject(*args, **kwargs):
        events.append("source")
        raise ValueError("source reject")

    def raw_open(*args, **kwargs):
        events.append("raw")
        raise AssertionError("raw target opened before source promotion")

    monkeypatch.setattr(formal, "load_promoted_source_threshold_envelope", reject)
    monkeypatch.setattr(script, "_load_raw_lineage", raw_open)
    with pytest.raises(ValueError, match="source reject"):
        script._validate_source_and_raw(
            threshold_record=_record("threshold.json"), raw_record=_record("raw.pt")
        )
    assert events == ["source"]


def test_contract_keeps_absolute_candidate_independent_and_metric_free() -> None:
    contract = formal.relevance_contract()
    assert contract["parameters"]["scene_parameters"] is False
    assert contract["parameters"]["query_parameters"] is False
    assert contract["candidate_independence"]["frozen_relative_path_opened"] is False
    assert contract["candidate_independence"]["mixed_with_frozen_relative"] is False
    assert contract["audit_only"]["ground_truth_or_metric"] is False
    assert formal.access_audit()["ground_truth_opened"] is False
    assert formal.access_audit()["target_metrics_computed"] is False
