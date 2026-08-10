from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts import build_lerf_o0_anchored_graph_residual_cache as fix3_builder
from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache as builder


FIX3_RESULT = Path(
    "/root/RADIO-GS/paper/artifacts/"
    "source_only_graph_consumer_exact_calibration_result_fix3_20260807.json"
)
FIX4_RESULT = Path(
    "/root/RADIO-GS/paper/artifacts/"
    "source_only_graph_expected_utility_audit_result_fix4_20260807.json"
)
FIX4B_RESULT = Path(
    "/root/RADIO-GS/paper/artifacts/"
    "source_only_graph_positive_utility_contract_result_fix4b_20260807.json"
)


def _config() -> builder.PositiveUtilityDeployment:
    raw = json.loads(FIX4B_RESULT.read_text(encoding="utf-8"))
    _, config = builder.validate_source_calibration(raw)
    return config


def _pair_features(count: int, *, reliability: float = 0.95) -> torch.Tensor:
    result = torch.zeros(count, 21, dtype=torch.float32)
    result[:, 17] = reliability
    result[:, 18] = reliability
    return result


def _evidence(*, rank: torch.Tensor | None = None):
    scores = torch.tensor(
        [
            [0.90, 0.90],
            [0.80, 0.80],
            [0.95, 0.20],
            [0.70, 0.20],
            [0.10, 0.10],
            [0.20, 0.20],
            [0.10, 0.10],
            [0.20, 0.20],
        ],
        dtype=torch.float32,
    )
    rows = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    core = torch.ones_like(rows, dtype=torch.bool)
    pairs = torch.tensor([[0], [2]], dtype=torch.long)
    probabilities = torch.tensor([0.95], dtype=torch.float32)
    feature = _pair_features(1)
    kwargs = {
        "o0_scores": scores,
        "region_rows": rows,
        "core_mask": core,
        "primitive_valid": torch.ones(8, dtype=torch.bool),
        "canonical_region_indices": torch.arange(4, dtype=torch.long),
        "pair_indices": pairs,
        "pair_probabilities": probabilities,
        "pair_features": feature,
        "pair_feature_median": torch.zeros(21),
        "pair_feature_robust_scale": torch.ones(21),
        "descriptor_reliability": torch.zeros(4),
        "descriptor_active": torch.ones(4, dtype=torch.bool),
        "descriptor_ood": torch.zeros(4, dtype=torch.bool),
        "rank256_relevance": (
            torch.rand(4, 2, generator=torch.Generator().manual_seed(3))
            if rank is None
            else rank
        ),
    }
    return kwargs, builder.build_region_evidence(**kwargs, config=_config())


def test_only_clean_promoted_fix4b_source_schema_is_accepted() -> None:
    raw = json.loads(FIX4B_RESULT.read_text(encoding="utf-8"))
    payload, config = builder.validate_source_calibration(raw)
    assert payload["schema"] == builder.fix4b_calibration.RESULT_SCHEMA
    assert payload["status"] == "source_only_positive_utility_fix4b_promoted_target_unopened"
    assert config.maximum_selected_regions == 3
    assert config.residual_config().minimum_stability == 1.0

    with pytest.raises(ValueError, match="FIX4B source result fields differ"):
        builder.validate_source_calibration(
            json.loads(FIX4_RESULT.read_text(encoding="utf-8"))
        )
    with pytest.raises(ValueError, match="FIX4B source result fields differ"):
        builder.validate_source_calibration(
            json.loads(FIX3_RESULT.read_text(encoding="utf-8"))
        )


def test_deployment_rejects_null_gate_or_sequential_threshold_fields() -> None:
    raw = json.loads(FIX4B_RESULT.read_text(encoding="utf-8"))["deployment_config"]
    null_gate = copy.deepcopy(raw)
    null_gate["query_gate"]["null_activation"] = 0.0
    with pytest.raises(ValueError, match="nested fields differ"):
        builder._validate_deployment_config(null_gate)

    sequential = copy.deepcopy(raw)
    sequential["selection"]["null_step_thresholds"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="nested fields differ"):
        builder._validate_deployment_config(sequential)


def test_positive_utility_evidence_matches_fix3_non_null_evidence_bitwise() -> None:
    kwargs, clean = _evidence()
    fix3_raw = json.loads(FIX3_RESULT.read_text(encoding="utf-8"))
    _, fix3_config = fix3_builder.validate_source_calibration(fix3_raw)
    old = fix3_builder.build_region_evidence(**kwargs, config=fix3_config)
    assert torch.equal(clean.anchor_region, old.anchor_region)
    assert torch.equal(clean.direct_anchor_support, old.direct_anchor_support)
    assert torch.equal(clean.candidate_region, old.candidate_region)
    assert torch.equal(clean.lower.view(torch.int32), old.lower.view(torch.int32))
    assert torch.equal(clean.eligible, old.eligible)
    assert torch.equal(clean.query_gate, old.query_gate)
    assert "null_activation" not in clean.diagnostics
    assert set(clean.diagnostics) == set(old.diagnostics) - {"null_activation"}


def test_rank256_relevance_remains_diagnostic_only() -> None:
    _, first = _evidence(rank=torch.zeros(4, 2))
    _, second = _evidence(rank=torch.eye(4, 2))
    assert torch.equal(first.lower, second.lower)
    assert torch.equal(first.query_gate, second.query_gate)
    assert torch.equal(first.candidate_region, second.candidate_region)


def test_fusion_preserves_signed_zero_endpoints_and_failed_gate_bits() -> None:
    base = torch.tensor(
        [
            [-0.0, 1.0],
            [torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps],
        ],
        dtype=torch.float32,
    )
    logits = torch.logit(base.clamp(builder.O0_LOGIT_CLAMP, 1.0 - builder.O0_LOGIT_CLAMP))
    delta = torch.tensor([[0.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    logits[1, 0] += 0.2
    result = builder.positive.O0AnchoredPositiveUtilityResidualResult(
        fused_logits=logits,
        residual_logits=delta,
        query_gate=torch.tensor([True, False]),
        selected_region_rows=((0,), ()),
        selected_canonical_region_indices=((0,), ()),
        selected_lower_scores=((0.9,), ()),
        selected_gains=((1.0,), ()),
        selected_marginal_primitives=((1,), ()),
    )
    fused, changed = builder.fuse_exact_o0_probabilities(base, result)
    assert changed.tolist() == [[False, False], [True, False]]
    assert torch.equal(fused[~changed].view(torch.int32), base[~changed].view(torch.int32))
    assert torch.equal(fused[:, 1].view(torch.int32), base[:, 1].view(torch.int32))
    assert fused[1, 0] > base[1, 0]


def test_cli_has_no_target_quality_or_null_policy_controls() -> None:
    destinations = {action.dest for action in builder.build_parser()._actions}
    assert destinations == {
        "help",
        "execution_authority",
        "expected_execution_authority_sha256",
    }

