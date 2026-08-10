from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.lerf_o0_anchored_conformal_residual import (
    O0AnchoredConformalResidualResult,
)
from radio_gs.scripts import build_lerf_o0_anchored_graph_residual_cache as builder


CALIBRATION = Path(
    "/root/RADIO-GS/paper/artifacts/"
    "source_only_graph_consumer_exact_calibration_result_fix3_20260807.json"
)


def _config() -> builder.DeploymentCalibration:
    raw = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    _, config = builder.validate_source_calibration(raw)
    return config


def _pair_features(count: int, *, reliability: float = 0.95) -> torch.Tensor:
    result = torch.zeros(count, 21, dtype=torch.float32)
    result[:, 17] = reliability
    result[:, 18] = reliability
    return result


def _evidence(
    *,
    rows: torch.Tensor,
    core: torch.Tensor,
    rank: torch.Tensor | None = None,
    feature: torch.Tensor | None = None,
    pairs: torch.Tensor | None = None,
    probabilities: torch.Tensor | None = None,
):
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
    pairs = (
        torch.tensor([[0], [2]], dtype=torch.long) if pairs is None else pairs
    )
    probabilities = (
        torch.tensor([0.95], dtype=torch.float32)
        if probabilities is None
        else probabilities
    )
    feature = _pair_features(pairs.shape[1]) if feature is None else feature
    return builder.build_region_evidence(
        o0_scores=scores,
        region_rows=rows,
        core_mask=core,
        primitive_valid=torch.ones(8, dtype=torch.bool),
        canonical_region_indices=torch.arange(4, dtype=torch.long),
        pair_indices=pairs,
        pair_probabilities=probabilities,
        pair_features=feature,
        pair_feature_median=torch.zeros(21),
        pair_feature_robust_scale=torch.ones(21),
        descriptor_reliability=torch.zeros(4),
        descriptor_active=torch.ones(4, dtype=torch.bool),
        descriptor_ood=torch.zeros(4, dtype=torch.bool),
        rank256_relevance=(
            torch.rand(4, 2, generator=torch.Generator().manual_seed(3))
            if rank is None
            else rank
        ),
        config=_config(),
    )


def test_latest_source_calibration_is_consumed_exactly() -> None:
    raw = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    payload, config = builder.validate_source_calibration(raw)
    assert payload["content_authority_sha256"] == raw["content_authority_sha256"]
    assert config.graph_method == "direct_O0_anchor_edge_residual"
    assert config.raw_edge_probability_minimum == 0.9
    assert config.maximum_selected_regions == 3
    assert len(config.null_step_thresholds) == 3
    assert all(value >= 0.0 for value in config.null_step_thresholds)


def test_exact_o0_readout_matches_frozen_vala_helper_bitwise() -> None:
    generator = torch.Generator().manual_seed(9)
    positive = torch.rand(9, 3, 2, generator=generator)
    negative = torch.rand(9, 3, 4, generator=generator)
    xyz = torch.arange(27, dtype=torch.float32).reshape(9, 3)
    valid = torch.tensor([True, True, True, True, True, True, True, True, False])
    result = builder.exact_o0_readout(
        positive_scores=positive,
        negative_scores=negative,
        xyz=xyz,
        valid=valid,
        chunk_size=4,
    )
    probability = builder.v2.frozen.canonical_negative_relevancy_query_scores(
        positive, negative, logit_scale=10.0
    )
    official = builder.v2.frozen.vala_multiscale_knn_peak_select_scores(
        probability,
        xyz,
        k=10,
        chunk_size=4,
        valid_mask=valid,
    )
    assert torch.equal(result.final_scores.view(torch.int32), official.scores.view(torch.int32))
    assert torch.equal(result.selected_scale_indices, official.selected_scale_indices)
    assert torch.equal(result.raw_smoothed_peaks, official.raw_smoothed_peaks)


def test_two_query_anchors_and_one_direct_edge_admit_candidate() -> None:
    rows = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    evidence = _evidence(rows=rows, core=torch.ones_like(rows, dtype=torch.bool))
    assert evidence.anchor_region[:, 0].tolist() == [True, True, False, False]
    assert evidence.anchor_region[:, 1].tolist() == [True, False, False, False]
    assert evidence.direct_anchor_support[2, 0] == 1
    assert evidence.candidate_region[:, 0].tolist() == [False, False, True, False]
    assert not bool(evidence.candidate_region[:, 1].any())
    assert evidence.query_gate.tolist() == [True, False]
    expected = torch.sigmoid(
        torch.logit(torch.tensor(0.95)) - _config().epsilon_logit
    )
    assert evidence.lower[2, 0] == pytest.approx(float(expected))
    assert evidence.diagnostics["null_activation"].tolist() == [0.0, 0.0]
    assert evidence.diagnostics["reliability"][0] == pytest.approx(0.95)


def test_token_permutation_does_not_change_anchor_or_candidate_evidence() -> None:
    rows = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    core = torch.ones_like(rows, dtype=torch.bool)
    original = _evidence(rows=rows, core=core)
    permutation = torch.tensor([1, 0])
    permuted = _evidence(rows=rows[:, permutation], core=core[:, permutation])
    assert torch.equal(original.anchor_region, permuted.anchor_region)
    assert torch.equal(original.candidate_region, permuted.candidate_region)
    assert torch.equal(original.lower, permuted.lower)
    assert torch.equal(original.query_gate, permuted.query_gate)


def test_rank256_top_tail_is_diagnostic_only() -> None:
    rows = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    core = torch.ones_like(rows, dtype=torch.bool)
    first = _evidence(rows=rows, core=core, rank=torch.zeros(4, 2))
    second = _evidence(rows=rows, core=core, rank=torch.eye(4, 2))
    assert torch.equal(first.lower, second.lower)
    assert torch.equal(first.query_gate, second.query_gate)
    assert torch.equal(first.candidate_region, second.candidate_region)


def test_ood_normalization_filters_only_the_bad_edge() -> None:
    rows = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    core = torch.ones_like(rows, dtype=torch.bool)
    feature = _pair_features(1)
    feature[0, 0] = _config().feature_ood_raw_limit * 2.0
    blocked = _evidence(rows=rows, core=core, feature=feature)
    assert not bool(blocked.candidate_region.any())
    assert blocked.query_gate.tolist() == [False, False]


def test_weak_edge_does_not_kill_query_with_an_independent_strong_edge() -> None:
    rows = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    core = torch.ones_like(rows, dtype=torch.bool)
    pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    feature = _pair_features(2)
    feature[1, 17:19] = 0.1
    evidence = _evidence(
        rows=rows,
        core=core,
        pairs=pairs,
        probabilities=torch.tensor([0.95, 0.96]),
        feature=feature,
    )
    assert evidence.candidate_region[:, 0].tolist() == [False, False, True, False]
    assert evidence.query_gate.tolist() == [True, False]


def test_exact_probability_fallback_preserves_signed_zero_and_endpoints() -> None:
    base = torch.tensor(
        [[-0.0, 1.0], [torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps]],
        dtype=torch.float32,
    )
    delta = torch.tensor([[0.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    logits = torch.logit(base.clamp(builder.O0_LOGIT_CLAMP, 1.0 - builder.O0_LOGIT_CLAMP))
    logits[1, 0] += 0.2
    result = O0AnchoredConformalResidualResult(
        fused_logits=logits,
        residual_logits=delta,
        query_gate=torch.tensor([True, False]),
        selected_region_rows=((0,), ()),
        selected_canonical_region_indices=((0,), ()),
        selected_lower_bounds=((0.9,), ()),
        selected_gains=((1.0,), ()),
        selected_marginal_primitives=((1,), ()),
    )
    fused, changed = builder.fuse_exact_o0_probabilities(base, result)
    assert changed.tolist() == [[False, False], [True, False]]
    unchanged = ~changed
    assert torch.equal(fused[unchanged].view(torch.int32), base[unchanged].view(torch.int32))
    assert torch.equal(fused[:, 1].view(torch.int32), base[:, 1].view(torch.int32))
    assert fused[1, 0] > base[1, 0]


def test_cli_exposes_no_target_quality_controls() -> None:
    destinations = {action.dest for action in builder.build_parser()._actions}
    assert destinations == {
        "help", "execution_authority", "expected_execution_authority_sha256"
    }
