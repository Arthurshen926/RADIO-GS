from __future__ import annotations

import torch
import pytest

from radio_gs.interfaces import lerf_raw_unary_region_specificity as unary
from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache as fix4b
from radio_gs.scripts import build_lerf_o0_anchored_raw_dominant_positive_utility_cache_fix5 as fix5


def _config() -> fix4b.PositiveUtilityDeployment:
    return fix4b.PositiveUtilityDeployment(
        epsilon_logit=0.4,
        novel_mass_reference=256.0,
        minimum_reliability=0.7,
        maximum_feature_ood_score=0.5,
        minimum_anchor_agreement=0.6,
        minimum_stability=1.0,
        raw_edge_probability_minimum=0.9,
        maximum_selected_regions=3,
        anchor_quorum=2,
        o0_supermajority_fraction=0.75,
        o0_final_score_minimum=0.6,
        feature_ood_raw_limit=10.0,
        stability_required_fraction=1.0,
    )


def _specificity(dominant: torch.Tensor) -> unary.RawUnaryRegionSpecificity:
    regions, queries = dominant.shape
    mean = torch.where(dominant, torch.ones_like(dominant, dtype=torch.float32), torch.zeros_like(dominant, dtype=torch.float32))
    return unary.RawUnaryRegionSpecificity(
        mean_raw_probability=mean,
        dominant_query_mask=dominant,
        primitive_top1_fraction=mean,
        valid_core_counts=torch.ones(regions, dtype=torch.long),
    )


def _base() -> fix4b.RegionEvidence:
    anchor = torch.tensor([[True], [True], [False], [False]])
    candidate = torch.tensor([[False], [False], [True], [True]])
    return fix4b.RegionEvidence(
        lower=torch.tensor([[0.0], [0.0], [0.9], [0.9]], dtype=torch.float32),
        eligible=torch.ones(4, dtype=torch.bool),
        query_gate=torch.tensor([True]),
        anchor_region=anchor,
        direct_anchor_support=torch.tensor([[0], [0], [1], [1]]),
        candidate_region=candidate,
        diagnostics={"stability": torch.tensor([1.0])},
        rank256_top_tail=torch.zeros((4, 1), dtype=torch.bool),
    )


def _features() -> torch.Tensor:
    values = torch.zeros((2, 21), dtype=torch.float32)
    values[:, 17:19] = 1.0
    return values


def test_fix5_filters_anchor_before_support_and_candidate_after_support() -> None:
    # Anchor region 0 is semantically invalid; candidate region 2 is connected
    # only to it and must not survive.  Candidate 3 is connected to valid anchor
    # 1 and survives because two specific anchors still satisfy the global quorum.
    result = fix5.apply_raw_dominant_gate_to_evidence(
        _base(),
        specificity=_specificity(torch.tensor([[False], [True], [True], [True]])),
        pair_indices=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        pair_probabilities=torch.tensor([0.99, 0.99]),
        pair_features=_features(),
        pair_feature_median=torch.zeros(21),
        pair_feature_robust_scale=torch.ones(21),
        config=_config(),
    )
    # Only one specific anchor remains, so the original quorum closes the query.
    assert result.anchor_region[:, 0].tolist() == [False, True, False, False]
    assert not bool(result.candidate_region.any())
    assert not bool(result.query_gate[0])
    assert torch.count_nonzero(result.lower) == 0


def test_fix5_keeps_only_raw_dominant_candidate_with_reliable_edge() -> None:
    result = fix5.apply_raw_dominant_gate_to_evidence(
        _base(),
        specificity=_specificity(torch.tensor([[True], [True], [False], [True]])),
        pair_indices=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        pair_probabilities=torch.tensor([0.99, 0.99]),
        pair_features=_features(),
        pair_feature_median=torch.zeros(21),
        pair_feature_robust_scale=torch.ones(21),
        config=_config(),
    )
    assert result.anchor_region[:, 0].tolist() == [True, True, False, False]
    assert result.candidate_region[:, 0].tolist() == [False, False, False, True]
    assert result.direct_anchor_support[:, 0].tolist() == [0, 0, 1, 1]
    assert result.lower[:, 0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.9])
    assert bool(result.query_gate[0])


def test_fix5_edge_reliability_failure_removes_direct_support() -> None:
    features = _features()
    features[1, 17] = 0.1
    result = fix5.apply_raw_dominant_gate_to_evidence(
        _base(),
        specificity=_specificity(torch.ones((4, 1), dtype=torch.bool)),
        pair_indices=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        pair_probabilities=torch.tensor([0.99, 0.99]),
        pair_features=features,
        pair_feature_median=torch.zeros(21),
        pair_feature_robust_scale=torch.ones(21),
        config=_config(),
    )
    assert result.candidate_region[:, 0].tolist() == [False, False, True, False]
    assert result.lower[:, 0].tolist() == pytest.approx([0.0, 0.0, 0.9, 0.0])


def test_all_invalid_region_core_is_semantically_inactive_exact_fallback() -> None:
    specificity = fix5.raw_specificity_with_invalid_region_fallback(
        raw_query_probabilities=torch.tensor(
            [[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32
        ),
        region_rows=torch.tensor([[0], [1]], dtype=torch.long),
        core_mask=torch.ones((2, 1), dtype=torch.bool),
        primitive_valid_mask=torch.tensor([True, False]),
    )
    assert specificity.valid_core_counts.tolist() == [1, 0]
    assert specificity.dominant_query_mask[0].tolist() == [True, False]
    assert specificity.dominant_query_mask[1].tolist() == [False, False]
    assert specificity.mean_raw_probability[1].tolist() == [0.0, 0.0]
    assert specificity.primitive_top1_fraction[1].tolist() == [0.0, 0.0]
