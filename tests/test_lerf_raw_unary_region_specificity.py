from __future__ import annotations

import json

import pytest
import torch

from radio_gs.interfaces import lerf_raw_unary_region_specificity as unary


def _run(
    raw: torch.Tensor,
    rows: torch.Tensor,
    *,
    core: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> unary.RawUnaryRegionSpecificity:
    return unary.raw_unary_region_specificity(
        raw_query_probabilities=raw,
        region_rows=rows,
        core_mask=rows >= 0 if core is None else core,
        primitive_valid_mask=(
            torch.ones(raw.shape[0], dtype=torch.bool) if valid is None else valid
        ),
    )


def test_contract_requires_raw_shared_scale_and_keeps_fraction_diagnostic_only() -> None:
    contract = unary.specificity_contract()
    serialized = json.dumps(contract, sort_keys=True).lower()
    assert "same_negative_bank_and_logit_scale" in serialized
    assert "minmax" in contract["forbidden_input"]
    assert contract["primitive_top1_fraction"] == "diagnostic_only_not_a_gate"
    assert contract["primitive_majority_threshold"] is None
    assert contract["graph_order"] == "apply_anchor_gate_before_direct_support_propagation"
    assert contract["query_conditioned_parameters"] is False
    assert contract["scene_conditioned_parameters"] is False
    assert contract["target_metrics_used"] is False


def test_region_mean_dominance_and_primitive_fraction_are_kept_separate() -> None:
    raw = torch.tensor(
        [[0.9, 0.1], [0.2, 0.8], [0.2, 0.8]], dtype=torch.float32
    )
    result = _run(raw, torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert result.mean_raw_probability[0].tolist() == pytest.approx(
        [1.3 / 3.0, 1.7 / 3.0]
    )
    assert result.dominant_query_mask[0].tolist() == [False, True]
    assert result.primitive_top1_fraction[0].tolist() == pytest.approx([1 / 3, 2 / 3])


def test_exact_query_ties_are_retained_without_query_index_tie_break() -> None:
    raw = torch.tensor([[0.7, 0.7], [0.3, 0.3]], dtype=torch.float32)
    result = _run(raw, torch.tensor([[0, 1]], dtype=torch.long))
    assert result.dominant_query_mask[0].tolist() == [True, True]
    assert result.primitive_top1_fraction[0].tolist() == [1.0, 1.0]


def test_query_permutation_is_exactly_equivariant() -> None:
    raw = torch.tensor(
        [[0.9, 0.1, 0.2], [0.3, 0.8, 0.4], [0.2, 0.7, 0.6]],
        dtype=torch.float32,
    )
    rows = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    original = _run(raw, rows)
    order = torch.tensor([2, 0, 1])
    permuted = _run(raw[:, order], rows)
    assert torch.equal(
        permuted.mean_raw_probability, original.mean_raw_probability[:, order]
    )
    assert torch.equal(
        permuted.dominant_query_mask, original.dominant_query_mask[:, order]
    )
    assert torch.equal(
        permuted.primitive_top1_fraction,
        original.primitive_top1_fraction[:, order],
    )


def test_token_permutation_and_invalid_padding_are_exactly_invariant() -> None:
    raw = torch.tensor(
        [[0.9, 0.1], [0.2, 0.8], [1.0, 0.0]], dtype=torch.float32
    )
    rows = torch.tensor([[0, 1, 2]], dtype=torch.long)
    valid = torch.tensor([True, True, False])
    original = _run(raw, rows, valid=valid)
    permuted = _run(raw, rows[:, torch.tensor([2, 1, 0])], valid=valid)
    assert torch.equal(original.mean_raw_probability, permuted.mean_raw_probability)
    assert torch.equal(original.dominant_query_mask, permuted.dominant_query_mask)
    assert torch.equal(original.primitive_top1_fraction, permuted.primitive_top1_fraction)
    assert original.valid_core_counts.tolist() == [2]


def test_region_without_valid_core_fails_closed() -> None:
    with pytest.raises(ValueError, match="valid core"):
        _run(
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            torch.tensor([[0]], dtype=torch.long),
            valid=torch.tensor([False]),
        )


def test_symmetric_gate_filters_anchor_before_support_propagation() -> None:
    # Region 0 is an O0 anchor for q0 but raw-dominant for q1, so its edge must
    # not make region 2 a q0 candidate.  Regions 1 and 3 are the two valid q0
    # anchors; only region 4 receives direct support from one of them.
    gate = unary.symmetric_raw_dominant_graph_gate(
        base_anchor_region=torch.tensor(
            [
                [True, False],
                [True, False],
                [False, False],
                [True, False],
                [False, False],
            ]
        ),
        dominant_query_mask=torch.tensor(
            [
                [False, True],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            ]
        ),
        pair_indices=torch.tensor([[0, 1], [2, 4]], dtype=torch.long),
        edge_eligible_mask=torch.tensor([True, True]),
        region_eligible_mask=torch.ones(5, dtype=torch.bool),
        anchor_quorum=2,
    )
    assert gate.specific_anchor_region[:, 0].tolist() == [False, True, False, True, False]
    assert gate.direct_specific_anchor_support[:, 0].tolist() == [0, 0, 0, 0, 1]
    assert gate.specific_candidate_region[:, 0].tolist() == [False, False, False, False, True]


def test_symmetric_gate_requires_candidate_dominance_and_quorum() -> None:
    common = {
        "base_anchor_region": torch.tensor([[True], [True], [False]]),
        "dominant_query_mask": torch.tensor([[True], [True], [False]]),
        "pair_indices": torch.tensor([[0], [2]], dtype=torch.long),
        "edge_eligible_mask": torch.tensor([True]),
        "region_eligible_mask": torch.ones(3, dtype=torch.bool),
    }
    gate = unary.symmetric_raw_dominant_graph_gate(**common, anchor_quorum=2)
    assert not bool(gate.specific_candidate_region.any())
    insufficient = unary.symmetric_raw_dominant_graph_gate(**common, anchor_quorum=3)
    assert not bool(insufficient.specific_candidate_region.any())
