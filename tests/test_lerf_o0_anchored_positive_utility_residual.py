from __future__ import annotations

from dataclasses import fields
import inspect
import json

import pytest
import torch

from radio_gs.interfaces import lerf_o0_anchored_positive_utility_residual as positive


def _config(**overrides) -> positive.SourceFixedPositiveUtilityConfig:
    values = {
        "epsilon_logit": 0.4,
        "novel_mass_reference": 1.0,
        "minimum_reliability": 0.7,
        "maximum_feature_ood_score": 0.2,
        "minimum_anchor_agreement": 0.6,
        "minimum_stability": 0.8,
    }
    values.update(overrides)
    return positive.SourceFixedPositiveUtilityConfig(**values)


def _run(
    *,
    o0: torch.Tensor,
    lower: torch.Tensor,
    rows: torch.Tensor,
    gate: torch.Tensor,
    config: positive.SourceFixedPositiveUtilityConfig | None = None,
    canonical: torch.Tensor | None = None,
    eligible: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    core: torch.Tensor | None = None,
) -> positive.O0AnchoredPositiveUtilityResidualResult:
    region_count = int(lower.shape[0])
    return positive.o0_anchored_positive_utility_residual(
        o0_logits=o0,
        region_confidence_lower=lower,
        region_rows=rows,
        core_mask=rows >= 0 if core is None else core,
        primitive_valid_mask=(
            torch.ones(o0.shape[0], dtype=torch.bool) if valid is None else valid
        ),
        region_eligible_mask=(
            torch.ones(region_count, dtype=torch.bool)
            if eligible is None
            else eligible
        ),
        canonical_region_indices=(
            torch.arange(region_count, dtype=torch.long)
            if canonical is None
            else canonical
        ),
        query_gate=gate,
        config=_config() if config is None else config,
    )


def test_contract_is_honest_and_internal_policy_is_exactly_three_zeros() -> None:
    contract = positive.residual_contract()
    serialized = json.dumps(contract, sort_keys=True).lower()
    assert contract["internal_gain_thresholds"] == [0.0, 0.0, 0.0]
    assert contract["maximum_regions"] == 3
    assert positive.INTERNAL_GAIN_THRESHOLDS == (0.0, 0.0, 0.0)
    assert positive.MAXIMUM_REGIONS == 3
    assert all(
        forbidden not in serialized
        for forbidden in ("conformal", "fwer", "posterior", "null_activation")
    )
    assert {field.name for field in fields(positive.SourceFixedPositiveUtilityConfig)} == {
        "epsilon_logit",
        "novel_mass_reference",
        "minimum_reliability",
        "maximum_feature_ood_score",
        "minimum_anchor_agreement",
        "minimum_stability",
    }


def test_query_gate_has_only_four_diagnostics_and_is_one_conjunction() -> None:
    assert set(inspect.signature(positive.source_fixed_query_gate).parameters) == {
        "reliability",
        "feature_ood_score",
        "anchor_agreement",
        "stability",
        "config",
    }
    gate = positive.source_fixed_query_gate(
        reliability=torch.tensor([0.9, 0.6, 0.9, 0.9]),
        feature_ood_score=torch.tensor([0.1, 0.1, 0.3, 0.1]),
        anchor_agreement=torch.tensor([0.8, 0.8, 0.8, 0.5]),
        stability=torch.tensor([0.9, 0.9, 0.9, 0.9]),
        config=_config(),
    )
    assert torch.equal(gate, torch.tensor([True, False, False, False]))


def test_failed_gate_and_no_positive_utility_are_bitwise_o0() -> None:
    o0 = torch.tensor([[-0.0, 1.0], [2.0, -3.0]], dtype=torch.float32)
    result = _run(
        o0=o0,
        lower=torch.tensor([[0.99, 0.5]], dtype=torch.float32),
        rows=torch.tensor([[0, 1]], dtype=torch.long),
        gate=torch.tensor([False, True]),
    )
    assert torch.equal(result.fused_logits.view(torch.int32), o0.view(torch.int32))
    assert torch.count_nonzero(result.residual_logits) == 0
    assert result.selected_region_rows == ((), ())


def test_selection_is_strictly_positive_and_stops_at_three_regions() -> None:
    result = _run(
        o0=torch.zeros((5, 1), dtype=torch.float32),
        lower=torch.tensor(
            [[0.8], [0.8], [0.8], [0.8], [0.5]], dtype=torch.float32
        ),
        rows=torch.arange(5, dtype=torch.long).view(5, 1),
        gate=torch.tensor([True]),
        canonical=torch.tensor([9, 2, 7, 1, 0], dtype=torch.long),
    )
    assert result.selected_region_rows == ((3, 1, 2),)
    assert result.selected_canonical_region_indices == ((1, 2, 7),)
    assert result.selected_lower_scores[0] == pytest.approx((0.8, 0.8, 0.8))
    assert len(result.selected_region_rows[0]) == positive.MAXIMUM_REGIONS
    assert result.residual_logits[4, 0] == 0.0


def test_lower_score_just_above_half_has_positive_gain() -> None:
    result = _run(
        o0=torch.zeros((2, 1), dtype=torch.float32),
        lower=torch.tensor([[0.5], [0.5001]], dtype=torch.float32),
        rows=torch.tensor([[0], [1]], dtype=torch.long),
        gate=torch.tensor([True]),
    )
    assert result.selected_region_rows == ((1,),)
    assert result.selected_gains[0][0] > 0.0
    assert result.residual_logits[0, 0] == 0.0
    assert result.residual_logits[1, 0] > 0.0


def test_overlap_uses_bounded_pointwise_max_not_accumulation() -> None:
    result = _run(
        o0=torch.zeros((4, 1), dtype=torch.float32),
        lower=torch.tensor([[0.75], [0.9]], dtype=torch.float32),
        rows=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        gate=torch.tensor([True]),
    )
    assert result.selected_region_rows == ((1, 0),)
    assert result.residual_logits[:, 0].tolist() == pytest.approx(
        [0.2, 0.32, 0.32, 0.0]
    )
    assert float(result.residual_logits.max()) <= 0.4


def test_invalid_and_region_union_outside_primitives_are_bitwise_o0() -> None:
    o0 = torch.tensor([[0.1], [-0.0], [0.3], [0.4]], dtype=torch.float32)
    result = _run(
        o0=o0,
        lower=torch.tensor([[0.9]], dtype=torch.float32),
        rows=torch.tensor([[0, 1, 2]], dtype=torch.long),
        gate=torch.tensor([True]),
        valid=torch.tensor([True, False, True, True]),
    )
    assert result.selected_region_rows == ((0,),)
    unchanged = torch.tensor([False, True, False, True])
    assert torch.equal(
        result.fused_logits[unchanged].view(torch.int32),
        o0[unchanged].view(torch.int32),
    )
    assert torch.equal(
        result.residual_logits[unchanged],
        torch.zeros_like(result.residual_logits[unchanged]),
    )


def test_ineligible_region_and_duplicate_core_fail_closed() -> None:
    o0 = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    result = _run(
        o0=o0,
        lower=torch.tensor([[0.99]], dtype=torch.float32),
        rows=torch.tensor([[0]], dtype=torch.long),
        gate=torch.tensor([True]),
        eligible=torch.tensor([False]),
    )
    assert torch.equal(result.fused_logits, o0)
    assert result.selected_region_rows == ((),)
    with pytest.raises(ValueError, match="duplicate"):
        _run(
            o0=o0,
            lower=torch.tensor([[0.9]], dtype=torch.float32),
            rows=torch.tensor([[0, 0]], dtype=torch.long),
            gate=torch.tensor([True]),
        )


def test_token_permutation_is_exactly_invariant() -> None:
    o0 = torch.zeros((4, 1), dtype=torch.float32)
    lower = torch.tensor([[0.9], [0.8]], dtype=torch.float32)
    rows = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    original = _run(o0=o0, lower=lower, rows=rows, gate=torch.tensor([True]))
    permuted = _run(
        o0=o0,
        lower=lower,
        rows=rows[:, torch.tensor([1, 0])],
        gate=torch.tensor([True]),
    )
    assert original.selected_region_rows == permuted.selected_region_rows
    assert torch.equal(original.residual_logits, permuted.residual_logits)
    assert torch.equal(original.fused_logits, permuted.fused_logits)


def test_nonzero_internal_threshold_policy_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(positive, "INTERNAL_GAIN_THRESHOLDS", (0.0, 0.1, 0.0))
    with pytest.raises(RuntimeError, match="exactly three zeros"):
        _run(
            o0=torch.zeros((1, 1), dtype=torch.float32),
            lower=torch.tensor([[0.9]], dtype=torch.float32),
            rows=torch.tensor([[0]], dtype=torch.long),
            gate=torch.tensor([True]),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epsilon_logit": -0.1}, "epsilon_logit"),
        ({"novel_mass_reference": 0.0}, "novel_mass_reference"),
        ({"minimum_reliability": 1.1}, "lie in"),
    ],
)
def test_source_fixed_config_fails_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)
