from __future__ import annotations

import pytest
import torch

from radio_gs.interfaces.lerf_o0_anchored_conformal_residual import (
    SourceFixedResidualConfig,
    o0_anchored_conformal_residual,
    source_fixed_query_gate,
)


def _config(**overrides) -> SourceFixedResidualConfig:
    values = {
        "epsilon_logit": 0.4,
        "novel_mass_reference": 1.0,
        "null_step_thresholds": (0.0, 0.9, 0.9),
        "minimum_reliability": 0.7,
        "maximum_feature_ood_score": 0.2,
        "minimum_anchor_agreement": 0.6,
        "maximum_null_activation": 0.1,
        "minimum_stability": 0.8,
    }
    values.update(overrides)
    return SourceFixedResidualConfig(**values)


def _run(
    *,
    o0: torch.Tensor,
    lower: torch.Tensor,
    rows: torch.Tensor,
    gate: torch.Tensor,
    config: SourceFixedResidualConfig | None = None,
    canonical: torch.Tensor | None = None,
    eligible: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
):
    region_count = int(lower.shape[0])
    return o0_anchored_conformal_residual(
        o0_logits=o0,
        region_confidence_lower=lower,
        region_rows=rows,
        core_mask=rows >= 0,
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


def test_source_fixed_gate_is_one_conjunction_for_every_query() -> None:
    gate = source_fixed_query_gate(
        reliability=torch.tensor([0.9, 0.6, 0.9]),
        feature_ood_score=torch.tensor([0.1, 0.1, 0.3]),
        anchor_agreement=torch.tensor([0.8, 0.8, 0.8]),
        null_activation=torch.tensor([0.05, 0.05, 0.05]),
        stability=torch.tensor([0.9, 0.9, 0.9]),
        config=_config(),
    )
    assert torch.equal(gate, torch.tensor([True, False, False]))


def test_failed_query_gate_is_bitwise_o0_and_selects_nothing() -> None:
    o0 = torch.tensor([[-0.0, -1.0], [2.0, 3.0]], dtype=torch.float32)
    result = _run(
        o0=o0,
        lower=torch.tensor([[0.99, 0.99]], dtype=torch.float32),
        rows=torch.tensor([[0, 1]], dtype=torch.long),
        gate=torch.tensor([False, False]),
    )
    assert torch.equal(result.fused_logits.view(torch.int32), o0.view(torch.int32))
    assert torch.count_nonzero(result.residual_logits) == 0
    assert result.selected_region_rows == ((), ())


def test_residual_is_positive_bounded_and_union_outside_is_bitwise_o0() -> None:
    o0 = torch.tensor([[0.1], [-0.0], [0.3], [0.4]], dtype=torch.float32)
    result = _run(
        o0=o0,
        lower=torch.tensor([[0.75], [0.95]], dtype=torch.float32),
        rows=torch.tensor([[0, 1], [2, -1]], dtype=torch.long),
        gate=torch.tensor([True]),
        config=_config(null_step_thresholds=(0.0,), epsilon_logit=0.4),
    )
    assert result.selected_region_rows == ((0,),)
    assert torch.all(result.residual_logits >= 0.0)
    assert float(result.residual_logits.max()) <= 0.4
    assert torch.equal(
        result.fused_logits[2:].view(torch.int32), o0[2:].view(torch.int32)
    )
    assert torch.allclose(result.residual_logits[:2, 0], torch.tensor([0.2, 0.2]))


def test_source_null_sequential_threshold_stops_late_fragment() -> None:
    result = _run(
        o0=torch.zeros((3, 1), dtype=torch.float32),
        lower=torch.tensor([[0.90], [0.90]], dtype=torch.float32),
        rows=torch.tensor([[0, 1], [2, -1]], dtype=torch.long),
        gate=torch.tensor([True]),
        config=_config(null_step_thresholds=(0.0, 0.9)),
    )
    assert result.selected_region_rows == ((0,),)
    assert result.selected_marginal_primitives == ((2,),)
    assert result.selected_gains[0][0] == pytest.approx(1.6)
    assert result.residual_logits[2, 0] == 0.0


def test_exact_gain_tie_uses_lower_canonical_region_index() -> None:
    result = _run(
        o0=torch.zeros((2, 1), dtype=torch.float32),
        lower=torch.tensor([[0.80], [0.80]], dtype=torch.float32),
        rows=torch.tensor([[0], [1]], dtype=torch.long),
        gate=torch.tensor([True]),
        canonical=torch.tensor([9, 2], dtype=torch.long),
        config=_config(null_step_thresholds=(0.0,)),
    )
    assert result.selected_region_rows == ((1,),)
    assert result.selected_canonical_region_indices == ((2,),)


def test_disconnected_region_is_not_hard_rejected_when_source_gain_passes() -> None:
    result = _run(
        o0=torch.zeros((4, 1), dtype=torch.float32),
        lower=torch.tensor([[0.90], [0.85]], dtype=torch.float32),
        rows=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        gate=torch.tensor([True]),
        config=_config(null_step_thresholds=(0.0, 0.0)),
    )
    assert result.selected_region_rows == ((0, 1),)
    assert torch.count_nonzero(result.residual_logits) == 4


def test_invalid_or_ineligible_region_cannot_modify_o0() -> None:
    result = _run(
        o0=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        lower=torch.tensor([[0.99], [0.99]], dtype=torch.float32),
        rows=torch.tensor([[0], [1]], dtype=torch.long),
        gate=torch.tensor([True]),
        valid=torch.tensor([False, True]),
        eligible=torch.tensor([True, False]),
    )
    assert torch.equal(result.fused_logits, torch.tensor([[1.0], [2.0]]))
    assert result.selected_region_rows == ((),)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epsilon_logit": -0.1}, "epsilon_logit"),
        ({"novel_mass_reference": 0.0}, "novel_mass_reference"),
        ({"null_step_thresholds": ()}, "null step"),
        ({"minimum_reliability": 1.1}, "lie in"),
    ],
)
def test_source_fixed_config_fails_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_duplicate_region_core_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _run(
            o0=torch.zeros((2, 1), dtype=torch.float32),
            lower=torch.tensor([[0.9]], dtype=torch.float32),
            rows=torch.tensor([[0, 0]], dtype=torch.long),
            gate=torch.tensor([True]),
        )
