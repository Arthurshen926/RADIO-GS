from __future__ import annotations

import torch

from radio_gs.querying.coverage_deficit_native_v3_support_fusion import (
    coverage_deficit_native_v3_support_fusion,
    readout_contract,
)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "primitive_relevance_by_scale": torch.tensor(
            [[[0.95]], [[0.20]], [[0.80]], [[0.40]]], dtype=torch.float32
        ),
        "selected_scale_indices": torch.tensor([0]),
        "primitive_valid": torch.ones(4, dtype=torch.bool),
        "region_rows": torch.tensor([[0, 1], [2, 3]]),
        "semantic_core_mask": torch.ones((2, 2), dtype=torch.bool),
        "region_anchor_positions": torch.zeros(2, dtype=torch.long),
        "region_scale_indices": torch.zeros(2, dtype=torch.long),
        "pair_indices": torch.tensor([[0], [1]], dtype=torch.long),
        "pair_probabilities": torch.tensor([0.925]),
        "pair_observation_evidence": torch.tensor([0.75]),
    }


def test_continuous_coverage_deficit_completion_is_monotone() -> None:
    result = coverage_deficit_native_v3_support_fusion(**_inputs())
    assert result.completion_strength[1, 0] > 0
    assert result.coverage_deficit[1, 0] > 0
    assert torch.equal(
        result.final_primitive_relevance[:2], result.primitive_unary[:2]
    )
    assert result.final_primitive_relevance[3, 0] > result.primitive_unary[3, 0]


def test_single_view_observation_is_exact_O1() -> None:
    inputs = _inputs()
    inputs["pair_observation_evidence"] = torch.tensor([0.5])
    result = coverage_deficit_native_v3_support_fusion(**inputs)
    assert torch.equal(result.final_primitive_relevance, result.primitive_unary)


def test_missing_anchor_evidence_is_exact_O1() -> None:
    inputs = _inputs()
    inputs["primitive_relevance_by_scale"][2, 0, 0] = 0.59
    result = coverage_deficit_native_v3_support_fusion(**inputs)
    assert torch.equal(result.final_primitive_relevance, result.primitive_unary)


def test_saturated_core_has_zero_residual() -> None:
    inputs = _inputs()
    inputs["primitive_relevance_by_scale"][0, 0, 0] = 1.0
    inputs["primitive_relevance_by_scale"][2:, 0, 0] = 1.0
    result = coverage_deficit_native_v3_support_fusion(**inputs)
    assert result.coverage_deficit[1, 0].item() <= 1e-12
    assert torch.allclose(
        result.final_primitive_relevance,
        result.primitive_unary,
        rtol=0.0,
        atol=1e-7,
    )


def test_contract_has_no_new_tunable_constant() -> None:
    contract = readout_contract()
    assert contract["invariants"]["new_tunable_constants"] is False
    assert contract["invariants"]["single_view_effective_path_is_exact_O1"]
