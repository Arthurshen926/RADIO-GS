from __future__ import annotations

import torch

from radio_gs.querying.corroborated_scale_aware_native_v3_support_fusion import (
    SEMANTIC_BOUNDARY,
    corroborated_scale_aware_native_v3_support_fusion,
    readout_contract,
)


def _inputs(*, target_witness: float) -> dict[str, torch.Tensor]:
    return {
        "primitive_relevance_by_scale": torch.tensor(
            [
                [[0.95]],
                [[0.20]],
                [[target_witness]],
                [[0.45]],
                [[0.10]],
                [[0.20]],
            ],
            dtype=torch.float32,
        ),
        "selected_scale_indices": torch.tensor([0]),
        "primitive_valid": torch.ones(6, dtype=torch.bool),
        "region_rows": torch.tensor([[0, 1], [2, 3], [4, 5]]),
        "semantic_core_mask": torch.ones((3, 2), dtype=torch.bool),
        "region_anchor_positions": torch.zeros(3, dtype=torch.long),
        "region_scale_indices": torch.zeros(3, dtype=torch.long),
        "pair_indices": torch.tensor([[0], [1]], dtype=torch.long),
        "pair_probabilities": torch.tensor([0.925]),
    }


def test_uncorroborated_relation_target_is_exact_unary() -> None:
    result = corroborated_scale_aware_native_v3_support_fusion(
        **_inputs(target_witness=SEMANTIC_BOUNDARY - 0.01)
    )
    assert result.query_gate.tolist() == [True]
    assert result.relation_selected_region_masks[:, 0].tolist() == [True, True, False]
    assert not bool(result.corroborated_relation_region_masks.any())
    assert torch.equal(result.final_primitive_relevance, result.primitive_unary)


def test_independent_target_witness_allows_continuous_completion() -> None:
    result = corroborated_scale_aware_native_v3_support_fusion(
        **_inputs(target_witness=SEMANTIC_BOUNDARY)
    )
    base = result.primitive_unary[:, 0]
    final = result.final_primitive_relevance[:, 0]
    assert result.corroborated_relation_region_masks[:, 0].tolist() == [False, True, False]
    assert torch.equal(final[:2], base[:2])
    assert final[2] > base[2]
    assert final[3] > base[3]
    assert torch.equal(final[4:], base[4:])


def test_invalid_target_witness_does_not_corroborate() -> None:
    inputs = _inputs(target_witness=0.9)
    inputs["primitive_valid"][2] = False
    result = corroborated_scale_aware_native_v3_support_fusion(**inputs)
    assert not bool(result.corroborated_relation_region_masks.any())
    assert torch.equal(result.final_primitive_relevance, result.primitive_unary)


def test_contract_adds_no_tunable_constant() -> None:
    contract = readout_contract()
    assert contract["semantic_boundary"] == SEMANTIC_BOUNDARY
    assert contract["invariants"]["new_tunable_constants"] is False
    assert contract["invariants"]["relation_cannot_create_first_target_region_foreground"]
