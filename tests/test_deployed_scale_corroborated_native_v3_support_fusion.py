from __future__ import annotations

import torch

from radio_gs.querying.deployed_scale_corroborated_native_v3_support_fusion import (
    SEMANTIC_BOUNDARY,
    deployed_scale_corroborated_native_v3_support_fusion,
    readout_contract,
)


def _inputs(*, deployed_witness: float) -> dict[str, torch.Tensor]:
    # Query is deployed at scale zero.  Relation target is registered at scale
    # one, where it always looks strong; only scale-zero evidence may witness it.
    relevance = torch.tensor(
        [
            [[0.95], [0.10]],
            [[0.20], [0.10]],
            [[deployed_witness], [0.90]],
            [[0.45], [0.50]],
        ],
        dtype=torch.float32,
    )
    return {
        "primitive_relevance_by_scale": relevance,
        "selected_scale_indices": torch.tensor([0]),
        "primitive_valid": torch.ones(4, dtype=torch.bool),
        "region_rows": torch.tensor([[0, 1], [2, 3]]),
        "semantic_core_mask": torch.ones((2, 2), dtype=torch.bool),
        "region_anchor_positions": torch.zeros(2, dtype=torch.long),
        "region_scale_indices": torch.tensor([0, 1]),
        "pair_indices": torch.tensor([[0], [1]], dtype=torch.long),
        "pair_probabilities": torch.tensor([0.925]),
    }


def test_registered_only_evidence_cannot_corroborate_deployed_target() -> None:
    result = deployed_scale_corroborated_native_v3_support_fusion(
        **_inputs(deployed_witness=SEMANTIC_BOUNDARY - 0.01)
    )
    assert not bool(result.deployed_corroborated_region_masks.any())
    assert torch.equal(result.final_primitive_relevance, result.primitive_unary)


def test_deployed_scale_witness_allows_registered_scale_completion() -> None:
    result = deployed_scale_corroborated_native_v3_support_fusion(
        **_inputs(deployed_witness=SEMANTIC_BOUNDARY)
    )
    assert result.deployed_corroborated_region_masks[:, 0].tolist() == [False, True]
    assert result.final_primitive_relevance[2, 0] > result.primitive_unary[2, 0]
    assert result.final_primitive_relevance[3, 0] > result.primitive_unary[3, 0]


def test_contract_adds_no_tunable_constant_and_matches_consumer_scale() -> None:
    contract = readout_contract()
    assert contract["semantic_boundary"] == 0.6
    assert contract["invariants"]["new_tunable_constants"] is False
    assert contract["invariants"]["corroboration_matches_deployed_consumer_scale"]
