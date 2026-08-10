from __future__ import annotations

import pytest
import torch

from radio_gs.querying.scale_aware_native_v3_support_fusion import (
    MAXIMUM_REGIONS,
    RELATION_THRESHOLD,
    SEMANTIC_BOUNDARY,
    readout_contract,
    scale_aware_native_v3_support_fusion,
)


def _inputs() -> dict[str, torch.Tensor]:
    # Four regions, each with a two-primitive semantic core.  Region zero is
    # the selected-scale seed; region one is its native-V3 relation target.
    relevance = torch.tensor(
        [
            [[0.95], [0.10]],
            [[0.20], [0.10]],
            [[0.55], [0.30]],
            [[0.45], [0.20]],
            [[0.99], [0.99]],
            [[0.10], [0.40]],
            [[0.10], [0.90]],
            [[0.10], [0.80]],
        ],
        dtype=torch.float32,
    )
    return {
        "primitive_relevance_by_scale": relevance,
        "selected_scale_indices": torch.tensor([0]),
        "primitive_valid": torch.ones(8, dtype=torch.bool),
        "region_rows": torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long
        ),
        "semantic_core_mask": torch.ones((4, 2), dtype=torch.bool),
        "region_anchor_positions": torch.zeros(4, dtype=torch.long),
        "region_scale_indices": torch.tensor([0, 0, 1, 1]),
        "pair_indices": torch.tensor([[0], [1]], dtype=torch.long),
        "pair_probabilities": torch.tensor([0.925]),
    }


def test_relation_is_monotone_continuous_and_does_not_fill_seed_core() -> None:
    result = scale_aware_native_v3_support_fusion(**_inputs())
    base = result.primitive_unary[:, 0]
    final = result.final_primitive_relevance[:, 0]

    assert result.seed_region_indices.tolist() == [0]
    assert result.query_gate.tolist() == [True]
    # Seed core is untouched: it is not converted to binary foreground.
    assert torch.equal(final[:2], base[:2])
    # Region one receives continuous relation completion.
    assert final[2] > base[2]
    assert final[3] > base[3]
    assert 0.0 < final[3].item() < 1.0
    assert torch.equal(final[4:], base[4:])
    assert result.changed_primitive_query_cells == 2


def test_isolated_seed_is_bitwise_exact_primitive_unary() -> None:
    inputs = _inputs()
    inputs["pair_probabilities"] = torch.tensor([0.84])
    result = scale_aware_native_v3_support_fusion(**inputs)
    assert torch.equal(result.final_primitive_relevance, result.primitive_unary)
    assert result.changed_primitive_query_cells == 0


def test_failed_semantic_gate_is_bitwise_exact_primitive_unary() -> None:
    inputs = _inputs()
    inputs["primitive_relevance_by_scale"][:, 0, 0] *= 0.5
    result = scale_aware_native_v3_support_fusion(**inputs)
    assert result.query_gate.tolist() == [False]
    assert torch.equal(result.final_primitive_relevance, result.primitive_unary)


def test_off_scale_distractor_cannot_steal_seed() -> None:
    inputs = _inputs()
    # Region two has the globally largest anchor score, but belongs to scale 1
    # while the frozen query scale is zero.
    assert inputs["primitive_relevance_by_scale"][4, 1, 0] > inputs[
        "primitive_relevance_by_scale"
    ][0, 0, 0]
    result = scale_aware_native_v3_support_fusion(**inputs)
    assert result.seed_region_indices.tolist() == [0]


def test_relation_target_uses_its_registered_scale() -> None:
    inputs = _inputs()
    # Connect the selected-scale seed directly to the scale-one region two.
    inputs["pair_indices"] = torch.tensor([[0], [2]])
    inputs["pair_probabilities"] = torch.tensor([0.925])
    result = scale_aware_native_v3_support_fusion(**inputs)
    final = result.final_primitive_relevance[:, 0]
    # Primitive four uses its registered scale-one value 0.99 and therefore
    # receives only the remaining one-percent noisy-OR headroom.
    expected_four = 0.99 + (1.0 - 0.99) * 0.875 * 0.5
    assert final[4].item() == pytest.approx(expected_four)
    # Primitive five must start from its region's scale-one value 0.40, not
    # its selected-scale value 0.99; 0.925 gives normalized path excess 0.5.
    expected = 0.40 + (1.0 - 0.40) * 0.875 * 0.5
    assert final[5].item() == pytest.approx(expected)


def test_invalid_primitive_is_never_completed() -> None:
    inputs = _inputs()
    inputs["primitive_valid"][3] = False
    result = scale_aware_native_v3_support_fusion(**inputs)
    assert result.final_primitive_relevance[3, 0].item() == 0.0
    assert result.primitive_unary[3, 0].item() == 0.0


def test_overlapping_relation_regions_use_deterministic_max_not_sum() -> None:
    inputs = _inputs()
    inputs["region_rows"] = torch.tensor(
        [[0, 1], [2, 3], [3, 4], [6, 7]], dtype=torch.long
    )
    inputs["region_scale_indices"] = torch.zeros(4, dtype=torch.long)
    inputs["pair_indices"] = torch.tensor([[0, 0], [1, 2]])
    inputs["pair_probabilities"] = torch.tensor([0.925, 1.0])
    result = scale_aware_native_v3_support_fusion(**inputs)
    # Region two has the stronger path and therefore owns the overlap via max.
    expected = 0.45 + (1.0 - 0.45) * 0.875
    assert result.final_primitive_relevance[3, 0].item() == pytest.approx(expected)


def test_contract_binds_frozen_constants_and_forbids_binary_union() -> None:
    contract = readout_contract()
    assert contract["semantic_boundary"] == SEMANTIC_BOUNDARY
    assert contract["relation"]["source_promoted_native_v3_threshold"] == (
        RELATION_THRESHOLD
    )
    assert contract["relation"]["maximum_regions"] == MAXIMUM_REGIONS
    assert contract["invariants"]["binary_region_union"] is False
    assert contract["invariants"]["scene_specific_parameters"] is False


@pytest.mark.parametrize("drift", ["scale", "anchor", "duplicate", "range"])
def test_invalid_region_or_scale_authority_fails_closed(drift: str) -> None:
    inputs = _inputs()
    if drift == "scale":
        inputs["region_scale_indices"][0] = 2
    elif drift == "anchor":
        inputs["region_anchor_positions"][0] = 2
    elif drift == "duplicate":
        inputs["region_rows"][0] = torch.tensor([0, 0])
    else:
        inputs["region_rows"][0, 0] = 99
    with pytest.raises(ValueError):
        scale_aware_native_v3_support_fusion(**inputs)
