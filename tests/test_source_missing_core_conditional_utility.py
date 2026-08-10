import torch

from radio_gs.interfaces.source_missing_core_conditional_utility import (
    source_missing_core_conditional_utility,
)


def test_missing_core_uses_strict_o0_and_labels_only_after_qualification():
    # Region 0 has exactly three of four scores strictly above .6 and one
    # missing unit at .6, so the inclusive 75% anchor gate passes.
    scores = torch.tensor(
        [[0.9], [0.8], [0.7], [0.6], [0.2]], dtype=torch.float32
    )
    rows = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    core = torch.ones_like(rows, dtype=torch.bool)
    valid = torch.ones(5, dtype=torch.bool)
    query = torch.tensor([0, -1], dtype=torch.long)
    instance = torch.tensor([1, 2], dtype=torch.long)
    mass = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.25, 0.75],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    result = source_missing_core_conditional_utility(
        o0_scores=scores,
        region_rows=rows,
        core_mask=core,
        primitive_valid_mask=valid,
        region_query_indices=query,
        region_dominant_instance_ids=instance,
        primitive_instance_mass=mass,
    )
    assert result.qualified_region_mask.tolist() == [True, False]
    assert result.positive_fraction.tolist() == [0.75, 0.0]
    assert result.unit_primitive_rows.tolist() == [3]
    assert result.unit_hard_labels.tolist() == [False]
    assert torch.allclose(
        result.unit_soft_target_mass_fraction, torch.tensor([0.25])
    )
    assert torch.allclose(result.unit_signed_utility, torch.tensor([-0.5]))


def test_missing_core_excludes_invalid_rows_from_anchor_and_units():
    scores = torch.tensor(
        [[0.9], [0.8], [0.7], [0.2], [0.1]], dtype=torch.float32
    )
    rows = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    core = torch.ones_like(rows, dtype=torch.bool)
    valid = torch.tensor([True, True, True, True, False])
    mass = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    result = source_missing_core_conditional_utility(
        o0_scores=scores,
        region_rows=rows,
        core_mask=core,
        primitive_valid_mask=valid,
        region_query_indices=torch.tensor([0]),
        region_dominant_instance_ids=torch.tensor([1]),
        primitive_instance_mass=mass,
    )
    assert result.valid_core_counts.item() == 4
    assert result.positive_fraction.item() == 0.75
    assert result.unit_primitive_rows.tolist() == [3]
    assert result.unit_hard_labels.tolist() == [True]


def test_missing_core_rejects_non_fp32_o0_scores():
    try:
        source_missing_core_conditional_utility(
            o0_scores=torch.ones(2, 1, dtype=torch.float16),
            region_rows=torch.tensor([[0, 1]]),
            core_mask=torch.ones(1, 2, dtype=torch.bool),
            primitive_valid_mask=torch.ones(2, dtype=torch.bool),
            region_query_indices=torch.tensor([0]),
            region_dominant_instance_ids=torch.tensor([1]),
            primitive_instance_mass=torch.ones(2, 2),
        )
    except ValueError as error:
        assert "inputs differ" in str(error)
    else:
        raise AssertionError("non-fp32 O0 must fail closed")
