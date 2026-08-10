from __future__ import annotations

import pytest
import torch

from radio_gs.evaluation.primitive_instance_union import (
    primitive_instance_union_metrics,
    region_seed_instance_evidence,
)


def _authority() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mass = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 0.0, 0.0],
        ]
    )
    rows = torch.tensor([[0, 1], [1, 2], [1, 3]], dtype=torch.int64)
    mask = torch.ones_like(rows, dtype=torch.bool)
    return rows, mask, mass


def test_seed_targets_use_positive_mass_and_lowest_id_tie_break() -> None:
    rows, mask, mass = _authority()
    evidence = region_seed_instance_evidence(
        region_rows=rows,
        token_mask=mask,
        primitive_instance_mass=mass,
    )
    assert evidence["dominant_instance_ids"].tolist() == [1, 2, 1]
    assert evidence["dominant_instance_mass"].tolist() == [3.0, 3.0, 1.0]
    tie_mass = torch.tensor([[0.0, 1.0, 1.0]])
    tie = region_seed_instance_evidence(
        region_rows=torch.tensor([[0]]),
        token_mask=torch.tensor([[True]]),
        primitive_instance_mass=tie_mass,
    )
    assert tie["dominant_instance_ids"].tolist() == [1]


def test_union_metric_deduplicates_shared_primitives_and_counts_background() -> None:
    rows, mask, mass = _authority()
    result = primitive_instance_union_metrics(
        region_rows=rows,
        token_mask=mask,
        primitive_instance_mass=mass,
        selections_by_seed={0: [0, 2], 1: [1], 2: [2]},
        maximum_regions=2,
    )
    instance_one = result["per_instance"]["1"]
    # Seed 0 union is primitives {0,1,3}: correct=3, selected=7, target=3.
    # Primitive 1 is present in both regions but is counted exactly once.
    seed_zero_iou = 3.0 / 7.0
    seed_two_iou = 1.0 / 7.0
    expected_instance_one_iou = (3.0 * seed_zero_iou + seed_two_iou) / 4.0
    assert instance_one["iou"] == pytest.approx(expected_instance_one_iou)
    assert instance_one["selected_unique_primitives"] == pytest.approx(2.75)
    assert instance_one["selected_regions"] == pytest.approx(1.75)
    assert instance_one["contamination"] > 0
    assert result["eligible_seeds"] == 3
    assert result["eligible_instances"] == 2


@pytest.mark.parametrize(
    "selections,maximum",
    [
        ({0: [0], 1: [1]}, 2),
        ({0: [2, 0], 1: [1], 2: [2]}, 2),
        ({0: [0, 0], 1: [1], 2: [2]}, 2),
        ({0: [0, 1, 2], 1: [1], 2: [2]}, 2),
    ],
)
def test_union_metric_fails_closed_on_incomplete_or_invalid_selections(
    selections: dict[int, list[int]], maximum: int
) -> None:
    rows, mask, mass = _authority()
    with pytest.raises(ValueError, match="selection"):
        primitive_instance_union_metrics(
            region_rows=rows,
            token_mask=mask,
            primitive_instance_mass=mass,
            selections_by_seed=selections,
            maximum_regions=maximum,
        )


def test_union_metric_rejects_negative_mass_and_invalid_active_row() -> None:
    rows, mask, mass = _authority()
    negative = mass.clone()
    negative[0, 1] = -1
    with pytest.raises(ValueError, match="authority"):
        region_seed_instance_evidence(
            region_rows=rows,
            token_mask=mask,
            primitive_instance_mass=negative,
        )
    bad_rows = rows.clone()
    bad_rows[0, 0] = 99
    with pytest.raises(ValueError, match="authority"):
        region_seed_instance_evidence(
            region_rows=bad_rows,
            token_mask=mask,
            primitive_instance_mass=mass,
        )
