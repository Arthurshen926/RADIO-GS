from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.interfaces.lerf_o0_conditional_missing_core_completion import (
    O0_SCORE_MINIMUM,
    build_external_query_score_cache,
    conditional_missing_core_completion,
    validate_external_query_score_cache,
)
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    MonotoneAdditiveLogistic,
)


def _models() -> tuple[MonotoneAdditiveLogistic, ...]:
    weight = torch.zeros(6)
    weight[0] = 10.0
    model = MonotoneAdditiveLogistic(
        location=torch.zeros(6),
        scale=torch.ones(6),
        positive_weights=weight,
        bias=torch.tensor(-5.0),
    )
    return (model, model, model)


def _run(scores: torch.Tensor, selected_scales: torch.Tensor):
    rows = torch.arange(10).reshape(2, 5)
    return conditional_missing_core_completion(
        o0_scores=scores.float().contiguous(),
        region_rows=rows.long(),
        core_mask=torch.ones_like(rows, dtype=torch.bool),
        primitive_valid_mask=torch.ones(10, dtype=torch.bool),
        appearance_concentration=torch.tensor([0.9, 0.9]),
        boundary_concentration=torch.tensor([0.9, 0.9]),
        core_spatial_rms_radius=torch.tensor([0.1, 0.1]),
        selected_query_scale_indices=selected_scales.long(),
        full_scalar_source_robust_ood_linf=torch.tensor([0.5, 0.5]),
        fold_models=_models(),
        threshold_inclusive=0.5,
    )


def test_selector_rejects_low_confidence_FIX6_candidate_and_preserves_exact_o0() -> None:
    scores = torch.tensor(
        [
            [0.8],
            [0.8],
            [0.8],
            [0.8],
            [0.55],
            [0.8],
            [0.8],
            [0.8],
            [0.8],
            [0.20],
        ]
    )
    result = _run(scores, torch.tensor([0]))
    assert int(result.unit_region_indices.numel()) == 2
    assert result.selected_unit_mask.tolist() == [True, False]
    assert result.final_scores[4, 0] > O0_SCORE_MINIMUM
    assert result.final_scores[9, 0] == scores[9, 0]
    assert torch.equal(result.changed_mask, result.selected_cell_mask)
    assert torch.equal(
        result.final_scores[~result.selected_cell_mask],
        scores[~result.selected_cell_mask],
    )


def test_query_permutation_is_exactly_equivariant() -> None:
    first = torch.tensor([0.8, 0.8, 0.8, 0.8, 0.55, 0.8, 0.8, 0.8, 0.8, 0.2])
    second = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.51, 0.7, 0.7, 0.7, 0.7, 0.1])
    scores = torch.stack((first, second), dim=1)
    direct = _run(scores, torch.tensor([0, 1]))
    permutation = torch.tensor([1, 0])
    permuted = _run(scores[:, permutation], torch.tensor([1, 0]))
    assert torch.equal(permuted.final_scores, direct.final_scores[:, permutation])
    assert torch.equal(permuted.changed_mask, direct.changed_mask[:, permutation])


def test_invalid_primitive_remains_bitwise_exact_o0() -> None:
    scores = torch.tensor([[0.8], [0.8], [0.8], [0.8], [0.55]] * 2)
    rows = torch.arange(10).reshape(2, 5)
    valid = torch.ones(10, dtype=torch.bool)
    valid[4] = False
    result = conditional_missing_core_completion(
        o0_scores=scores.float(),
        region_rows=rows.long(),
        core_mask=torch.ones_like(rows, dtype=torch.bool),
        primitive_valid_mask=valid,
        appearance_concentration=torch.tensor([0.9, 0.9]),
        boundary_concentration=torch.tensor([0.9, 0.9]),
        core_spatial_rms_radius=torch.tensor([0.1, 0.1]),
        selected_query_scale_indices=torch.tensor([0]),
        full_scalar_source_robust_ood_linf=torch.tensor([0.5, 0.5]),
        fold_models=_models(),
        threshold_inclusive=0.5,
    )
    assert torch.equal(result.final_scores[~valid], scores[~valid])


def test_external_cache_fails_closed_on_score_tamper() -> None:
    scores = torch.tensor([[0.8], [0.8], [0.8], [0.8], [0.55]] * 2)
    result = _run(scores, torch.tensor([0]))
    payload = build_external_query_score_cache(
        result=result,
        o0_valid=torch.ones(10, dtype=torch.bool),
        o0_xyz=torch.arange(30, dtype=torch.float32).reshape(10, 3),
        query_names=["query"],
        scene_id="scene",
        input_authority={
            "input": {"path": "/tmp/input.pt", "sha256": "1" * 64}
        },
        threshold_inclusive=0.5,
    )
    changed = copy.deepcopy(payload)
    changed["query_scores"][0, 0] -= 0.01
    with pytest.raises(ValueError, match="channel changed"):
        validate_external_query_score_cache(changed)
