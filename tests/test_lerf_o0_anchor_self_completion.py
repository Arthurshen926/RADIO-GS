from __future__ import annotations

import torch

from radio_gs.interfaces import lerf_o0_anchor_self_completion as formal


def _run(scores: torch.Tensor, *, eligible: torch.Tensor | None = None):
    rows = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=torch.int64)
    core = torch.ones_like(rows, dtype=torch.bool)
    return formal.o0_anchor_self_completion(
        o0_scores=scores.float(),
        region_rows=rows,
        core_mask=core,
        primitive_valid_mask=torch.ones(6, dtype=torch.bool),
        region_eligible_mask=(
            torch.ones(2, dtype=torch.bool) if eligible is None else eligible
        ),
    )


def test_qualified_anchor_self_fills_and_overlap_uses_maximum() -> None:
    scores = torch.tensor(
        [
            [0.8, 0.1],
            [0.7, 0.1],
            [0.9, 0.8],
            [0.2, 0.9],
            [0.1, 0.7],
            [0.1, 0.2],
        ]
    )
    result = _run(scores)
    assert result.qualified_anchor_mask.tolist() == [[True, False], [False, True]]
    assert torch.equal(
        result.completion_probability[:, 0],
        torch.tensor([0.75, 0.75, 0.75, 0.75, 0.0, 0.0]),
    )
    assert torch.equal(
        result.completion_probability[:, 1],
        torch.tensor([0.0, 0.0, 0.75, 0.75, 0.75, 0.75]),
    )
    assert float(result.final_scores[3, 0]) == 0.75
    assert torch.equal(result.final_scores[2, 1], scores[2, 1])


def test_strict_o0_threshold_and_inclusive_supermajority() -> None:
    scores = torch.tensor(
        [[0.6], [0.61], [0.7], [0.8], [0.1], [0.1]], dtype=torch.float32
    )
    result = _run(scores)
    assert result.o0_positive_fraction[:, 0].tolist() == [0.75, 0.5]
    assert result.qualified_anchor_mask[:, 0].tolist() == [True, False]


def test_ineligible_or_no_anchor_is_bitwise_o0() -> None:
    scores = torch.tensor(
        [[0.9], [0.8], [0.7], [0.1], [0.1], [0.1]], dtype=torch.float32
    )
    result = _run(scores, eligible=torch.zeros(2, dtype=torch.bool))
    assert torch.equal(result.final_scores, scores)
    assert not bool(result.changed_mask.any())


def test_invalid_core_rows_are_not_counted_or_changed() -> None:
    scores = torch.tensor(
        [[0.9], [0.8], [0.7], [0.1], [0.1], [0.1]], dtype=torch.float32
    )
    rows = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    valid = torch.tensor([True, True, True, False, True, True])
    result = formal.o0_anchor_self_completion(
        o0_scores=scores,
        region_rows=rows,
        core_mask=torch.ones_like(rows, dtype=torch.bool),
        primitive_valid_mask=valid,
        region_eligible_mask=torch.ones(1, dtype=torch.bool),
    )
    assert result.valid_core_counts.tolist() == [3]
    assert result.qualified_anchor_mask.tolist() == [[True]]
    assert float(result.final_scores[3, 0]) == float(scores[3, 0])


def test_query_permutation_equivariance() -> None:
    scores = torch.tensor(
        [
            [0.8, 0.1],
            [0.7, 0.1],
            [0.9, 0.8],
            [0.2, 0.9],
            [0.1, 0.7],
            [0.1, 0.2],
        ]
    )
    original = _run(scores)
    order = torch.tensor([1, 0])
    permuted = _run(scores[:, order])
    assert torch.equal(permuted.final_scores, original.final_scores[:, order])
    assert torch.equal(
        permuted.qualified_anchor_mask,
        original.qualified_anchor_mask[:, order],
    )


def test_external_cache_requires_complete_exact_replay_lineage() -> None:
    scores = torch.tensor(
        [[0.9], [0.8], [0.7], [0.1], [0.1], [0.1]], dtype=torch.float32
    )
    result = _run(scores)
    names = {
        "exact_o0_cache",
        "positive_o0_cache",
        "negative_o0_cache",
        "region_features",
        "target_descriptor",
        "factorized_primitive_state",
        "renderer_geometry_checkpoint",
    }
    records = {
        name: {"path": f"/tmp/{name}", "sha256": "0" * 64} for name in names
    }
    cache = formal.build_external_query_score_cache(
        result=result,
        o0_valid=torch.ones(6, dtype=torch.bool),
        o0_xyz=torch.zeros(6, 3, dtype=torch.float32),
        query_names=["opaque"],
        scene_id="scene",
        input_authority=records,
    )
    assert formal.validate_external_query_score_cache(cache) == cache
    records.pop("negative_o0_cache")
    try:
        formal.build_external_query_score_cache(
            result=result,
            o0_valid=torch.ones(6, dtype=torch.bool),
            o0_xyz=torch.zeros(6, 3, dtype=torch.float32),
            query_names=["opaque"],
            scene_id="scene",
            input_authority=records,
        )
    except ValueError as error:
        assert "axes differ" in str(error)
    else:
        raise AssertionError("incomplete exact replay lineage was accepted")
