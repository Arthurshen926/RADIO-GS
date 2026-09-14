from __future__ import annotations

import pytest
import torch

from radio_gs.v4.evaluation.lerf_fragment_text_evaluator import (
    compose_local_region_queries,
    compose_ranked_fragment_union,
    oracle_fragment_retrieval,
)


def test_oracle_fragment_retrieval_reports_ranked_recall():
    result = oracle_fragment_retrieval(
        torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.7, 0.1]]),
        ["first", "second"],
        {
            "first": {"stable_top2_token_ids": [1, 2]},
            "second": {"stable_top2_token_ids": [1]},
        },
        capacity=2,
    )
    assert result["recall_at_1"] == pytest.approx(0.5)
    assert result["recall_at_3"] == pytest.approx(1.0)
    assert result["per_category"]["second"]["best_oracle_fragment_rank"] == 2


def test_compose_ranked_fragment_union_preserves_overlapping_fragments():
    membership = torch.tensor([
        [0.8, 0.0, 0.2],
        [0.5, 0.5, 0.0],
        [0.0, 0.7, 0.6],
    ])
    scores = torch.tensor([[0.9, 0.8, 0.1], [0.0, 0.2, 0.7]])
    posterior, audit = compose_ranked_fragment_union(
        membership, scores, maximum_query_tokens=2
    )
    assert posterior[:, 0].tolist() == pytest.approx([0.8, 0.75, 0.7])
    assert posterior[:, 1].tolist() == pytest.approx([0.2, 0.5, 0.88])
    assert audit["assignment_compression"] == "none_raw_fragment_membership"
    assert audit["promotion_eligible"] is False


def test_compose_ranked_fragment_union_rejects_bad_capacity():
    with pytest.raises(ValueError, match="capacity"):
        compose_ranked_fragment_union(torch.ones(2, 2), torch.ones(1, 2), maximum_query_tokens=3)


def test_local_region_query_uses_strongest_overlapping_region_without_top2():
    posterior, audit = compose_local_region_queries(
        torch.tensor([[0.8, 0.2, 0.7], [0.1, 0.9, 0.3]]),
        torch.tensor([[0.5, 0.9, 0.2], [0.1, 0.4, 0.8]]),
        fragment_chunk_size=2,
    )
    assert torch.allclose(posterior, torch.tensor([[0.4, 0.56], [0.81, 0.36]]))
    assert audit["selection_mode"] == "local_semantic"
    assert audit["object_codebook_used"] is False
