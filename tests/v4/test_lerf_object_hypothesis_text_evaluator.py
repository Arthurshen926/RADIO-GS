from __future__ import annotations

import pytest
import torch

from radio_gs.v4.evaluation.lerf_object_hypothesis_text_evaluator import (
    _flatten_object_prototypes,
    prototype_fragment_consensus_scores,
    oracle_hypothesis_retrieval,
)


def test_object_prototype_flattening_preserves_hypothesis_ids():
    descriptors = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    valid = torch.tensor([[True, False, True], [False, True, False]])
    flat, token_ids = _flatten_object_prototypes({
        "object_prototype_descriptors": descriptors,
        "object_prototype_valid": valid,
    })
    assert flat.shape == (3, 4)
    assert token_ids.tolist() == [0, 0, 1]


def test_object_prototype_flattening_rejects_missing_receipts():
    with pytest.raises(ValueError, match="lacks aligned"):
        _flatten_object_prototypes({})


def test_oracle_hypothesis_retrieval_reports_ranked_recall():
    result = oracle_hypothesis_retrieval(
        torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.7, 0.1]]),
        ["first", "second"],
        {
            "first": {"stable_top3_token_ids": [1, 2]},
            "second": {"stable_top3_token_ids": [1]},
        },
    )
    assert result["recall_at_1"] == pytest.approx(0.5)
    assert result["recall_at_3"] == pytest.approx(1.0)
    assert result["per_category"]["second"]["best_oracle_hypothesis_rank"] == 2


def test_fragment_consensus_counts_distinct_fragments_not_crop_variants():
    hypothesis = {
        "object_prototype_descriptors": torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
        "object_prototype_valid": torch.tensor([[True, True, True]]),
        "object_prototype_fragment_ids": torch.tensor([[4, 4, 9]]),
    }
    score = prototype_fragment_consensus_scores(
        hypothesis, torch.tensor([[1.0, 0.0]]), consensus_fragments=2
    )
    assert score.item() == pytest.approx(0.5)


def test_same_view_nested_fragments_do_not_fake_multiview_consensus():
    hypothesis = {
        "object_prototype_descriptors": torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]]),
        "object_prototype_valid": torch.ones(1, 3, dtype=torch.bool),
        "object_prototype_fragment_ids": torch.tensor([[0, 1, 2]]),
        "object_prototype_view_ids": torch.tensor([[5, 5, 6]]),
    }
    assert prototype_fragment_consensus_scores(hypothesis, torch.tensor([[1., 0.]])).item() == pytest.approx(.5)
    del hypothesis["object_prototype_view_ids"]
    assert prototype_fragment_consensus_scores(hypothesis, torch.tensor([[1., 0.]])).item() == pytest.approx(1.)
