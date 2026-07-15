import pytest
import torch

from radio_gs.querying.unified_query import cosine_relevancy_torch
from radio_gs.scripts.build_primitive_text_score_cache import (
    apply_completion_evidence,
    compile_scores,
)


def test_compile_relevancy_scores_matches_shared_query_primitive_readout() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    negatives = torch.tensor([[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
    valid = torch.tensor([True, False, True])

    actual = compile_scores(
        features,
        queries,
        valid,
        temperature=10.0,
        chunk_size=1,
        peak_normalize=False,
        scoring="relevancy",
        canonical=negatives,
    ).float()
    expected = cosine_relevancy_torch(
        torch.nn.functional.normalize(features[valid], dim=-1),
        torch.nn.functional.normalize(queries, dim=-1),
        torch.nn.functional.normalize(negatives, dim=-1),
        logit_scale=10.0,
        assume_normalized=True,
    )

    torch.testing.assert_close(actual[valid], expected, atol=5e-4, rtol=5e-4)
    assert torch.equal(actual[~valid], torch.zeros(1, 2))


def test_compile_relevancy_requires_generic_negatives() -> None:
    with pytest.raises(ValueError, match="requires canonical embeddings"):
        compile_scores(
            torch.eye(2),
            torch.eye(2),
            torch.ones(2, dtype=torch.bool),
            temperature=10.0,
            chunk_size=2,
            peak_normalize=False,
            scoring="relevancy",
        )


def test_peak_normalization_can_preserve_primary_domain() -> None:
    features = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    queries = torch.tensor([[1.0, 0.0]])
    scores = compile_scores(
        features,
        queries,
        torch.tensor([True, True]),
        temperature=1.0,
        chunk_size=2,
        peak_normalize=True,
        scoring="cosine",
        peak_mask=torch.tensor([False, True]),
    ).float()

    assert scores[1, 0] == pytest.approx(1.0)
    # A fallback row may be stronger, but it cannot change the primary peak.
    assert scores[0, 0] == pytest.approx(1.0)


def test_completion_evidence_applies_row_confidence() -> None:
    scores = torch.tensor(
        [[0.8, 0.3], [0.7, 0.9], [0.6, 0.8]], dtype=torch.float32
    )
    actual, stats = apply_completion_evidence(
        scores,
        torch.tensor([True, True, False]),
        semantic_confidence=torch.tensor([1.0, 0.25, 1.0]),
        routing="direct",
    )

    torch.testing.assert_close(
        actual.float(),
        torch.tensor([[0.8, 0.3], [0.175, 0.225], [0.0, 0.0]]),
        atol=5e-4,
        rtol=5e-4,
    )
    assert stats["semantic_confidence_applied"] is True


def test_completion_primary_first_preserves_supported_queries() -> None:
    scores = torch.tensor(
        [
            [0.8, 0.4],  # primary: query 0 is already supported
            [0.7, 0.9],  # fallback: only query 1 should remain
            [0.6, 0.8],  # invalid
        ],
        dtype=torch.float32,
    )
    actual, stats = apply_completion_evidence(
        scores,
        torch.tensor([True, True, False]),
        semantic_confidence=torch.tensor([1.0, 0.5, 1.0]),
        primary_valid=torch.tensor([True, False, False]),
        routing="primary_first",
        primary_support_threshold=0.5,
    )

    torch.testing.assert_close(
        actual.float(),
        torch.tensor([[0.8, 0.4], [0.0, 0.45], [0.0, 0.0]]),
        atol=5e-4,
        rtol=5e-4,
    )
    assert stats["primary_supported_queries"] == 1
    assert stats["primary_valid_count"] == 1
    assert stats["fallback_valid_count"] == 1


def test_completion_rejects_primary_rows_outside_valid_support() -> None:
    with pytest.raises(ValueError, match="must also be valid"):
        apply_completion_evidence(
            torch.eye(2),
            torch.tensor([True, False]),
            primary_valid=torch.tensor([False, True]),
        )


def test_completion_primary_only_zeros_every_fallback_score() -> None:
    actual, stats = apply_completion_evidence(
        torch.tensor([[0.8], [0.9], [0.7]]),
        torch.tensor([True, True, True]),
        primary_valid=torch.tensor([True, False, False]),
        routing="primary_only",
    )

    torch.testing.assert_close(
        actual.float(), torch.tensor([[0.8], [0.0], [0.0]]), atol=5e-4, rtol=5e-4
    )
    assert stats["routing"] == "primary_only"
