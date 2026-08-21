from __future__ import annotations

import torch

from radio_gs.scripts.build_lerf_anchor_preserving_latent_completion import compose


def _payload(scores: torch.Tensor) -> dict[str, object]:
    rows = scores.shape[0]
    return {
        "scene": "test",
        "query_scores": scores,
        "identity_query_scores": scores.clone(),
        "xyz": torch.arange(rows * 3).reshape(rows, 3).float(),
        "valid": torch.tensor([True] * (rows - 1) + [False]),
        "metadata": {"query_names": ["a", "b"]},
    }


def test_completion_can_only_add_valid_support_and_preserves_invalid_anchor() -> None:
    anchor = _payload(torch.tensor([[0.8, 0.2], [0.3, 0.7], [-1.0e4, -1.0e4]]))
    marginal = _payload(torch.tensor([[0.4, 0.6], [0.9, 0.1], [-2.0e4, -2.0e4]]))
    result = compose(anchor, marginal)
    assert torch.equal(
        result["query_scores"],
        torch.tensor([[0.8, 0.6], [0.9, 0.7], [-1.0e4, -1.0e4]]),
    )
    assert torch.equal(result["identity_query_scores"], anchor["identity_query_scores"])
    metadata = result["metadata"]
    assert metadata["separate_identity_localization"] is True
    assert metadata["localization_authority"] == "field_siglip2_relevancy_identity"
    assert str(metadata["typed_posterior"]).startswith(
        "official_sam3_siglip2_identity_extent_factorization_"
    )


def test_completion_rejects_different_query_identity() -> None:
    anchor = _payload(torch.ones(3, 2))
    marginal = _payload(torch.ones(3, 2))
    marginal["metadata"] = {"query_names": ["b", "a"]}
    try:
        compose(anchor, marginal)
    except ValueError as error:
        assert "identities differ" in str(error)
    else:
        raise AssertionError("different query identity was accepted")
