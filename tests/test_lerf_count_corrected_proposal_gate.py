from __future__ import annotations

import torch

from radio_gs.scripts.build_lerf_count_corrected_proposal_gate import compose


def _payload(scores: torch.Tensor, *, null=None, counts=None) -> dict[str, object]:
    rows, queries = scores.shape
    metadata: dict[str, object] = {"query_names": [f"q{i}" for i in range(queries)]}
    if null is not None:
        metadata["topology"] = {
            "null_probability": null,
            "valid_proposal_counts": counts,
        }
    return {
        "scene": "test",
        "query_scores": scores,
        "identity_query_scores": scores.clone(),
        "xyz": torch.arange(rows * 3).reshape(rows, 3).float(),
        "valid": torch.tensor([True] * (rows - 1) + [False]),
        "metadata": metadata,
    }


def test_gate_is_count_corrected_and_falls_back_per_query() -> None:
    anchor = _payload(torch.tensor([[0.8, 0.2, 0.3], [-1e4, -1e4, -1e4]]))
    marginal = _payload(
        torch.tensor([[0.4, 0.9, 0.7], [-2e4, -2e4, -2e4]]),
        null=[0.2, 0.3, 1.0],
        counts=[1, 3, 0],
    )
    result = compose(anchor, marginal)
    assert result["metadata"]["admitted_queries"] == [True, False, False]
    assert torch.equal(result["query_scores"], torch.tensor([[0.4, 0.2, 0.3], [-1e4, -1e4, -1e4]]))
    assert result["metadata"]["gaussian_union"] is False


def test_gate_rejects_missing_latent_authority() -> None:
    anchor = _payload(torch.ones(2, 1))
    marginal = _payload(torch.ones(2, 1))
    try:
        compose(anchor, marginal)
    except ValueError as error:
        assert "topology" in str(error)
    else:
        raise AssertionError("missing latent authority was accepted")
