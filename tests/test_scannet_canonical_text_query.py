from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.eval_scannet_canonical_text_query import (
    load_frozen_text_cache,
    load_primitive_semantic_cache,
    project_primitive_semantics_to_points,
)


def _semantic_payload(*, source: str = "canonical_radio_primitive_neighborhood") -> dict:
    return {
        "schema_version": 1,
        "xyz": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "valid": torch.tensor([True, True]),
        "summary_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "metadata": {
            "source": source,
            "official_summary_head": True,
            "custom_text_projection": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }


def test_semantic_cache_loader_is_query_free_and_rejects_mpr_by_default(tmp_path):
    canonical = tmp_path / "canonical.pt"
    torch.save(_semantic_payload(), canonical)
    xyz, valid, features, metadata = load_primitive_semantic_cache(canonical)
    assert xyz.shape == (2, 3)
    assert valid.tolist() == [True, True]
    assert features.shape == (2, 2)
    assert metadata["custom_text_projection"] is False

    oracle = tmp_path / "oracle.pt"
    torch.save(_semantic_payload(source="mpr_radio_primitive_neighborhood"), oracle)
    with pytest.raises(ValueError, match="oracle diagnostics"):
        load_primitive_semantic_cache(oracle)
    load_primitive_semantic_cache(oracle, allow_mpr_oracle=True)


def test_primitive_projection_uses_only_valid_rows_and_is_normalized():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([True, True, False])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    query = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).numpy()
    projected = project_primitive_semantics_to_points(
        xyz,
        valid,
        features,
        query,
        k=2,
        device="cpu",
    ).float()
    assert projected[0, 0] > 0.999
    assert torch.allclose(
        projected[1],
        torch.tensor([2**-0.5, 2**-0.5]),
        atol=2e-3,
    )
    assert torch.allclose(projected.norm(dim=-1), torch.ones(2), atol=2e-3)


def test_frozen_text_cache_fails_closed_without_reencoding(tmp_path):
    cache = tmp_path / "text.pt"
    torch.save(
        {
            "queries": ["wall", "floor"],
            "prompt_templates": ["{query}"],
            "embeddings": torch.randn(2, 1536),
        },
        cache,
    )
    embeddings = load_frozen_text_cache(
        cache,
        class_names=["wall", "floor"],
        prompt_templates=["{query}"],
        device=torch.device("cpu"),
    )
    assert embeddings.shape == (2, 1536)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2), atol=1e-5)
    with pytest.raises(ValueError, match="query mismatch"):
        load_frozen_text_cache(
            cache,
            class_names=["floor", "wall"],
            prompt_templates=["{query}"],
            device=torch.device("cpu"),
        )
