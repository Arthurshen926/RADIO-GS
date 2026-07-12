import torch
import pytest

from radio_gs.evaluation.openclip_readout import (
    cosine_logits,
    load_or_generate_openclip_prompt_ensemble_embeddings,
    normalized_embeddings,
)


def test_normalized_embeddings_leaves_zero_rows_zero():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    y = normalized_embeddings(x)

    assert torch.allclose(y[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(y[1], torch.zeros(2))


def test_normalized_embeddings_rejects_non_finite_rows():
    with pytest.raises(FloatingPointError, match="non-finite OpenCLIP"):
        normalized_embeddings(torch.tensor([[float("nan"), 0.0]]))


def test_cosine_logits_scores_features_against_text():
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    logits = cosine_logits(features, text)

    assert torch.allclose(logits, torch.eye(2), atol=1e-6)


def test_openclip_prompt_ensemble_uses_matching_cache(tmp_path):
    cache_path = tmp_path / "openclip_text.pt"
    embeddings = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    torch.save(
        {
            "queries": ["chair", "table"],
            "prompt_templates": ["a {query}"],
            "text_encoder": "openclip",
            "openclip_model": "ViT-B-16",
            "openclip_pretrained": "laion2b_s34b_b88k",
            "embeddings": embeddings,
        },
        cache_path,
    )

    cached = load_or_generate_openclip_prompt_ensemble_embeddings(
        ["chair", "table"],
        torch.device("cpu"),
        cache_path=cache_path,
        prompt_templates=["a {query}"],
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
    )

    assert torch.allclose(cached, torch.tensor([[0.6, 0.8], [0.0, 1.0]]))


def test_openclip_prompt_ensemble_selects_subset_without_overwriting_cache(tmp_path):
    cache_path = tmp_path / "openclip_text.pt"
    payload = {
        "queries": ["chair", "table", "lamp"],
        "prompt_templates": ["{query}"],
        "text_encoder": "openclip",
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "embeddings": torch.tensor([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]]),
    }
    torch.save(payload, cache_path)
    original_bytes = cache_path.read_bytes()

    cached = load_or_generate_openclip_prompt_ensemble_embeddings(
        ["lamp", "chair"],
        torch.device("cpu"),
        cache_path=cache_path,
        prompt_templates=["{query}"],
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
    )

    assert torch.allclose(cached, torch.tensor([[1.0, 0.0], [0.6, 0.8]]))
    assert cache_path.read_bytes() == original_bytes
