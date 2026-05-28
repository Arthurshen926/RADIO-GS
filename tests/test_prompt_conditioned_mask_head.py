import torch

from radio_gs.models.prompt_conditioned_mask_head import PromptConditionedMaskHead


def test_prompt_conditioned_mask_head_outputs_query_masks():
    head = PromptConditionedMaskHead(feature_dim=8, prompt_dim=6, hidden_dim=12)
    features = torch.randn(2, 8, 5, 7)
    prompts = torch.randn(2, 3, 6)
    coarse = torch.zeros(2, 3, 5, 7)

    logits = head(features, prompts, coarse)

    assert logits.shape == (2, 3, 5, 7)


def test_prompt_conditioned_mask_head_changes_with_prompt():
    head = PromptConditionedMaskHead(feature_dim=8, prompt_dim=6, hidden_dim=12)
    features = torch.randn(1, 8, 5, 7)
    coarse = torch.zeros(1, 2, 5, 7)
    prompt_a = torch.zeros(1, 6)
    prompt_b = torch.ones(1, 6)
    prompts = torch.stack([prompt_a, prompt_b], dim=1)

    logits = head(features, prompts, coarse)

    assert not torch.allclose(logits[:, 0], logits[:, 1])


def test_prompt_conditioned_mask_head_uses_coarse_mask_spatial_prompt():
    head = PromptConditionedMaskHead(feature_dim=8, prompt_dim=6, hidden_dim=12)
    features = torch.randn(1, 8, 5, 7)
    prompts = torch.randn(1, 1, 6)
    coarse_a = torch.zeros(1, 1, 5, 7)
    coarse_b = torch.zeros(1, 1, 5, 7)
    coarse_b[:, :, 1:4, 2:5] = 1.0

    logits_a = head(features, prompts, coarse_a)
    logits_b = head(features, prompts, coarse_b)

    assert not torch.allclose(logits_a, logits_b)


def test_prompt_conditioned_mask_head_can_predict_query_quality():
    head = PromptConditionedMaskHead(
        feature_dim=8,
        prompt_dim=6,
        hidden_dim=12,
        predict_quality=True,
    )
    features = torch.randn(2, 8, 5, 7)
    prompts = torch.randn(2, 3, 6)
    coarse = torch.zeros(2, 3, 5, 7)

    logits, quality_logits = head.forward_with_quality(features, prompts, coarse)

    assert logits.shape == (2, 3, 5, 7)
    assert quality_logits.shape == (2, 3)
    assert torch.isfinite(quality_logits).all()
