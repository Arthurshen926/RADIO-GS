import torch

from radio_gs.config import RadioGSConfig
from radio_gs.models.feature_quality import (
    cosine_feature_quality_target,
    visibility_target_from_alpha,
)
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.scripts.eval_lerf_grounding import apply_readout_confidence_gate
from radio_gs.scripts.train_feature_field import set_quality_visibility_heads_only_trainable


def test_cosine_feature_quality_target_maps_agreement_to_unit_interval():
    pred = torch.tensor(
        [[[[1.0, -1.0]], [[0.0, 0.0]]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[[1.0, 1.0]], [[0.0, 0.0]]]],
        dtype=torch.float32,
    )

    quality = cosine_feature_quality_target(pred, target)

    assert quality.shape == (1, 1, 1, 2)
    assert torch.allclose(quality[0, 0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_visibility_target_from_alpha_supports_soft_and_binary_targets():
    alpha = torch.tensor([[[[0.0, 0.2], [0.7, 1.0]]]], dtype=torch.float32)
    logits = torch.empty(1, 1, 1, 2)

    soft = visibility_target_from_alpha(alpha, logits)
    binary = visibility_target_from_alpha(alpha, logits, threshold=0.5, binary=True)

    assert soft.shape == (1, 1, 1, 2)
    assert binary.shape == (1, 1, 1, 2)
    assert torch.allclose(soft, torch.tensor([[[[0.35, 0.6]]]]), atol=1e-6)
    assert torch.allclose(binary, torch.tensor([[[[0.0, 1.0]]]]), atol=1e-6)


def test_feature_quality_config_defaults_are_disabled_and_explicit():
    cfg = RadioGSConfig()

    assert cfg.hybrid_quality_head is False
    assert cfg.hybrid_visibility_head is False
    assert cfg.quality_loss_weight == 0.0
    assert cfg.visibility_loss_weight == 0.0
    assert cfg.visibility_target_binary is False


def test_apply_readout_confidence_gate_uses_quality_logits_without_changing_shape():
    heatmaps = torch.ones(2, 2, 2)
    aux = {
        "quality_logit": torch.tensor([[[[0.0, 2.0], [-2.0, 0.0]]]]),
    }

    gated = apply_readout_confidence_gate(heatmaps, aux, gate="quality", gamma=1.0)

    assert gated.shape == heatmaps.shape
    expected_gate = torch.sigmoid(aux["quality_logit"][0, 0])
    assert torch.allclose(gated[0], expected_gate, atol=1e-6)
    assert torch.allclose(gated[1], expected_gate, atol=1e-6)


def test_apply_readout_confidence_gate_can_combine_quality_and_visibility():
    heatmaps = torch.ones(1, 1, 2)
    aux = {
        "quality_logit": torch.tensor([[[[0.0, 0.0]]]]),
        "visibility_logit": torch.tensor([[[[2.0, -2.0]]]]),
    }

    gated = apply_readout_confidence_gate(
        heatmaps,
        aux,
        gate="quality_visibility",
        gamma=1.0,
    )

    expected = torch.sigmoid(aux["quality_logit"][0, 0]) * torch.sigmoid(aux["visibility_logit"][0, 0])
    assert torch.allclose(gated[0], expected, atol=1e-6)


def test_set_quality_visibility_heads_only_trainable_freezes_base_field():
    model = HybridFeatureGaussian(
        latent_dim=4,
        hash_output_dim=4,
        fine_dim=4,
        coarse_dim=4,
        output_dim=8,
        num_levels=1,
        features_per_level=2,
        log2_hashmap_size=4,
        base_resolution=2,
        max_resolution=2,
        fine_hidden_dim=4,
        coarse_hidden_dim=4,
        fusion_hidden_dim=8,
        decoupled_heads=True,
        use_quality_head=True,
        use_visibility_head=True,
    )
    codec = torch.nn.Linear(8, 8)

    trainable = set_quality_visibility_heads_only_trainable(model, extra_modules=[codec])

    assert trainable > 0
    assert all(not p.requires_grad for p in codec.parameters())
    assert model._latent.requires_grad is False
    assert all(not p.requires_grad for p in model.hash_field.parameters())
    assert all(p.requires_grad for p in model.fusion_head.quality_head.parameters())
    assert all(p.requires_grad for p in model.fusion_head.visibility_head.parameters())
