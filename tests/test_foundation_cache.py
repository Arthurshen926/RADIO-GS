import pytest
import torch

from radio_gs.models.foundation_cache import (
    compute_foundation_cache_supervision_loss,
    load_foundation_cache,
)
from radio_gs.scripts.train_feature_field import (
    FoundationFeatureMapProjector,
    FoundationMaskLogitProjector,
    resolve_foundation_cache_path,
)


def test_load_foundation_cache_accepts_mask_logits_and_tokens(tmp_path):
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "version": 1,
            "frame_id": 7,
            "heads": {
                "sam3": {"mask_logits": torch.randn(2, 8, 8)},
                "dino_v3": {"tokens": torch.randn(4, 16)},
                "siglip2": {"tokens": torch.randn(4, 16)},
            },
        },
        path,
    )
    cache = load_foundation_cache(path)
    assert cache.heads["sam3"].mask_logits.shape == (2, 8, 8)
    assert cache.heads["dino_v3"].tokens.shape == (4, 16)


def test_load_foundation_cache_preserves_sam3_proposal_metadata():
    payload = {
        "version": 1,
        "frame_id": 7,
        "heads": {
            "sam3": {
                "mask_logits": torch.randn(2, 8, 8),
                "queries": ["cup", "plate"],
                "scores": torch.tensor([0.9, 0.4]),
                "boxes_xyxy": torch.tensor([[1.0, 2.0, 5.0, 6.0], [0.0, 1.0, 4.0, 7.0]]),
            },
        },
    }

    cache = load_foundation_cache(payload)
    sam3 = cache.heads["sam3"]

    assert sam3.queries == ("cup", "plate")
    assert torch.allclose(sam3.scores, torch.tensor([0.9, 0.4]))
    assert sam3.boxes_xyxy.shape == (2, 4)


def test_load_foundation_cache_requires_official_producer_when_strict():
    payload = {
        "version": 1,
        "frame_id": 7,
        "heads": {
            "sam3": {
                "mask_logits": torch.randn(2, 8, 8),
            }
        },
    }

    with pytest.raises(ValueError, match="official"):
        load_foundation_cache(payload, require_official=True)

    payload["heads"]["sam3"]["producer"] = {
        "official": True,
        "backend": "facebookresearch/sam3",
        "decoder": "SAM3MaskDecoder",
    }
    cache = load_foundation_cache(payload, require_official=True)

    assert cache.heads["sam3"].producer is not None
    assert cache.heads["sam3"].producer.official is True
    assert cache.heads["sam3"].producer.backend == "facebookresearch/sam3"


def test_foundation_cache_loss_is_zero_without_cache():
    loss, stats = compute_foundation_cache_supervision_loss(
        decoded_features=torch.randn(1, 1280, 4, 4),
        cache=None,
        projectors={},
    )
    assert loss.item() == 0.0
    assert stats["enabled"] == 0


def test_load_foundation_cache_rejects_invalid_token_shape(tmp_path):
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "version": 1,
            "frame_id": "rgb_0007",
            "heads": {"dino_v3": {"tokens": torch.randn(1, 4, 16)}},
        },
        path,
    )

    with pytest.raises(ValueError, match="tokens"):
        load_foundation_cache(path)


def test_foundation_cache_loss_matches_projected_tokens():
    decoded = torch.randn(1, 3, 2, 2)
    cache_path_payload = {
        "version": 1,
        "frame_id": 0,
        "heads": {"dino_v3": {"tokens": decoded.flatten(2).transpose(1, 2).squeeze(0)}},
    }
    cache = load_foundation_cache(cache_path_payload)

    loss, stats = compute_foundation_cache_supervision_loss(
        decoded_features=decoded,
        cache=cache,
        projectors={"dino_v3": torch.nn.Identity()},
    )

    assert loss.item() < 1e-6
    assert stats["enabled"] == 1
    assert stats["token_heads"] == 1


def test_foundation_cache_boundary_loss_penalizes_blurred_mask_logits():
    target = torch.full((1, 12, 12), -6.0)
    target[:, 3:9, 3:9] = 6.0
    blurred = torch.zeros_like(target).unsqueeze(0)
    sharp = target.unsqueeze(0)
    cache = load_foundation_cache(
        {
            "version": 1,
            "frame_id": 0,
            "heads": {
                "sam3": {
                    "mask_logits": target,
                    "producer": {"official": True, "backend": "facebookresearch/sam3"},
                }
            },
        },
        require_official=True,
    )

    class FixedProjector(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, decoded):
            return self.logits.expand(decoded.shape[0], -1, -1, -1)

    decoded = torch.randn(1, 3, 12, 12)
    sharp_loss, sharp_stats = compute_foundation_cache_supervision_loss(
        decoded_features=decoded,
        cache=cache,
        projectors={"sam3": FixedProjector(sharp)},
        token_weight=0.0,
        mask_logit_weight=0.0,
        mask_boundary_weight=1.0,
    )
    blurred_loss, blurred_stats = compute_foundation_cache_supervision_loss(
        decoded_features=decoded,
        cache=cache,
        projectors={"sam3": FixedProjector(blurred)},
        token_weight=0.0,
        mask_logit_weight=0.0,
        mask_boundary_weight=1.0,
    )

    assert sharp_stats["mask_boundary_heads"] == 1
    assert blurred_stats["mask_boundary_heads"] == 1
    assert sharp_loss.item() < blurred_loss.item()


def test_sam3_region_compactness_prefers_consistent_features_without_projector():
    mask = torch.full((1, 8, 8), -8.0)
    mask[:, 2:6, 2:6] = 8.0
    cache = load_foundation_cache(
        {
            "version": 1,
            "frame_id": 0,
            "heads": {
                "sam3": {
                    "mask_logits": mask,
                    "scores": torch.tensor([0.95]),
                },
            },
        }
    )
    coherent = torch.zeros(1, 2, 8, 8)
    coherent[:, 0] = 1.0
    split = coherent.clone()
    split[:, 0, 2:6, 4:6] = -1.0

    coherent_loss, coherent_stats = compute_foundation_cache_supervision_loss(
        decoded_features=coherent,
        cache=cache,
        projectors={},
        token_weight=0.0,
        mask_logit_weight=0.0,
        region_consistency_weight=1.0,
    )
    split_loss, split_stats = compute_foundation_cache_supervision_loss(
        decoded_features=split,
        cache=cache,
        projectors={},
        token_weight=0.0,
        mask_logit_weight=0.0,
        region_consistency_weight=1.0,
    )

    assert coherent_stats["region_consistency_heads"] == 1
    assert split_stats["region_consistency_heads"] == 1
    assert coherent_loss.item() < split_loss.item()


def test_sam3_region_separation_penalizes_identical_disjoint_regions_without_projector():
    masks = torch.full((2, 8, 8), -8.0)
    masks[0, 2:6, 1:3] = 8.0
    masks[1, 2:6, 5:7] = 8.0
    cache = load_foundation_cache(
        {
            "version": 1,
            "frame_id": 0,
            "heads": {
                "sam3": {
                    "mask_logits": masks,
                    "scores": torch.tensor([0.9, 0.8]),
                },
            },
        }
    )
    identical = torch.zeros(1, 2, 8, 8)
    identical[:, 0] = 1.0
    separated = identical.clone()
    separated[:, 0, 2:6, 5:7] = -1.0

    identical_loss, identical_stats = compute_foundation_cache_supervision_loss(
        decoded_features=identical,
        cache=cache,
        projectors={},
        token_weight=0.0,
        mask_logit_weight=0.0,
        region_separation_weight=1.0,
    )
    separated_loss, separated_stats = compute_foundation_cache_supervision_loss(
        decoded_features=separated,
        cache=cache,
        projectors={},
        token_weight=0.0,
        mask_logit_weight=0.0,
        region_separation_weight=1.0,
    )

    assert identical_stats["region_separation_heads"] == 1
    assert separated_stats["region_separation_heads"] == 1
    assert separated_loss.item() < identical_loss.item()


def test_sam3_feature_boundary_loss_prefers_feature_edges_aligned_to_masks():
    mask = torch.full((1, 10, 10), -8.0)
    mask[:, 3:7, 3:7] = 8.0
    cache = load_foundation_cache(
        {
            "version": 1,
            "frame_id": 0,
            "heads": {
                "sam3": {
                    "mask_logits": mask,
                    "scores": torch.tensor([0.95]),
                },
            },
        }
    )
    smooth = torch.zeros(1, 2, 10, 10)
    smooth[:, 0] = 1.0
    aligned = smooth.clone()
    aligned[:, 0, 3:7, 3:7] = -1.0

    smooth_loss, smooth_stats = compute_foundation_cache_supervision_loss(
        decoded_features=smooth,
        cache=cache,
        projectors={},
        token_weight=0.0,
        mask_logit_weight=0.0,
        feature_boundary_weight=1.0,
    )
    aligned_loss, aligned_stats = compute_foundation_cache_supervision_loss(
        decoded_features=aligned,
        cache=cache,
        projectors={},
        token_weight=0.0,
        mask_logit_weight=0.0,
        feature_boundary_weight=1.0,
    )

    assert smooth_stats["feature_boundary_heads"] == 1
    assert aligned_stats["feature_boundary_heads"] == 1
    assert aligned_loss.item() < smooth_loss.item()


def test_foundation_mask_logit_projector_supervises_low_res_logits_from_official_cache():
    decoded = torch.randn(1, 8, 4, 5, requires_grad=True)
    cache = load_foundation_cache(
        {
            "version": 1,
            "frame_id": 0,
            "heads": {
                "sam3": {
                    "mask_logits": torch.randn(3, 16, 20),
                    "producer": {"official": True, "backend": "facebookresearch/sam3"},
                }
            },
        },
        require_official=True,
    )
    projector = FoundationMaskLogitProjector(
        input_dim=8,
        hidden_dim=4,
        output_masks=6,
    )

    logits = projector(decoded)
    loss, stats = compute_foundation_cache_supervision_loss(
        decoded_features=decoded,
        cache=cache,
        projectors={"sam3": projector},
        token_weight=0.0,
        mask_logit_weight=1.0,
        mask_boundary_weight=0.5,
    )
    loss.backward()

    assert logits.shape == (1, 6, 4, 5)
    assert stats["enabled"] == 1
    assert stats["mask_logit_heads"] == 1
    assert stats["mask_boundary_heads"] == 1
    assert decoded.grad is not None
    assert decoded.grad.abs().sum().item() > 0


def test_resolve_foundation_cache_path_supports_rgb_zero_padded(tmp_path):
    cache_path = tmp_path / "rgb_000007.pt"
    torch.save(
        {
            "version": 1,
            "frame_id": 7,
            "heads": {"dino_v3": {"tokens": torch.randn(2, 3)}},
        },
        cache_path,
    )

    assert resolve_foundation_cache_path(tmp_path, 7) == cache_path


def test_foundation_feature_map_projector_flattens_tokens():
    class ScaleProjector(torch.nn.Module):
        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            assert tokens.shape == (1, 4, 3)
            return tokens * 2.0

    decoded = torch.ones(1, 3, 2, 2)
    projected = FoundationFeatureMapProjector(ScaleProjector())(decoded)

    assert projected.shape == (1, 4, 3)
    assert torch.allclose(projected, torch.full((1, 4, 3), 2.0))
