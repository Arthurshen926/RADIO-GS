import torch

from radio_gs.models.sam3_decoder_bridge import Sam3BackboneBridge, sam3_backbone_bridge_loss


def test_sam3_backbone_bridge_matches_official_shapes():
    bridge = Sam3BackboneBridge(input_dim=8, hidden_dim=32)
    out = bridge(torch.randn(1, 8, 5, 7))

    assert out["vision_features"].shape == (1, 256, 72, 72)
    assert [tuple(x.shape) for x in out["backbone_fpn"]] == [
        (1, 256, 288, 288),
        (1, 256, 144, 144),
        (1, 256, 72, 72),
    ]


def test_sam3_backbone_bridge_loss_is_finite():
    bridge = Sam3BackboneBridge(input_dim=8, hidden_dim=32)
    pred = bridge(torch.randn(1, 8, 5, 7))
    target = {
        "vision_features": torch.randn(1, 256, 72, 72),
        "backbone_fpn": [
            torch.randn(1, 256, 288, 288),
            torch.randn(1, 256, 144, 144),
            torch.randn(1, 256, 72, 72),
        ],
    }

    loss, stats = sam3_backbone_bridge_loss(pred, target)

    assert torch.isfinite(loss)
    assert stats["total"] > 0
    assert "fpn0_mse" in stats

