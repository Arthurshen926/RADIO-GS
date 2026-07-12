from types import SimpleNamespace

import torch

from radio_gs.config import RadioGSConfig, load_config
from radio_gs.models.hcd_codec import DirectProjectionCodec, HCDCodec, IdentityFeatureCodec
from radio_gs.scripts.train_feature_field import RadioGSTrainer


def test_direct_projection_codec_round_trips_expected_shapes() -> None:
    codec = DirectProjectionCodec(input_dim=8, bottleneck_dim=3)

    features = torch.randn(2, 8, 4, 5)
    compact = codec.encode(features)
    decoded = codec.decode(compact)

    assert compact.shape == (2, 3, 4, 5)
    assert decoded.shape == (2, 8, 4, 5)


def test_direct_projection_codec_point_decode_is_chunk_invariant() -> None:
    codec = DirectProjectionCodec(input_dim=8, bottleneck_dim=3)
    compact = torch.randn(7, 3)

    decoded_all = codec.decode_points(compact)
    decoded_chunks = torch.cat(
        [codec.decode_points(compact[:2]), codec.decode_points(compact[2:])],
        dim=0,
    )

    assert torch.allclose(decoded_all, decoded_chunks, atol=1e-6)


def test_identity_codec_preserves_image_and_point_features() -> None:
    codec = IdentityFeatureCodec(input_dim=8, bottleneck_dim=8)

    features = torch.randn(2, 8, 4, 5)
    points = torch.randn(7, 8)

    assert torch.allclose(codec.encode(features), features)
    assert torch.allclose(codec.decode(features), features)
    assert torch.allclose(codec(features), features)
    assert torch.allclose(codec.decode_points(points), points)
    assert codec.compression_ratio == 1.0


def test_identity_codec_rejects_dimension_mismatch() -> None:
    try:
        IdentityFeatureCodec(input_dim=8, bottleneck_dim=4)
    except ValueError as exc:
        assert "requires bottleneck_dim == input_dim" in str(exc)
    else:
        raise AssertionError("IdentityFeatureCodec should reject dimensional compression")


def test_trainer_codec_factory_defaults_to_hcd_and_supports_direct_and_identity() -> None:
    default_codec = RadioGSTrainer._build_codec(
        SimpleNamespace(radio_feature_dim=8, bottleneck_dim=4)
    )
    direct_codec = RadioGSTrainer._build_codec(
        SimpleNamespace(radio_feature_dim=8, bottleneck_dim=3, codec_type="direct")
    )
    identity_codec = RadioGSTrainer._build_codec(
        SimpleNamespace(radio_feature_dim=8, bottleneck_dim=8, codec_type="identity")
    )

    assert isinstance(default_codec, HCDCodec)
    assert isinstance(direct_codec, DirectProjectionCodec)
    assert isinstance(identity_codec, IdentityFeatureCodec)


def test_codec_type_is_loaded_from_yaml(tmp_path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("codec_type: direct\n", encoding="utf-8")

    cfg = load_config(str(cfg_path))

    assert isinstance(cfg, RadioGSConfig)
    assert cfg.codec_type == "direct"


def test_samclip_mask_losses_default_to_disabled() -> None:
    cfg = RadioGSConfig()

    assert cfg.samclip_mask_loss_weight == 0.0
    assert cfg.samclip_contrastive_loss_weight == 0.0
    assert cfg.samclip_background_loss_weight == 0.0
