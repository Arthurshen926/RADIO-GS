from types import SimpleNamespace

import torch

from radio_gs.config import RadioGSConfig, load_config
from radio_gs.models.hcd_codec import DirectProjectionCodec, HCDCodec
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


def test_trainer_codec_factory_defaults_to_hcd_and_supports_direct() -> None:
    default_codec = RadioGSTrainer._build_codec(
        SimpleNamespace(radio_feature_dim=8, bottleneck_dim=4)
    )
    direct_codec = RadioGSTrainer._build_codec(
        SimpleNamespace(radio_feature_dim=8, bottleneck_dim=3, codec_type="direct")
    )

    assert isinstance(default_codec, HCDCodec)
    assert isinstance(direct_codec, DirectProjectionCodec)


def test_codec_type_is_loaded_from_yaml(tmp_path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("codec_type: direct\n", encoding="utf-8")

    cfg = load_config(str(cfg_path))

    assert isinstance(cfg, RadioGSConfig)
    assert cfg.codec_type == "direct"
