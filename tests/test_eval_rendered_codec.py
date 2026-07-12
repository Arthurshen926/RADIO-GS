from types import SimpleNamespace

from radio_gs.models.hcd_codec import DirectProjectionCodec, HCDCodec, IdentityFeatureCodec
from radio_gs.scripts.eval_rendered import _build_codec_from_config


def test_eval_rendered_builds_direct_codec_from_config():
    config = SimpleNamespace(
        radio_feature_dim=512,
        bottleneck_dim=192,
        codec_type="direct",
        dual_stream=False,
        symmetric_decoder=False,
    )

    codec = _build_codec_from_config(config)

    assert isinstance(codec, DirectProjectionCodec)


def test_eval_rendered_builds_identity_codec_from_config():
    config = SimpleNamespace(
        radio_feature_dim=512,
        bottleneck_dim=512,
        codec_type="identity",
        dual_stream=False,
        symmetric_decoder=False,
    )

    codec = _build_codec_from_config(config)

    assert isinstance(codec, IdentityFeatureCodec)


def test_eval_rendered_builds_hcd_codec_by_default():
    config = SimpleNamespace(
        radio_feature_dim=512,
        bottleneck_dim=192,
        dual_stream=True,
        symmetric_decoder=True,
    )

    codec = _build_codec_from_config(config)

    assert isinstance(codec, HCDCodec)
