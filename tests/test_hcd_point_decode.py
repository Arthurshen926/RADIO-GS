import torch
import pytest

from radio_gs.models.hcd_codec import HCDCodec


def test_decode_points_is_independent_of_eval_chunking():
    torch.manual_seed(7)
    codec = HCDCodec(
        input_dim=8,
        bottleneck_dim=4,
        dual_stream=True,
        symmetric_decoder=False,
    ).eval()
    compact = torch.randn(5, 4)

    decoded_all = codec.decode_points(compact)
    decoded_chunks = torch.cat(
        [codec.decode_points(compact[:2]), codec.decode_points(compact[2:])],
        dim=0,
    )

    assert decoded_all.shape == (5, 8)
    assert torch.allclose(decoded_all, decoded_chunks, atol=1e-6)


def _decode_map_and_tokens(codec: HCDCodec, compact: torch.Tensor):
    decoded_map = codec.decode(compact)
    batch, channels, height, width = compact.shape
    decoded_tokens = codec.decode_points(
        compact.permute(0, 2, 3, 1).reshape(-1, channels)
    )
    decoded_tokens = decoded_tokens.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
    return decoded_map, decoded_tokens


@pytest.mark.parametrize("normalization", ["token_layer", "token_rms", "none"])
def test_tokenwise_decoder_map_and_point_paths_are_identical(normalization):
    torch.manual_seed(7)
    codec = HCDCodec(
        input_dim=24,
        bottleneck_dim=8,
        dual_stream=True,
        hidden_normalization=normalization,
        final_normalization=normalization,
    ).eval()
    compact = torch.randn(2, 8, 5, 7)
    decoded_map, decoded_tokens = _decode_map_and_tokens(codec, compact)
    assert torch.allclose(decoded_map, decoded_tokens, atol=1e-5, rtol=1e-5)


def test_legacy_group_decoder_exposes_map_point_mismatch():
    torch.manual_seed(7)
    codec = HCDCodec(
        input_dim=24,
        bottleneck_dim=8,
        dual_stream=True,
        hidden_normalization="legacy_group",
        final_normalization="legacy_group",
    ).eval()
    compact = torch.randn(1, 8, 5, 7)
    decoded_map, decoded_tokens = _decode_map_and_tokens(codec, compact)
    assert float((decoded_map - decoded_tokens).abs().max()) > 1e-3
