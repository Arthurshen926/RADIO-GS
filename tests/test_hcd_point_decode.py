import torch

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
