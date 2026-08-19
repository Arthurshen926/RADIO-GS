import torch
import torch.nn.functional as F

import radio_gs.models.siglip_projection as siglip_projection


def _torch_memory_efficient_attention(query, key, value, *, p, scale):
    assert p == 0.0
    assert scale == query.shape[-1] ** -0.5
    projected = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        dropout_p=0.0,
    )
    return projected.permute(0, 2, 1, 3)


def test_xformers_attention_forward_matches_timm_global_attention(monkeypatch) -> None:
    torch.manual_seed(4)
    projection = siglip_projection.SigLIP2FeatureProjection().eval()
    attention = projection.blocks[0].attn
    tokens = torch.randn(1, 5, 1280)
    expected = attention(tokens)
    monkeypatch.setattr(
        siglip_projection,
        "_xformers_memory_efficient_attention",
        lambda: _torch_memory_efficient_attention,
    )

    actual = siglip_projection._xformers_attention_forward(attention, tokens)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_enable_xformers_attention_binds_every_projection_block(monkeypatch) -> None:
    projection = siglip_projection.SigLIP2FeatureProjection()
    monkeypatch.setattr(
        siglip_projection,
        "_xformers_memory_efficient_attention",
        lambda: _torch_memory_efficient_attention,
    )

    returned = projection.enable_xformers_memory_efficient_attention()

    assert returned is projection
    assert projection.attention_runtime == "xformers_memory_efficient_exact_global"
    assert all(
        block.attn.forward.__func__ is siglip_projection._xformers_attention_forward
        for block in projection.blocks
    )


def test_chunked_token_mlps_match_complete_forward_and_input_gradient() -> None:
    torch.manual_seed(7)
    reference = siglip_projection.SigLIP2FeatureProjection().eval()
    candidate = siglip_projection.SigLIP2FeatureProjection().eval()
    candidate.load_state_dict(reference.state_dict())
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    candidate.enable_chunked_token_mlp(3)
    reference_input = torch.randn(1, 7, 1280, requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input)
    candidate_output = candidate(candidate_input)
    reference_output.square().mean().backward()
    candidate_output.square().mean().backward()

    torch.testing.assert_close(candidate_output, reference_output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        candidate_input.grad, reference_input.grad, atol=2e-7, rtol=2e-5
    )
