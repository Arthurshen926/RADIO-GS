import pytest
import torch
from torch.nn import functional as F
from radio_gs.models.bounded_native_residual import bounded_native_cosine
from radio_gs.models.native_decoder_readout import affine_decoder_cosine
from radio_gs.models.element_retrieval_field import ElementRetrievalField


@pytest.mark.parametrize('strength', [0., .1, .5])
def test_exact_native_space_cosine_and_subsets(strength):
    torch.manual_seed(3)
    z, q, w, b, r, c = [x.double().requires_grad_() for x in
        (torch.randn(17, 6), torch.randn(7, 11), torch.randn(11, 3), torch.randn(11), torch.randn(11, 6), torch.randn(11))]
    actual = bounded_native_cosine(z, q, w, b, r, c, strength)
    base = F.linear(z[:, :3], w, b)
    delta = F.linear(z, r, c)
    gate = (strength * base.norm(dim=-1) / delta.norm(dim=-1).clamp_min(1e-12)).clamp_max(1)
    expected = F.normalize(base + gate[:, None]*delta, dim=-1) @ F.normalize(q, dim=-1).T
    assert torch.allclose(actual, expected, atol=1e-10)
    assert torch.allclose(actual[::2, ::2], bounded_native_cosine(z[::2], q[::2], w, b, r, c, strength), atol=1e-10)
    if strength:
        g1 = torch.autograd.grad(actual.sum(), (z,q,w,b,r,c), retain_graph=True)
        g2 = torch.autograd.grad(expected.sum(), (z,q,w,b,r,c))
        assert all(torch.allclose(a,b,atol=1e-9) for a,b in zip(g1,g2))
    else:
        assert torch.equal(actual, affine_decoder_cosine(z[:, :3], q, w, b))


def test_zero_anchor_and_zero_initialized_residual_have_finite_gradients():
    z = torch.randn(9, 4, requires_grad=True)
    w = torch.zeros(5, 2)
    r = torch.zeros(5, 4, requires_grad=True)
    b = torch.zeros(5)
    c = torch.zeros(5, requires_grad=True)
    value = bounded_native_cosine(z, torch.randn(3, 5), w, b, r, c, .1)
    assert torch.equal(value, torch.zeros_like(value))
    value.sum().backward()
    assert all(torch.isfinite(v.grad).all() for v in (z,r,c))


def test_plain_text_image_path_uses_same_native_residual_and_cold_load():
    args = dict(num_elements=8, latent_dim=6, query_dims={'text':9,'image':9,'prompt':4},
                shared_query_groups=[['text','image']], native_siglip_readout=True, native_semantic_dim=3,
                native_semantic_residual={'relative_bound':.2})
    field = ElementRetrievalField(**args)
    with torch.no_grad():
        field.native_residual_decoder.weight.normal_()
    q = torch.randn(3,9)
    expected = 10*field.native_cosine(q)-5
    for modality in ('text','image'):
        assert torch.allclose(field.logits(q,modality),expected)
        assert torch.allclose(field.logits(q,modality,torch.tensor([2,5])),expected[[2,5]])
    restored = ElementRetrievalField(**args)
    restored.load_state_dict(field.state_dict())
    assert torch.equal(restored.native_cosine(q),field.native_cosine(q))


def test_image_supervised_residual_does_not_touch_text_or_prompt():
    args=dict(num_elements=8,latent_dim=6,query_dims={'text':9,'image':9,'prompt':4},
              shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=3)
    parent=ElementRetrievalField(**args)
    field=ElementRetrievalField(**args,native_semantic_residual={'relative_bound':.25,'modalities':['image']})
    field.load_state_dict(parent.state_dict(),strict=False)
    with torch.no_grad():field.native_residual_decoder.weight.normal_()
    q=torch.randn(3,9);p=torch.randn(2,4)
    assert torch.equal(parent.logits(q,'text'),field.logits(q,'text'))
    assert torch.equal(parent.logits(p,'prompt'),field.logits(p,'prompt'))
    assert not torch.allclose(parent.logits(q,'image'),field.logits(q,'image'))
    assert torch.allclose(field.logits(q,'image'),10*field.native_cosine(q,modality='image')-5)
