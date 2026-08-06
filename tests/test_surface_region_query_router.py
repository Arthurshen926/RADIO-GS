from __future__ import annotations

import torch

from radio_gs.interfaces.surface_region_query_router import (
    SurfaceRegionQueryRouterV1,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    descriptors = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    tokens = torch.randn(2, 4, 10)
    text = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    negative = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    return descriptors, tokens, text, negative


def test_router_initializes_with_exact_fallback() -> None:
    router = SurfaceRegionQueryRouterV1(codebook_sha256="a" * 64).eval()
    descriptors, tokens, text, negative = _inputs()
    output = router(descriptors, tokens, text, negative)
    assert torch.equal(output.response, output.slot_scores[:, 0])
    assert torch.equal(output.residual_gate, torch.zeros(2, 5))
    assert torch.equal(output.slot_weights[:, 0], torch.ones(2, 5))
    assert torch.equal(output.slot_weights[:, 1:], torch.zeros(2, 3, 5))


def test_router_is_permutation_invariant_over_residual_slots() -> None:
    router = SurfaceRegionQueryRouterV1(codebook_sha256="b" * 64).eval()
    with torch.no_grad():
        router.scorer[-1].weight.normal_(std=0.1)
    descriptors, tokens, text, negative = _inputs()
    with torch.no_grad():
        router.gate[-1].bias.fill_(0.4)
    original = router(descriptors, tokens, text, negative)
    permutation = torch.tensor([0, 3, 1, 2])
    permuted = router(
        descriptors[:, permutation], tokens[:, permutation], text, negative
    )
    assert torch.allclose(original.response, permuted.response, atol=1e-6)
    assert torch.allclose(
        original.slot_weights[:, permutation], permuted.slot_weights, atol=1e-6
    )


def test_router_response_is_a_convex_slot_mixture_and_differentiable() -> None:
    router = SurfaceRegionQueryRouterV1(codebook_sha256="c" * 64)
    descriptors, tokens, text, negative = _inputs()
    output = router(descriptors, tokens, text, negative)
    assert bool((output.response >= output.slot_scores.amin(1) - 1e-6).all())
    assert bool((output.response <= output.slot_scores.amax(1) + 1e-6).all())
    output.response.square().mean().backward()
    assert router.gate[-1].weight.grad is not None
    assert float(router.gate[-1].weight.grad.abs().sum()) > 0


def test_router_checkpoint_round_trip(tmp_path) -> None:
    router = SurfaceRegionQueryRouterV1(codebook_sha256="d" * 64).eval()
    descriptors, tokens, text, negative = _inputs()
    path = tmp_path / "router.pt"
    torch.save(
        {
            "schema_version": 1,
            "architecture": router.architecture(),
            "state_dict": router.state_dict(),
        },
        path,
    )
    loaded, payload = SurfaceRegionQueryRouterV1.from_checkpoint(path)
    assert payload["schema_version"] == 1
    expected = router(descriptors, tokens, text, negative)
    actual = loaded(descriptors, tokens, text, negative)
    assert torch.equal(actual.response, expected.response)
    assert torch.equal(actual.slot_weights, expected.slot_weights)
