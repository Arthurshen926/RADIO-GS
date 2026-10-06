import torch
from torch.nn import functional as F
from radio_gs.training.native_semantic_residual import fit_native_residual, native_residual_cosine


def test_base_coordinates_preserved_and_residual_is_orthogonal():
    torch.manual_seed(31)
    x, q = torch.randn(70, 12), torch.randn(20, 12)
    base = torch.linalg.qr(torch.randn(12, 4)).Q.T
    copy = base.clone()
    residual = fit_native_residual(x, q, base, 5, 1.)
    assert torch.equal(base, copy)
    assert (base @ residual.T).abs().max() < 1e-5
    assert torch.allclose(residual @ residual.T, torch.eye(5), atol=1e-5)
    a, b = x @ base.T, x @ residual.T
    result = native_residual_cosine(a, b, q, base, residual)
    exact = F.normalize(a @ base + b @ residual, dim=-1) @ F.normalize(q, dim=-1).T
    assert torch.allclose(result, exact, atol=1e-6)


def test_full_complement_recovers_uncompressed_cosine():
    torch.manual_seed(32)
    x, q = torch.randn(70, 12), torch.randn(20, 12)
    base = torch.linalg.qr(torch.randn(12, 4)).Q.T
    residual = fit_native_residual(x, q, base, 8)
    result = native_residual_cosine(x @ base.T, x @ residual.T, q, base, residual)
    assert torch.allclose(result, F.normalize(x, dim=-1) @ F.normalize(q, dim=-1).T, atol=1e-6)
