import pytest
import torch
from torch.nn import functional as F

from radio_gs.training.query_preserving_basis import fit_query_preserving_basis, native_response_diagnostic


def test_population_duplication_does_not_reweight_text():
    torch.manual_seed(7)
    x, q = torch.randn(50, 12), torch.randn(20, 12)
    a = fit_query_preserving_basis(x, q, 4)
    b = fit_query_preserving_basis(x.repeat(3, 1), q.repeat(2, 1), 4)
    assert torch.allclose(a.T @ a, b.T @ b, atol=2e-5)


def test_query_sensitive_low_energy_axis_survives():
    x = torch.tensor([[1., .05, 0.], [1., -.05, 0.], [1., 0., .1], [1., 0., -.1]])
    q = torch.tensor([[0., 1., 0.], [0., -1., 0.]])
    old = fit_query_preserving_basis(x, q, 2, 0.)
    new = fit_query_preserving_basis(x, q, 2, 1.)
    assert native_response_diagnostic(x, q, new)['mse'] < native_response_diagnostic(x, q, old)['mse']


def test_native_response_uses_full_query_norm():
    x, q = torch.eye(3), torch.tensor([[1., 0., 2.], [0., 1., 0.]])
    basis = torch.eye(3)[:2]
    stats = native_response_diagnostic(x, q, basis)
    exact = F.normalize(x @ basis.T @ basis, dim=-1) @ F.normalize(q, dim=-1).T
    truth = x @ F.normalize(q, dim=-1).T
    assert stats['mse'] == pytest.approx(float((exact-truth).square().mean()))


def test_invalid_vectors_and_weights_fail():
    with pytest.raises(ValueError):
        fit_query_preserving_basis(torch.zeros(4, 3), torch.ones(2, 3), 2)
    with pytest.raises(ValueError):
        fit_query_preserving_basis(torch.ones(4, 3), torch.ones(2, 3), 2, float('nan'))
