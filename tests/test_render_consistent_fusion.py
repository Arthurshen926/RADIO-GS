import pytest
import torch
from radio_gs.training.render_consistent_fusion import fit_render_consistent_features


def test_matches_dense_regularized_solution_and_preserves_unseen():
    torch.manual_seed(9)
    a = torch.rand(15, 7, dtype=torch.float64)
    a[:, -1] = 0
    a /= a.sum(1, keepdim=True)
    y = torch.randn(15, 4, dtype=torch.float64)
    x = torch.randn(7, 4, dtype=torch.float64)
    out, report = fit_render_consistent_features(a.to_sparse(), y, x, tolerance=1e-10)
    d = a[:, :-1].square().sum(0) * .1
    expected = torch.linalg.solve(a[:, :-1].T @ a[:, :-1] + torch.diag(d),
                                  a[:, :-1].T @ y + d[:, None] * x[:-1])
    torch.testing.assert_close(out[:-1], expected)
    assert torch.equal(out[-1], x[-1])
    assert report['source_mse_after'] <= report['source_mse_before']


def test_exact_fit_and_zero_coordinate_stable():
    a = torch.tensor([[1., 0.], [.5, .5], [0., 1.]])
    x = torch.tensor([[1., 0.], [2., 0.]])
    out, _ = fit_render_consistent_features(a.to_sparse(), a @ x, x)
    assert torch.equal(out, x)


def test_bad_evidence_rejected():
    a = torch.tensor([[-1., 1.]]).to_sparse()
    with pytest.raises(ValueError):
        fit_render_consistent_features(a, torch.zeros(1, 2), torch.zeros(2, 2))
