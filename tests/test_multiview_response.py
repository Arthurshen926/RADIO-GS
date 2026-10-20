import pytest
import torch
from radio_gs.querying.multiview_response import fit_multiview_response, normalize_view_confidence


def operator():
    return torch.sparse_coo_tensor(torch.tensor([[0, 1], [0, 1]]), torch.ones(2), (2, 3)).coalesce()


def test_source_only_columns_do_not_observe_target_and_unknown_is_preserved():
    a = operator(); b = torch.full((3, 2), .3)
    y = torch.tensor([[.6, .6], [.4, .4]])
    c = torch.ones_like(y); tc = torch.tensor([[0., 1.], [0., 1.]])
    v, r = fit_multiview_response(b, [(a, y, c)], (a, 1-y, tc))
    torch.testing.assert_close(v[:2, 0], y[:, 0])
    torch.testing.assert_close(v[:2, 1], torch.full((2,), .5), atol=1e-6, rtol=0)
    torch.testing.assert_close(v[2], b[2], rtol=0, atol=0)
    assert r['unsupported_changed'] == 0 and r['target_confidence_mass'] == [0., 2.]
    assert r['objective_after'] < r['objective_before']


def test_duplicate_source_views_do_not_overpower_target_group():
    a = operator(); b = torch.full((3, 1), .5)
    y = torch.tensor([[.6], [.4]]); c = torch.ones_like(y)
    one, _ = fit_multiview_response(b, [(a, y, c)], (a, 1-y, c))
    many, _ = fit_multiview_response(b, [(a, y, c)]*7, (a, 1-y, c))
    torch.testing.assert_close(one, many, rtol=0, atol=1e-6)


def test_zero_confidence_is_unknown_not_background():
    a = operator(); b = torch.full((3, 1), .37); zero = torch.zeros((2, 1))
    v, r = fit_multiview_response(b, [(a, zero, zero)])
    torch.testing.assert_close(v, b, rtol=0, atol=0)
    assert r['active_source_views'] == [0]
    with pytest.raises(ValueError): normalize_view_confidence(torch.tensor([[float('nan')]]))
