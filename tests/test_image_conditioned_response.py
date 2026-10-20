import numpy as np
import pytest
import torch

from radio_gs.querying.image_conditioned_response import response_prompt, lift_image_response


def test_non_square_coordinates_and_log_odds():
    x = np.full((6, 10), .1, np.float32)
    x[1:5, 7:10] = .9
    p = response_prompt(x, (6, 10))
    np.testing.assert_array_equal(p['box'], [7, 1, 10, 5])
    np.testing.assert_allclose(p['mask_input'][0], np.log(x / (1 - x)), atol=1e-6)
    assert response_prompt(np.zeros_like(x), (4, 4)) is None
    assert np.isfinite(response_prompt(np.ones_like(x), (4, 4))['mask_input']).all()


def test_lift_independent_identity_system_and_unsupported_preservation():
    a = torch.sparse_coo_tensor(torch.tensor([[0, 1], [0, 1]]), torch.ones(2), (2, 3))
    b = torch.tensor([[.2], [.8], [.37]])
    y = torch.tensor([[.9], [.1]])
    v, report = lift_image_response(b, a, y, torch.ones_like(y))
    torch.testing.assert_close(v[:2], y)
    assert v[2].item() == b[2].item()
    assert report['unsupported_changed'] == 0
    assert report['source_image_loss_after'] < report['source_image_loss_before']
    unchanged, _ = lift_image_response(b, a, y, torch.zeros_like(y))
    torch.testing.assert_close(unchanged, b, rtol=0, atol=0)


def test_reject_bad_probabilities_and_unnormalized_operators():
    with pytest.raises(ValueError):
        response_prompt([[float('nan')]], (4, 4))
    with pytest.raises(ValueError):
        response_prompt([[1.1]], (4, 4))
    a = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([.5]), (1, 1))
    with pytest.raises(ValueError, match='normalized'):
        lift_image_response(torch.ones(1, 1), a, torch.ones(1, 1), torch.ones(1, 1))
