import numpy as np
import pytest
from radio_gs.querying.transport_residual import transfer_reference_delta


@pytest.mark.parametrize('space', ['probability', 'log_odds'])
def test_zero_intervention_preserves_all_incumbent_scores_exactly(space):
    b = np.array([[0, .21, 1], [.64, .5, .99]], np.float32)
    p = np.array([[1, .9, 0], [.1, .1, .1]], np.float32)
    out, receipt = transfer_reference_delta(b, p, p, space)
    assert np.array_equal(out, b) and receipt['updated_pixels'] == 0
    assert receipt['unaltered_teacher_exact_noop']


def test_paired_change_recovers_and_rejects_without_replacing_unchanged_teacher():
    b = np.array([[.2, .8, .7]], np.float32)
    old = np.array([[.1, .9, .1]], np.float32)
    new = np.array([[.8, .2, .1]], np.float32)
    out, _ = transfer_reference_delta(b, old, new)
    assert np.allclose(out, [[.9, .1, .7]])
    with pytest.raises(ValueError): transfer_reference_delta(b, old[:, :1], new)
