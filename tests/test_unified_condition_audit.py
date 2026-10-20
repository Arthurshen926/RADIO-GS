import numpy as np
from scipy.sparse import csr_matrix

from radio_gs.query_field.unified_condition_audit import support_evidence


def test_invisible_and_unlabelled_support_are_not_negatives():
    matrix = csr_matrix(np.array([[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.]]))
    result = support_evidence(matrix, [0], [1], [0, 1, 2, 3], [.1, .2, .3, .4])
    assert np.isclose(result["positive_mass"], .1)
    assert np.isclose(result["negative_mass"], .2)
    assert np.isclose(result["unknown_mass"], .7)
    assert np.isclose(result["invisible_mass"], .4)
    assert not result["negative_majority"]


def test_only_a_majority_of_known_negative_mass_certifies_mismatch():
    matrix = csr_matrix(np.array([[.8, 0.], [.2, 1.]]))
    result = support_evidence(matrix, [0], [1], [0, 1], [1., 3.])
    assert np.isclose(result["positive_mass"], .2)
    assert np.isclose(result["negative_mass"], .8)
    assert result["negative_majority"]
    assert np.isclose(result["unknown_mass"], 0.)
