import numpy as np
from radio_gs.querying.response_replacement import FEATURE_NAMES
from radio_gs.querying.response_value import set_features, relative_utilities, training_values


class Value:
    def predict(self, x):
        return x[:, 0] + .3*x[:, len(FEATURE_NAMES)] + x[:, -1]**2


def test_complete_candidate_values_are_permutation_equivariant():
    x = np.random.default_rng(4).normal(size=(4, len(FEATURE_NAMES))).astype(np.float32)
    order = [2, 0, 3, 1]
    np.testing.assert_allclose(set_features(x)[order], set_features(x[order]), atol=1e-6)
    g = relative_utilities(Value(), x); h = relative_utilities(Value(), x[order])
    np.testing.assert_allclose(h, g[order]-g[order[0]], atol=1e-6)


def test_values_have_zero_self_gain_and_consistent_cycles():
    x = np.zeros((3, len(FEATURE_NAMES)), np.float32); x[:, 0] = [1, 2, 4]
    a = relative_utilities(Value(), x)
    b = relative_utilities(Value(), x[[1, 2, 0]])
    c = relative_utilities(Value(), x[[2, 0, 1]])
    assert a[0] == b[0] == c[0] == 0
    assert a[1]+b[1]+c[1] == 0


def test_query_offset_does_not_change_targets_and_groups_have_equal_weight():
    x = np.zeros((2, len(FEATURE_NAMES))).tolist()
    reports = [dict(rows=[dict(features=x, ious=[.2, .5])]),
               dict(rows=[dict(features=x, ious=[.3, .6]), dict(features=x, ious=[.1, .4])])]
    _, y, weight = training_values(reports)
    np.testing.assert_allclose(y, [-.15, .15]*3, atol=1e-7)
    assert np.isclose(weight[:2].sum(), weight[2:].sum())
