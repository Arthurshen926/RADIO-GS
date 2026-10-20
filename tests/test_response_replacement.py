import numpy as np
from scipy import sparse

from radio_gs.querying.response_replacement import (
    FEATURE_NAMES, RegionResponseEvidence, propose_regions, choose_response,
    calibrate_margin, relative_utilities,
)


def fixture():
    membership = sparse.csr_matrix(np.array([
        [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]], dtype=float))
    evidence = RegionResponseEvidence(membership, sparse.csr_matrix(np.ones((4, 3))), np.repeat(np.arange(3), 2))
    probability = np.array([[.9, .8, .1, .2], [.1, .2, .9, .8]])
    return evidence, probability, probability > .5, np.array([[3, 6], [1, 2], [4, 5], [0, 1], [5, 4], [2, 0]])


def test_teacher_order_and_duplicate_teacher_do_not_change_decisions():
    evidence, p, s, q = fixture()
    x, _ = evidence.encode(p, s, q)
    reverse, _ = evidence.encode(p, s, q[:, ::-1])
    duplicate, _ = evidence.encode(p, s, np.tile(q, (1, 2)))
    assert x.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(x, reverse)
    np.testing.assert_allclose(x, duplicate)
    args = evidence.available, evidence.frames
    assert propose_regions(q, *args) == propose_regions(q[:, ::-1], *args)


def test_common_exclusions_apply_to_every_candidate_and_no_evidence_abstains():
    evidence, p, s, q = fixture()
    x, info = evidence.encode(p, s, q, exclude_views=[0, 1, 2])
    assert info['verification_views'] == 0
    assert np.isfinite(x).all()
    assert choose_response([0, .9], 0, evidence_available=False) == 0


def test_permutation_changes_only_query_signature_not_response_geometry():
    evidence, p, s, q = fixture()
    x, _ = evidence.encode(p, s, q)
    shuffled, _ = evidence.encode(p, s, q, permuted=True)
    np.testing.assert_array_equal(x[:, :9], shuffled[:, :9])
    assert not np.array_equal(x[:, 9:], shuffled[:, 9:])
    np.testing.assert_array_equal(s, p > .5)


def test_calibration_rejects_mean_gain_that_harms_one_calibration_room():
    episodes = [dict(scene='a', gains=[0, .2], ious=[.1, .9], evidence_available=True),
                dict(scene='b', gains=[0, .3], ious=[.9, .5], evidence_available=True)]
    result, _ = calibrate_margin(episodes)
    assert result['margin'] is None
    assert result['delta'] == 0
    assert choose_response([0, 100.], result['margin']) == 0


def test_calibration_uses_one_global_margin_and_keeps_strict_ties():
    episodes = [dict(scene='a', gains=[0, .2], ious=[.5, .4], evidence_available=True),
                dict(scene='a', gains=[0, .6], ious=[.1, .6], evidence_available=True),
                dict(scene='b', gains=[0, .5], ious=[.2, .5], evidence_available=True)]
    result, _ = calibrate_margin(episodes)
    assert result['margin'] == .2
    assert choose_response([0, .2], .2) == 0
    assert choose_response([0, .5], .2) == 1


def test_antisymmetry_and_identical_answer_gain():
    class DeliberatelyBiasedRegressor:
        def predict(self, x):
            return 5 + 2*x[:, len(FEATURE_NAMES)]-x[:, 0]
    features = np.zeros((3, len(FEATURE_NAMES))); features[1, 0] = .8
    gains = relative_utilities(DeliberatelyBiasedRegressor(), features)
    assert gains[0] == 0 and gains[2] == 0
    reverse = relative_utilities(DeliberatelyBiasedRegressor(), features[[1, 0]])
    assert gains[1] == -reverse[1]
