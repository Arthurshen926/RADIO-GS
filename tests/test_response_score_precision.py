import cv2
import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import average_precision_score

from radio_gs.evaluation.response_score_precision import stable_probability_readout, saturation_counts


def test_saturated_sigmoid_keeps_true_ranking_in_complement_readout():
    logits = np.array([[40., 45., 50., 55.]])
    truth = np.array([[False, False, True, True]])
    probability, score = stable_probability_readout(logits, [4, 1])
    assert np.all(probability == 1)
    assert average_precision_score(truth.ravel(), probability.ravel()) == .5
    assert average_precision_score(truth.ravel(), score.ravel()) == 1
    np.testing.assert_allclose(score, logits)
    assert saturation_counts(probability, truth)['one'] == dict(pixels=4, tp=2, fp=2)


def test_readout_is_probability_interpolation_not_logit_interpolation():
    logits = np.array([[-8., 2.], [1., 4.]])
    p, score = stable_probability_readout(logits, [9, 5])
    expected = cv2.resize(expit(logits), (9, 5), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(expit(score), expected, atol=1e-15)
    assert np.max(np.abs(score-cv2.resize(logits, (9, 5)))) > 1
    np.testing.assert_array_equal(p >= .5, score >= 0)


def test_precision_diagnostic_rejects_invalid_data_and_underflow():
    for value in [np.array([1.]), np.array([[np.nan]]), np.array([[1000.]])]:
        with pytest.raises(ValueError): stable_probability_readout(value, [1, 1])
