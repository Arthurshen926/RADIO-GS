import numpy as np

from radio_gs.scripts.eval_lerf_teacher_view_oracle_diagnostic import (
    _binary_midrank_correlation,
    _grouped_average_precision,
    _oracle_iou,
)


def test_grouped_ap_is_tie_invariant() -> None:
    scores = np.array([0.8, 0.8, 0.2, 0.1])
    labels = np.array([True, False, True, False])
    expected = 0.5 * 0.5 + 0.5 * (2.0 / 3.0)
    assert abs(_grouped_average_precision(scores, labels) - expected) < 1e-12
    assert abs(_grouped_average_precision(scores[[1, 0, 2, 3]], labels[[1, 0, 2, 3]]) - expected) < 1e-12


def test_oracle_iou_scans_distinct_score_levels() -> None:
    iou, threshold = _oracle_iou(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([True, True, False, False]),
    )
    assert iou == 1.0
    assert threshold == 0.8


def test_binary_midrank_correlation_has_expected_sign() -> None:
    labels = np.array([False, False, True, True])
    assert _binary_midrank_correlation(np.array([0.0, 0.1, 0.8, 0.9]), labels) > 0.8
    assert _binary_midrank_correlation(np.array([0.9, 0.8, 0.1, 0.0]), labels) < -0.8
