from __future__ import annotations

import numpy as np
import pytest

from radio_gs.querying.source_oof_transport_admission import (
    DirectionalAdmissionCalibration,
)
from radio_gs.scripts.eval_spin_rendered_directional_admission import _iou
from radio_gs.scripts.materialize_spin_rendered_directional_admission import (
    build_candidate_frame,
)


def _calibration() -> DirectionalAdmissionCalibration:
    return DirectionalAdmissionCalibration(
        expansion=0.0,
        contraction=1.0,
        leave_one_fold_expansion=(0.0, 0.0, 0.0),
        leave_one_fold_contraction=(1.0, 1.0, 1.0),
        folds=(0, 1, 2),
        eligible_rows=8,
    )


def test_rendered_admission_locks_full_coverage_and_rejects_expansion() -> None:
    unary = np.full((2, 2), 0.5, dtype=np.float32)
    proposal = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    coverage = np.array([[0.0, 0.0], [1.0, 0.5]], dtype=np.float32)
    candidate = build_candidate_frame(unary, proposal, coverage, _calibration())
    assert candidate[0, 0] == unary[0, 0]
    assert candidate[0, 1] == pytest.approx(proposal[0, 1])
    assert candidate[1, 0] == unary[1, 0]
    assert candidate[1, 1] == unary[1, 1]


def test_exact_iou_empty_union_and_regular_case() -> None:
    assert _iou(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)) == 1.0
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    assert _iou(prediction, target) == 1.0 / 3.0
