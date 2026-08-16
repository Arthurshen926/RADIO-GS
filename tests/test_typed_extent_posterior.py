import numpy as np
import pytest

from radio_gs.querying.typed_extent_posterior import (
    PeakAnchoredExtentPolicy,
    apply_peak_anchored_extent,
)


def _multipart() -> np.ndarray:
    support = np.zeros((10, 10), dtype=bool)
    support[0:2, 0:2] = True
    support[4:8, 4:8] = True
    return support


def test_dense_and_projected_domains_share_peak_connectivity_operator() -> None:
    support = _multipart()
    dense = apply_peak_anchored_extent(
        support,
        (0, 0),
        policy=PeakAnchoredExtentPolicy("dense_raster", 0.0),
    )
    projected = apply_peak_anchored_extent(
        support,
        (0, 0),
        policy=PeakAnchoredExtentPolicy("projected_primitive_alpha", 0.25),
    )

    assert dense.candidate_accepted is True
    assert int(dense.mask.sum()) == 4
    assert projected.candidate_accepted is False
    assert np.array_equal(projected.mask, support)
    assert dense.retained_fraction == projected.retained_fraction == 0.2


def test_peak_outside_support_snaps_to_nearest_foreground() -> None:
    support = _multipart()
    result = apply_peak_anchored_extent(
        support,
        (3, 3),
        policy=PeakAnchoredExtentPolicy("dense_raster", 0.0),
    )

    expected = np.zeros_like(support)
    expected[4:8, 4:8] = True
    assert np.array_equal(result.mask, expected)


def test_receipt_declares_information_and_persistence_boundary() -> None:
    result = apply_peak_anchored_extent(
        np.ones((2, 2), dtype=bool),
        (0, 0),
        policy=PeakAnchoredExtentPolicy("dense_raster", 0.25),
    )

    receipt = result.receipt()
    assert receipt["operator"] == "peak_anchored_conservative_extent_v1"
    assert receipt["rgb_opened"] is False
    assert receipt["ground_truth_opened"] is False
    assert receipt["persistent_state_updated"] is False


def test_policy_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="minimum_retained_fraction"):
        PeakAnchoredExtentPolicy("dense_raster", 1.1)
