import numpy as np

from radio_gs.models.prompt_conditioned_mask_refinement import (
    choose_mask_candidate_by_initial_overlap,
)


def test_choose_mask_candidate_rejects_refined_mask_that_is_too_small():
    initial = np.zeros((6, 6), dtype=bool)
    initial[1:5, 1:5] = True
    candidate = np.zeros((6, 6), dtype=bool)
    candidate[2:4, 2:4] = True

    refined, report = choose_mask_candidate_by_initial_overlap(
        initial,
        candidate[None],
        min_initial_iou=0.01,
        min_refined_area_ratio=0.5,
    )

    assert np.array_equal(refined, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "refined_mask_too_small"


def test_choose_mask_candidate_accepts_boundary_snap_with_bounded_area():
    initial = np.zeros((6, 6), dtype=bool)
    initial[1:5, 1:5] = True
    candidate = np.zeros((6, 6), dtype=bool)
    candidate[1:5, 2:5] = True

    refined, report = choose_mask_candidate_by_initial_overlap(
        initial,
        candidate[None],
        min_initial_iou=0.5,
        min_refined_area_ratio=0.5,
        max_refined_area_ratio=1.2,
    )

    assert np.array_equal(refined, candidate)
    assert report["accepted"] is True
    assert report["fallback_reason"] == "accepted"


def test_choose_mask_candidate_can_clip_to_initial_support_band():
    initial = np.zeros((8, 8), dtype=bool)
    initial[2:5, 2:5] = True
    candidate = initial.copy()
    candidate[6:8, 6:8] = True

    refined, report = choose_mask_candidate_by_initial_overlap(
        initial,
        candidate[None],
        min_initial_iou=0.5,
        min_refined_area_ratio=0.5,
        max_refined_area_ratio=1.2,
        support_dilate=1,
    )

    assert report["accepted"] is True
    assert refined[6:8, 6:8].sum() == 0
    assert np.logical_and(refined, initial).sum() == initial.sum()
