import torch

from radio_gs.v3.evaluation.analyze_lerf_posterior_failure import (
    _pairwise_jaccard,
    _posterior_correlation,
    _projection_diagnostics,
    _top_fraction_mask,
)


def test_pairwise_jaccard_detects_shared_query_extent():
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
            [False, False, True],
        ]
    )
    summary = _pairwise_jaccard(mask)
    assert summary["max"] == 1.0
    assert summary["median"] == 0.0


def test_posterior_correlation_detects_collapsed_query_columns():
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.9],
            [0.2, 0.4, 0.7],
            [0.3, 0.6, 0.5],
            [0.4, 0.8, 0.3],
        ]
    )
    summary = _posterior_correlation(scores)
    assert summary["max"] > 0.999


def test_top_fraction_mask_has_fixed_per_query_budget():
    scores = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    mask = _top_fraction_mask(scores, 0.2)
    assert mask.sum(dim=0).tolist() == [2, 2, 2]


def test_projection_diagnostics_separates_recall_from_overselection():
    result = {
        "miou": 0.1,
        "n": 2,
        "query_details": [
            {
                "intersection_pixels": 90,
                "pred_pixels": 900,
                "gt_pixels": 100,
                "overselect_ratio": 9.0,
            },
            {
                "intersection_pixels": 45,
                "pred_pixels": 450,
                "gt_pixels": 50,
                "overselect_ratio": 9.0,
            },
        ],
    }
    diagnostic = _projection_diagnostics(result)
    assert diagnostic["pixel_precision"]["mean"] == 0.1
    assert diagnostic["pixel_recall"]["mean"] == 0.9
    assert diagnostic["predicted_to_gt_area_ratio"]["median"] == 9.0
