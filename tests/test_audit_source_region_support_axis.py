import torch

from radio_gs.scripts.audit_source_region_support_axis import (
    _score_global_rows,
    _source_targets,
    _summarize,
)


def test_source_targets_excludes_background_and_keeps_positive_instances():
    mass = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.5, 2.5],
        ]
    )
    target, observed, ids = _source_targets(mass)
    assert ids == [1, 2]
    assert target.tolist() == [[False, False], [True, False], [False, True]]
    assert observed.tolist() == [[False, False], [True, True], [True, True]]


def test_score_global_rows_is_exact_primitive_iou():
    target = torch.tensor([[True], [True], [False], [False]])
    observed = torch.ones_like(target)
    rows = torch.tensor([[0, 1, -1], [0, 2, -1], [2, 3, -1]])
    score = _score_global_rows(
        rows, target=target, observed=observed, batch_size=2
    )
    assert torch.allclose(score[:, 0], torch.tensor([1.0, 1.0 / 3.0, 0.0]))


def test_summarize_reports_macro_and_selection_counts():
    rows = [
        {
            "D2": {"primitive_iou": 0.25},
            "D3": {"primitive_iou": 0.75, "regions": 2, "selected_rows": 5},
        },
        {
            "D2": {"primitive_iou": 0.5},
            "D3": {"primitive_iou": 1.0, "regions": 1, "selected_rows": 3},
        },
    ]
    result = _summarize(rows)
    assert result["best_single_region_macro_primitive_iou"] == 0.375
    assert result["up_to_eight_region_macro_primitive_iou"] == 0.875
    assert result["mean_selected_regions"] == 1.5
    assert result["mean_selected_primitives"] == 4.0
