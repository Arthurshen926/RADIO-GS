import torch

from radio_gs.scripts.evaluate_source_proposal_retrieval_membership import (
    transferred_support_iou,
    unique_view_topk,
)


def test_unique_view_topk_keeps_best_proposal_per_view():
    candidates = torch.tensor([2, 3, 4, 5])
    scores = torch.tensor([0.8, 0.9, 0.7, 0.6])
    views = torch.tensor([9, 9, 0, 0, 1, 2])
    assert unique_view_topk(scores, candidates, views, 3).tolist() == [3, 4, 5]


def test_transferred_support_iou_restricts_unknown_rows_to_target_visibility():
    value = transferred_support_iou(
        candidate_supports=[torch.tensor([1, 2, 7]), torch.tensor([2, 3])],
        target_support=torch.tensor([2, 3, 8]),
        visible=torch.tensor([0, 1, 2, 3, 4]),
        num_rows=10,
    )
    # Predicted {1,2,3}, target {2,3}; row 7/8 are unknown in this view.
    assert value == 2 / 3
