import torch

from radio_gs.v4.contracts.build_sam3_video_pair_association import (
    _candidate_pairs,
    _pair_iou,
)


def test_candidate_pairs_prefer_small_temporal_gaps():
    assert _candidate_pairs([100, 20, 60, 70], 2) == [(60, 70), (70, 100)]


def test_pair_iou_matches_each_binary_mask_independently():
    first = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    second = torch.tensor([[[1, 0], [0, 0]], [[1, 1], [0, 0]]])
    value = _pair_iou(first, second)
    assert torch.allclose(value, torch.tensor([[1.0, 0.5], [0.0, 0.0]]))
