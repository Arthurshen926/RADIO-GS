import numpy as np
import torch

from radio_gs.scripts.evaluate_sam3_video_track_source_consistency import (
    binary_iou,
    pooled_dino_descriptor,
    select_seed_proposals,
)


def test_seed_selection_rejects_duplicate_masks():
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, :2, :2] = True
    masks[2, 2:, 2:] = True
    selected = select_seed_proposals(
        masks,
        torch.tensor([0.4, 0.3, 0.2]),
        torch.tensor([0.8, 0.9, 0.7]),
        maximum=3,
        maximum_pair_iou=0.5,
    )
    assert selected == [0, 2]


def test_pooled_dino_descriptor_is_unit_normalized():
    feature = torch.zeros(2, 2, 2)
    feature[0, 0, 0] = 2
    feature[1, 0, 0] = 2
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    value = pooled_dino_descriptor(feature, mask)
    torch.testing.assert_close(value, torch.tensor([2**-0.5, 2**-0.5]))
    assert binary_iou(mask, mask) == 1.0
