import numpy as np
import torch

from radio_gs.scripts.eval_lerf_sam_dino_tasks import (
    bbox_from_mask,
    dense_match_points,
    feature_token_to_image_xy,
    mask_centroid_token,
)


def test_bbox_from_mask_returns_tight_xyxy_box():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 1

    assert bbox_from_mask(mask) == (2, 1, 4, 3)


def test_mask_centroid_token_uses_nearest_foreground_feature_cell():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[4, 5] = 1

    assert mask_centroid_token(mask, feature_height=3, feature_width=3) == (2, 2)


def test_dense_match_points_retrieves_nearest_target_tokens():
    source = torch.zeros(2, 2, 2)
    target = torch.zeros(2, 2, 2)
    source[:, 0, 0] = torch.tensor([1.0, 0.0])
    source[:, 1, 1] = torch.tensor([0.0, 1.0])
    target[:, 0, 1] = torch.tensor([1.0, 0.0])
    target[:, 1, 0] = torch.tensor([0.0, 1.0])
    points = [(0, 0), (1, 1)]

    matches = dense_match_points(source, target, points)

    assert [(m["src_y"], m["src_x"], m["tgt_y"], m["tgt_x"]) for m in matches] == [
        (0, 0, 0, 1),
        (1, 1, 1, 0),
    ]


def test_feature_token_to_image_xy_maps_token_center_to_image_center():
    assert feature_token_to_image_xy(1, 1, feature_height=2, feature_width=2, image_height=100, image_width=200) == (
        150,
        75,
    )
