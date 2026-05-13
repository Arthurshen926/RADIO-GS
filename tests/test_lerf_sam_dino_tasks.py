import numpy as np
import torch

from radio_gs.scripts.eval_lerf_sam_dino_tasks import (
    bbox_from_mask,
    binary_iou,
    connected_component_from_seed,
    dense_match_points,
    feature_token_to_image_xy,
    mask_heatmap_outside_prompt,
    mask_centroid_token,
    propagate_mask_by_dense_matches,
    topk_mask_from_scores,
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


def test_topk_mask_from_scores_keeps_requested_highest_tokens():
    scores = torch.tensor([[0.1, 0.8, 0.2], [0.7, 0.3, 0.6]])

    mask = topk_mask_from_scores(scores, k=3)

    assert np.array_equal(mask, np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8))


def test_mask_heatmap_outside_prompt_suppresses_invalid_peak():
    heatmap = torch.tensor([[0.2, 0.9], [0.4, 0.1]])
    prompt = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    constrained = mask_heatmap_outside_prompt(heatmap, prompt)

    assert int(constrained.reshape(-1).argmax().item()) == 2
    assert constrained[0, 1] < constrained[prompt.astype(bool)].min()


def test_connected_component_from_seed_removes_disconnected_regions():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[3:5, 4:6] = 1

    component = connected_component_from_seed(mask, seed_y=1, seed_x=1)

    expected = np.zeros_like(mask)
    expected[1:3, 1:3] = 1
    assert np.array_equal(component, expected)


def test_binary_iou_handles_empty_union():
    assert binary_iou(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)) == 0.0
    assert binary_iou(np.array([[1, 0]], dtype=np.uint8), np.array([[1, 1]], dtype=np.uint8)) == 0.5


def test_propagate_mask_by_dense_matches_transfers_source_mask_shape():
    source = torch.zeros(2, 2, 2)
    target = torch.zeros(2, 2, 2)
    source[:, 0, 0] = torch.tensor([1.0, 0.0])
    source[:, 0, 1] = torch.tensor([1.0, 0.0])
    source[:, 1, 0] = torch.tensor([0.0, 1.0])
    source[:, 1, 1] = torch.tensor([0.0, 1.0])
    target[:, 1, 0] = torch.tensor([1.0, 0.0])
    target[:, 1, 1] = torch.tensor([1.0, 0.0])
    target[:, 0, 0] = torch.tensor([0.0, 1.0])
    target[:, 0, 1] = torch.tensor([0.0, 1.0])
    source_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    propagated, score_map = propagate_mask_by_dense_matches(source, target, source_mask)

    assert np.array_equal(propagated, np.array([[0, 0], [1, 1]], dtype=np.uint8))
    assert score_map[1, 0] > score_map[0, 0]


def test_propagate_mask_by_dense_matches_can_contrast_source_background():
    source = torch.zeros(2, 2, 2)
    target = torch.zeros(2, 2, 2)
    source[:, 0, 0] = torch.tensor([1.0, 0.0])
    source[:, 0, 1] = torch.tensor([1.0, 0.0])
    source[:, 1, 0] = torch.tensor([0.0, 1.0])
    source[:, 1, 1] = torch.tensor([0.0, 1.0])
    target[:, 0, 0] = torch.tensor([0.0, 1.0])
    target[:, 0, 1] = torch.tensor([0.0, 1.0])
    target[:, 1, 0] = torch.tensor([1.0, 0.0])
    target[:, 1, 1] = torch.tensor([0.5, 0.5])
    source_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    _, plain_scores = propagate_mask_by_dense_matches(source, target, source_mask)
    _, contrast_scores = propagate_mask_by_dense_matches(
        source,
        target,
        source_mask,
        background_contrast=1.0,
    )

    assert plain_scores[0, 0] == 0.0
    assert contrast_scores[0, 0] < plain_scores[0, 0]
    assert contrast_scores[1, 0] > contrast_scores[0, 0]
