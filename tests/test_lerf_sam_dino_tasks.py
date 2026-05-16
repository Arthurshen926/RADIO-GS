import numpy as np
import torch

from radio_gs.scripts.eval_lerf_sam_dino_tasks import (
    bbox_from_mask,
    binary_iou,
    connected_component_from_seed,
    dense_match_points,
    feature_token_to_image_xy,
    filter_matches_by_ransac,
    keep_component_by_score,
    mask_heatmap_outside_prompt,
    mask_centroid_token,
    pooled_token_similarity,
    propagate_mask_by_dense_matches,
    scaled_bounded_area,
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


def test_dense_match_points_mutual_filter_removes_cycle_inconsistent_match():
    source = torch.zeros(2, 2, 2)
    target = torch.zeros(2, 2, 2)
    source[:, 0, 0] = torch.tensor([1.0, 0.0])
    source[:, 0, 1] = torch.tensor([0.8, 0.2])
    source[:, 1, 1] = torch.tensor([0.0, 1.0])
    target[:, 0, 0] = torch.tensor([1.0, 0.0])
    target[:, 1, 1] = torch.tensor([0.0, 1.0])
    points = [(0, 0), (0, 1), (1, 1)]

    matches = dense_match_points(
        source,
        target,
        points,
        mutual_check=True,
        cycle_max_distance=0.01,
    )

    assert [(m["src_y"], m["src_x"], m["tgt_y"], m["tgt_x"]) for m in matches] == [
        (0, 0, 0, 0),
        (1, 1, 1, 1),
    ]


def test_filter_matches_by_ransac_removes_homography_outliers():
    matches = []
    for idx, (sy, sx) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2), (1, 3), (3, 1)]):
        matches.append(
            {
                "id": idx,
                "src_y": sy,
                "src_x": sx,
                "tgt_y": sy + 4,
                "tgt_x": sx + 7,
                "score": 0.9,
            }
        )
    matches.extend(
        [
            {"id": 6, "src_y": 0, "src_x": 3, "tgt_y": 9, "tgt_x": 0, "score": 0.9},
            {"id": 7, "src_y": 3, "src_x": 0, "tgt_y": 0, "tgt_x": 9, "score": 0.9},
        ]
    )

    filtered = filter_matches_by_ransac(
        matches,
        model="homography",
        reproj_threshold=0.5,
        min_inliers=4,
    )

    assert [match["id"] for match in filtered] == list(range(6))


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


def test_pooled_token_similarity_supports_topk_mean_pooling():
    target_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ]
    )
    reference_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [0.6, 0.8],
            [0.0, 1.0],
        ]
    )

    max_scores = pooled_token_similarity(target_tokens, reference_tokens, mode="max")
    top2_scores = pooled_token_similarity(target_tokens, reference_tokens, mode="topk_mean", topk_ratio=2 / 3)

    assert torch.allclose(max_scores, torch.tensor([1.0, 0.8, 1.0]), atol=1e-5)
    assert torch.allclose(top2_scores, torch.tensor([0.8, 0.72, 0.9]), atol=1e-5)


def test_scaled_bounded_area_applies_scale_floor_and_cap():
    assert scaled_bounded_area(source_area=10, total_area=100, scale=1.5) == 15
    assert scaled_bounded_area(source_area=1, total_area=100, scale=1.0, min_area_ratio=0.05) == 5
    assert scaled_bounded_area(source_area=90, total_area=100, scale=1.0, max_area_ratio=0.2) == 20


def test_keep_component_by_score_keeps_peak_component():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[3:5, 4:6] = 1
    scores = np.zeros_like(mask, dtype=np.float32)
    scores[4, 5] = 3.0

    kept = keep_component_by_score(mask, scores, mode="peak")

    expected = np.zeros_like(mask)
    expected[3:5, 4:6] = 1
    assert np.array_equal(kept, expected)


def test_keep_component_by_score_can_keep_multiple_ranked_components():
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[2:4, 3:5] = 1
    mask[4:6, 5:7] = 1
    scores = np.zeros_like(mask, dtype=np.float32)
    scores[0:2, 0:2] = 0.1
    scores[2:4, 3:5] = 0.8
    scores[4:6, 5:7] = 0.6

    kept = keep_component_by_score(mask, scores, mode="score_sum", keep_count=2)

    expected = np.zeros_like(mask)
    expected[2:4, 3:5] = 1
    expected[4:6, 5:7] = 1
    assert np.array_equal(kept, expected)


def test_keep_component_by_score_can_anchor_to_dense_match_seed():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[3:5, 4:6] = 1
    scores = np.zeros_like(mask, dtype=np.float32)
    scores[0:2, 0:2] = 0.9
    scores[3:5, 4:6] = 0.2

    kept = keep_component_by_score(
        mask,
        scores,
        mode="match_seed",
        seed_points=[(3, 4)],
    )

    expected = np.zeros_like(mask)
    expected[3:5, 4:6] = 1
    assert np.array_equal(kept, expected)
