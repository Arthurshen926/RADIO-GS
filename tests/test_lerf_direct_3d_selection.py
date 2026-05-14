import pytest
import numpy as np
import torch

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    apply_registration_confidence,
    apply_selection_ratio_bounds,
    aggregate_scores_by_voxel,
    bootstrap_mean_ci,
    choose_registration_refiner,
    load_score_cache,
    merge_registered_scores,
    refine_mask_with_rgb_edges,
    sample_registration_view_weights,
    save_score_cache,
    refine_selection_by_voxel_components,
    score_text_aligned_embeddings,
    select_gaussians_with_seed_expand_components,
    select_registration_frame_ids,
)


def test_select_registration_frame_ids_uses_official_available_frames():
    frames = select_registration_frame_ids(
        available_pose_ids=[1, 2, 3, 4, 5],
        annotated_frame_ids=[2, 3, 4],
        official_frame_ids=[3, 5, 9],
        mode="official",
        max_frames=0,
    )

    assert frames == [3]


def test_select_registration_frame_ids_evenly_subsamples_all_poses():
    frames = select_registration_frame_ids(
        available_pose_ids=[0, 2, 4, 6, 8],
        annotated_frame_ids=[2, 4],
        official_frame_ids=[],
        mode="all_poses",
        max_frames=3,
    )

    assert frames == [0, 4, 8]


def test_select_registration_frame_ids_supports_train_pose_subset():
    frames = select_registration_frame_ids(
        available_pose_ids=[0, 1, 2, 3, 4, 5],
        annotated_frame_ids=[1, 3],
        official_frame_ids=[],
        train_frame_ids=[0, 2, 4, 5],
        val_frame_ids=[1, 3],
        mode="train",
        max_frames=2,
    )

    assert frames == [0, 5]


def test_choose_registration_refiner_can_disable_vfa_for_ablation():
    refiner = object()

    assert choose_registration_refiner(refiner, disable_registered_refiner=False) is refiner
    assert choose_registration_refiner(refiner, disable_registered_refiner=True) is None


def test_score_text_aligned_embeddings_supports_canonical_relevancy():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = torch.tensor([[1.0, 0.0]])
    canonical = torch.tensor([[0.0, 1.0]])

    scores = score_text_aligned_embeddings(
        embeddings,
        text,
        canonical_embeddings=canonical,
        scoring="relevancy",
        softmax_temperature=10.0,
    )

    assert scores.shape == (2, 1)
    assert scores[0, 0] > 0.99
    assert scores[1, 0] < 0.01


def test_merge_registered_scores_uses_fallback_for_unregistered_gaussians():
    registered = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    valid = torch.tensor([True, False])
    text = torch.tensor([[1.0, 0.0]])
    fallback = torch.tensor([[0.2], [0.7]])

    scores = merge_registered_scores(
        registered,
        valid,
        text,
        fallback_scores=fallback,
        scoring="cosine",
    )

    assert torch.allclose(scores, torch.tensor([[1.0], [0.7]]), atol=1e-6)


def test_apply_registration_confidence_blend_zero_preserves_scores():
    scores = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
    counts = torch.tensor([0.0, 10.0], dtype=torch.float32)

    calibrated = apply_registration_confidence(scores, counts, blend=0.0, mode="log")

    assert calibrated is scores


def test_apply_registration_confidence_downweights_low_support_rows():
    scores = torch.ones(3, 2, dtype=torch.float32)
    counts = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float32)

    calibrated = apply_registration_confidence(scores, counts, blend=0.5, mode="linear")

    assert torch.allclose(calibrated[0], torch.full((2,), 0.5))
    assert torch.allclose(calibrated[1], torch.full((2,), 0.75))
    assert torch.allclose(calibrated[2], torch.ones(2))


def test_sample_registration_view_weights_uses_alpha_confidence():
    points = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=torch.float32)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    K = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    alpha = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.8, 1.0], [0.7, 0.5, 0.9]]])

    valid, weights = sample_registration_view_weights(
        points,
        pose,
        K,
        image_height=3,
        image_width=3,
        alpha_map=alpha,
        mode="alpha",
    )

    assert torch.equal(valid, torch.tensor([True, True]))
    assert weights[1] > weights[0]


def test_sample_registration_view_weights_penalizes_depth_mismatch():
    points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]], dtype=torch.float32)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    K = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    depth = torch.ones(1, 3, 3, dtype=torch.float32)
    alpha = torch.ones(1, 3, 3, dtype=torch.float32)

    valid, weights = sample_registration_view_weights(
        points,
        pose,
        K,
        image_height=3,
        image_width=3,
        depth_map=depth,
        alpha_map=alpha,
        depth_tolerance=0.5,
        relative_depth_tolerance=0.0,
        mode="alpha_depth",
    )

    assert torch.equal(valid, torch.tensor([True, True]))
    assert weights[0] > weights[1]


def test_aggregate_scores_by_voxel_dilate_propagates_neighbor_context():
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[0.1], [0.9], [0.2]], dtype=torch.float32)

    same_voxel = aggregate_scores_by_voxel(
        scores,
        xyz,
        mode="voxel_max",
        resolution=6,
        blend=1.0,
    )
    dilated = aggregate_scores_by_voxel(
        scores,
        xyz,
        mode="voxel_max_dilate",
        resolution=6,
        blend=1.0,
    )

    assert same_voxel[0, 0] == scores[0, 0]
    assert dilated[0, 0] == scores[1, 0]


def test_refine_selection_by_voxel_components_keeps_top_score_components():
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.01, 1.0, 1.0],
            [1.02, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    selected = torch.ones(6, 1)
    scores = torch.tensor([[0.1], [0.2], [0.3], [0.9], [0.8], [0.7]])

    refined = refine_selection_by_voxel_components(
        selected,
        scores,
        xyz,
        mode="top_score_components",
        resolution=8,
        keep_components=1,
        min_component_size=1,
        rank_by="mean_score",
    )

    assert torch.equal(refined[:, 0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


def test_refine_selection_by_voxel_components_none_returns_input():
    xyz = torch.rand(4, 3)
    selected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    scores = torch.rand(4, 2)

    refined = refine_selection_by_voxel_components(
        selected,
        scores,
        xyz,
        mode="none",
        resolution=8,
        keep_components=1,
        min_component_size=1,
        rank_by="mean_score",
    )

    assert refined is selected


def test_seed_expand_components_expands_only_seeded_support_component():
    xyz = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.90, 0.90, 0.90],
            [0.91, 0.90, 0.90],
            [0.92, 0.90, 0.90],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[0.95], [0.50], [0.45], [0.70], [0.65], [0.60]])
    seed_selected = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    expanded = select_gaussians_with_seed_expand_components(
        seed_selected,
        scores,
        xyz,
        support_ratio=1.0,
        resolution=8,
        keep_components=1,
        min_component_size=1,
        rank_by="score_sum",
    )

    assert torch.equal(expanded[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_seed_expand_components_falls_back_to_seed_when_no_support_overlap():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    scores = torch.tensor([[0.9], [0.8], [0.7]])
    seed_selected = torch.tensor([[0.0], [0.0], [1.0]])

    expanded = select_gaussians_with_seed_expand_components(
        seed_selected,
        scores,
        xyz,
        support_ratio=0.34,
        resolution=8,
        keep_components=1,
        min_component_size=1,
        rank_by="score_sum",
    )

    assert torch.equal(expanded[:, 0], seed_selected[:, 0])


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean():
    summary = bootstrap_mean_ci([0.1, 0.2, 0.4, 0.8], num_samples=200, seed=7)

    assert summary["mean"] == torch.tensor([0.375]).item()
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    assert summary == bootstrap_mean_ci([0.1, 0.2, 0.4, 0.8], num_samples=200, seed=7)


def test_apply_selection_ratio_bounds_adds_floor_and_caps_selected_scores():
    scores = torch.tensor(
        [
            [0.90, 0.10],
            [0.80, 0.70],
            [0.70, 0.60],
            [0.20, 0.95],
            [0.10, 0.05],
        ],
        dtype=torch.float32,
    )
    selected = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    bounded = apply_selection_ratio_bounds(
        selected,
        scores,
        min_ratio=0.4,
        max_ratio=0.4,
        min_select=1,
    )

    assert torch.equal(bounded[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(bounded[:, 1], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))


def test_score_cache_roundtrip_preserves_scores_and_metadata(tmp_path):
    cache_path = tmp_path / "scores.pt"
    metadata = {
        "scene": "figurines",
        "checkpoint": "checkpoint.pth",
        "score_source": "registered_view",
        "registration_max_frames": 128,
    }
    scores = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    stats = {"registered_gaussians": 2, "num_frames": 128}

    save_score_cache(cache_path, scores, metadata=metadata, registration_stats=stats)
    loaded_scores, loaded_stats = load_score_cache(cache_path, expected_metadata=metadata)

    assert torch.equal(loaded_scores, scores)
    assert loaded_stats == stats


def test_score_cache_rejects_mismatched_protocol(tmp_path):
    cache_path = tmp_path / "scores.pt"
    metadata = {
        "scene": "figurines",
        "score_source": "registered_view",
        "registration_max_frames": 128,
    }
    save_score_cache(
        cache_path,
        torch.ones(2, 1),
        metadata=metadata,
        registration_stats={},
    )

    with pytest.raises(ValueError, match="score cache metadata mismatch"):
        load_score_cache(
            cache_path,
            expected_metadata={
                **metadata,
                "registration_max_frames": 96,
            },
        )


def test_refine_mask_with_rgb_edges_snaps_to_color_boundary():
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[:, :] = (20, 20, 180)
    rgb[16:32, 16:32] = (180, 20, 20)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[10:38, 10:38] = 1

    refined = refine_mask_with_rgb_edges(
        rgb,
        mask,
        iterations=2,
        dilate_pixels=4,
        erode_pixels=3,
    )

    assert refined.dtype == np.bool_
    assert refined.sum() < mask.sum()
    assert refined[20:28, 20:28].mean() > 0.9
    assert refined[:8, :8].sum() == 0


def test_refine_mask_with_rgb_edges_preserves_empty_mask():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)

    refined = refine_mask_with_rgb_edges(rgb, mask)

    assert refined.shape == mask.shape
    assert refined.sum() == 0
