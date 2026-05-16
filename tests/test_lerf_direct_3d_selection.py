import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    SelectionSpec,
    _blend_point_summary_adapter_features,
    apply_registration_confidence,
    apply_selection_ratio_bounds,
    aggregate_scores_by_voxel,
    bootstrap_mean_ci,
    boundary_f_score,
    choose_sam3_box_refined_mask,
    choose_registration_refiner,
    compute_selection_ranking_scores,
    compute_raster_contribution_weights,
    load_score_cache,
    merge_registered_scores,
    accumulate_raster_contribution_features,
    keep_largest_mask_component,
    refine_mask_with_rgb_edges,
    sample_registration_view_weights,
    save_score_cache,
    refine_selection_by_voxel_components,
    score_text_aligned_embeddings,
    select_dominant_raster_hits,
    select_gaussians_from_scores,
    select_top_raster_hits_per_gaussian,
    select_gaussians_by_proposal_components,
    select_gaussians_with_seed_expand_components,
    select_registration_frame_ids,
    save_registered_feature_cache,
    trimap_iou,
    mask_to_sam3_box_prompt,
)


def test_direct_3d_cli_help_builds_without_duplicate_options():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "radio_gs/scripts/eval_lerf_direct_3d_selection.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--mask_refinement_erode" in result.stdout


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


def test_compute_selection_ranking_scores_supports_margin_and_entropy_confidence():
    scores = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.45, 0.40, 0.15],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=torch.float32,
    )

    margin = compute_selection_ranking_scores(scores, mode="score_margin")
    ratio = compute_selection_ranking_scores(scores, mode="score_ratio")
    entropy = compute_selection_ranking_scores(scores, mode="entropy_score")

    assert torch.allclose(margin[0], torch.tensor([0.70, -0.70, -0.70]), atol=1e-6)
    assert ratio[0, 0] > ratio[1, 0] > ratio[2, 0]
    assert entropy[0, 0] > entropy[1, 0] > entropy[2, 0]
    assert torch.allclose(entropy[2], torch.zeros(3), atol=1e-5)


def test_score_margin_selection_uses_confidence_for_floor_and_cap():
    scores = torch.tensor(
        [
            [0.90, 0.10],
            [0.60, 0.55],
            [0.50, 0.49],
            [0.20, 0.80],
        ],
        dtype=torch.float32,
    )

    selected = select_gaussians_from_scores(
        scores,
        SelectionSpec("score_margin", 0.05),
        min_select=1,
    )
    ranking = compute_selection_ranking_scores(scores, mode="score_margin")
    bounded = apply_selection_ratio_bounds(
        selected,
        ranking,
        min_ratio=0.25,
        max_ratio=0.25,
        min_select=1,
    )

    assert torch.equal(bounded[:, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(bounded[:, 1], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_save_registered_feature_cache_persists_summary_teacher(tmp_path):
    xyz = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    summary = torch.nn.functional.normalize(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        dim=-1,
    )
    valid = torch.tensor([True, False])
    view_counts = torch.tensor([2.0, 0.0])

    cache_path = tmp_path / "vpr_summary_cache.pt"
    save_registered_feature_cache(
        cache_path,
        xyz=xyz,
        summary_features=summary,
        valid=valid,
        view_counts=view_counts,
        metadata={"scene": "toy", "prompt_templates": ["{query}"]},
    )

    payload = torch.load(cache_path, map_location="cpu")
    assert payload["version"] == 1
    assert torch.equal(payload["xyz"], xyz)
    assert torch.allclose(payload["summary_features"], summary)
    assert torch.equal(payload["valid"], valid)
    assert torch.equal(payload["view_counts"], view_counts)
    assert payload["metadata"]["scene"] == "toy"


def test_point_summary_adapter_valid_mask_falls_back_to_base_summary():
    base = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        dim=-1,
    )
    adapter = torch.nn.functional.normalize(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        dim=-1,
    )

    blended = _blend_point_summary_adapter_features(
        base,
        adapter,
        alpha=1.0,
        valid_mask=torch.tensor([True, False]),
    )

    assert torch.allclose(blended[0], adapter[0])
    assert torch.allclose(blended[1], base[1])


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


def test_raster_contribution_weights_use_gaussian_footprint_and_depth():
    gaussian_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    pixel_ids = torch.tensor([4, 8, 4], dtype=torch.long)
    means2d = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    conics = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=torch.float32)
    opacities = torch.tensor([[0.8, 0.8]], dtype=torch.float32)
    depths = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    depth_map = torch.ones(1, 3, 3, dtype=torch.float32)
    alpha_map = torch.ones(1, 3, 3, dtype=torch.float32)

    valid, weights = compute_raster_contribution_weights(
        gaussian_ids,
        pixel_ids,
        means2d,
        conics,
        opacities,
        image_height=3,
        image_width=3,
        depths=depths,
        depth_map=depth_map,
        alpha_map=alpha_map,
        mode="alpha_depth",
        depth_tolerance=0.1,
        relative_depth_tolerance=0.0,
    )

    assert torch.equal(valid, torch.tensor([True, True, False]))
    assert weights[0] > weights[1]
    assert weights[2] == 0


def test_accumulate_raster_contribution_features_normalizes_by_weight():
    feature_map = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    gaussian_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    pixel_ids = torch.tensor([0, 1, 3], dtype=torch.long)
    weights = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32)

    sums, counts = accumulate_raster_contribution_features(
        feature_map,
        gaussian_ids,
        pixel_ids,
        weights,
        n_gaussians=2,
    )

    expected0 = torch.tensor([1.0, 2.0]) * 1.0 + torch.tensor([3.0, 4.0]) * 3.0
    expected1 = torch.tensor([7.0, 8.0]) * 2.0
    assert torch.allclose(sums[0], expected0)
    assert torch.allclose(sums[1], expected1)
    assert torch.allclose(counts, torch.tensor([4.0, 2.0]))


def test_select_dominant_raster_hits_keeps_strongest_hit_per_pixel():
    pixel_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    weights = torch.tensor([0.1, 0.8, 0.4, 0.2, 0.4], dtype=torch.float32)

    keep = select_dominant_raster_hits(pixel_ids, weights, num_pixels=2)

    assert torch.equal(keep, torch.tensor([False, True, True, False, True]))


def test_select_top_raster_hits_per_gaussian_keeps_strongest_hit():
    gaussian_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    weights = torch.tensor([0.1, 0.8, 0.4, 0.2, 0.0], dtype=torch.float32)

    keep = select_top_raster_hits_per_gaussian(gaussian_ids, weights, n_gaussians=3)

    assert torch.equal(keep, torch.tensor([False, True, True, False, False]))


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


def test_proposal_components_selects_top_object_support_component():
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
    scores = torch.tensor([[0.40], [0.39], [0.38], [0.92], [0.70], [0.69]])

    selected = select_gaussians_by_proposal_components(
        scores,
        xyz,
        support_ratio=1.0,
        resolution=8,
        keep_components=1,
        min_component_size=1,
        rank_by="mean_score",
    )

    assert torch.equal(selected[:, 0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean():
    summary = bootstrap_mean_ci([0.1, 0.2, 0.4, 0.8], num_samples=200, seed=7)

    assert summary["mean"] == torch.tensor([0.375]).item()
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    assert summary == bootstrap_mean_ci([0.1, 0.2, 0.4, 0.8], num_samples=200, seed=7)


def test_boundary_f_score_prefers_aligned_edges():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 8:24] = 1
    aligned = gt.copy()
    shifted = np.zeros_like(gt)
    shifted[10:26, 10:26] = 1

    assert boundary_f_score(aligned, gt, dilation_ratio=0.02) > boundary_f_score(
        shifted,
        gt,
        dilation_ratio=0.02,
    )


def test_trimap_iou_ignores_far_background():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 8:24] = 1
    pred = gt.copy()
    pred[0:2, 0:2] = 1

    score = trimap_iou(pred, gt, dilation_pixels=2)

    assert 0.95 <= score <= 1.0


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


def test_score_cache_rejects_point_adapter_blend_mismatch(tmp_path):
    cache_path = tmp_path / "scores.pt"
    metadata = {
        "scene": "figurines",
        "score_source": "direct",
        "use_point_summary_adapter": True,
        "point_summary_adapter_blend_alpha": 1.0,
        "point_summary_adapter_valid_mask": "teacher_cache",
    }
    save_score_cache(
        cache_path,
        torch.ones(2, 1),
        metadata=metadata,
        registration_stats={},
    )

    with pytest.raises(ValueError, match="point_summary_adapter_blend_alpha"):
        load_score_cache(
            cache_path,
            expected_metadata={
                **metadata,
                "point_summary_adapter_blend_alpha": 0.5,
            },
        )


def test_score_cache_accepts_legacy_center_assignment_metadata(tmp_path):
    cache_path = tmp_path / "scores.pt"
    legacy_metadata = {
        "scene": "figurines",
        "score_source": "registered_view",
        "registration_max_frames": 128,
    }
    save_score_cache(
        cache_path,
        torch.ones(2, 1),
        metadata=legacy_metadata,
        registration_stats={},
    )

    scores, _ = load_score_cache(
        cache_path,
        expected_metadata={
            **legacy_metadata,
            "registration_assignment_mode": "center",
        },
    )

    assert torch.equal(scores, torch.ones(2, 1))


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


def test_mask_to_sam3_box_prompt_uses_normalized_cxcywh_with_padding():
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[5:15, 10:30] = 1

    box = mask_to_sam3_box_prompt(mask, padding_pixels=2)

    assert np.allclose(box, [0.5, 0.5, 0.6, 0.7], atol=1e-6)


def test_mask_to_sam3_box_prompt_returns_none_for_empty_mask():
    assert mask_to_sam3_box_prompt(np.zeros((8, 8), dtype=np.uint8)) is None


def test_choose_sam3_box_refined_mask_uses_initial_overlap_not_gt():
    initial = np.zeros((16, 16), dtype=np.uint8)
    initial[4:12, 4:12] = 1
    bad = np.zeros_like(initial)
    bad[0:5, 0:5] = 1
    good = np.zeros_like(initial)
    good[3:13, 3:13] = 1
    masks = np.stack([bad, good], axis=0)
    scores = np.asarray([0.99, 0.2], dtype=np.float32)

    refined = choose_sam3_box_refined_mask(
        initial,
        masks,
        scores=scores,
        min_initial_iou=0.05,
    )

    assert refined.dtype == np.bool_
    assert np.array_equal(refined, good.astype(bool))


def test_choose_sam3_box_refined_mask_falls_back_when_overlap_is_too_low():
    initial = np.zeros((16, 16), dtype=np.uint8)
    initial[4:12, 4:12] = 1
    candidate = np.zeros_like(initial)
    candidate[0:2, 0:2] = 1

    refined = choose_sam3_box_refined_mask(
        initial,
        np.stack([candidate], axis=0),
        scores=np.asarray([1.0], dtype=np.float32),
        min_initial_iou=0.2,
    )

    assert np.array_equal(refined, initial.astype(bool))


def test_keep_largest_mask_component_removes_disconnected_fragments():
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    mask[10:20, 10:20] = 1
    mask[0, 23] = 1

    refined = keep_largest_mask_component(mask)

    assert refined.dtype == np.bool_
    assert refined.sum() == 100
    assert refined[12:18, 12:18].all()
    assert not refined[3, 3]
    assert not refined[0, 23]


def test_keep_largest_mask_component_preserves_empty_mask():
    mask = np.zeros((8, 8), dtype=np.uint8)

    refined = keep_largest_mask_component(mask)

    assert refined.shape == mask.shape
    assert refined.sum() == 0
