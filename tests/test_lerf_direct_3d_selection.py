import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    SelectionSpec,
    GaussianSubsetAlphaProxy,
    Sam3AdaptorMaskRefiner,
    apply_direct_primitive_confidence,
    apply_sam3_prompt_heatmap_guard,
    _blend_point_summary_adapter_features,
    _load_point_summary_adapter_valid_mask,
    apply_registration_confidence,
    apply_selection_ratio_bounds,
    aggregate_scores_by_voxel,
    bootstrap_mean_ci,
    boundary_f_score,
    build_direct3d_oracle_prompt_initial_mask,
    build_direct3d_prompt_initial_mask,
    build_direct_head_eval_status,
    choose_sam3_box_refined_mask,
    choose_sam3_box_refined_mask_with_report,
    choose_refined_mask_by_geometry_with_report,
    choose_registration_refiner,
    build_opacity_primitive_confidence,
    compute_geometry_boundary_alignment,
    compute_selection_ranking_scores,
    compute_raster_contribution_weights,
    geometry_discontinuity_maps,
    load_text_projection_head,
    load_score_cache,
    merge_registered_scores,
    normalize_registered_feature_sums,
    accumulate_raster_contribution_features,
    keep_largest_mask_component,
    keep_largest_mask_component_if_dominant,
    keep_mask_components_by_heatmap_score,
    refine_mask_with_rgb_edges,
    refine_mask_with_sam3_feature_grabcut,
    sample_registration_view_weights,
    save_score_cache,
    choose_sam3_mask_head_refined_mask_with_report,
    normalize_score_heatmap_features,
    finalize_prompt_conditioned_sam3_mask,
    enforce_direct_head_eval_consistency,
    refine_mask_with_prompt_conditioned_sam3_head,
    refine_selection_by_voxel_components,
    refine_mask_with_sam3_adaptor_features,
    score_text_aligned_embeddings,
    select_dominant_raster_hits,
    select_gaussians_from_scores,
    select_top_raster_hits_per_gaussian,
    select_gaussians_by_proposal_components,
    select_gaussians_with_seed_expand_components,
    select_registration_frame_ids,
    save_registered_feature_cache,
    smooth_scores_with_voxel_proposals,
    smooth_scores_with_sam3_training_view_proposals,
    summarize_initial_iou_buckets,
    _project_points_to_image,
    trimap_iou,
    vala_knn_minmax_scores,
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
    assert "--save_geometry_maps" in result.stdout
    assert "--sam3_proposal_registration_dir" in result.stdout
    assert "--sam3_proposal_registration_alpha" in result.stdout
    assert "--sam3_proposal_registration_query_conditioned" in result.stdout
    assert "--sam3_prompt_mask_head_initial_refinement" in result.stdout
    assert "--sam3_prompt_mask_head_require_peak_in_refined" in result.stdout
    assert "--sam3_prompt_mask_head_min_heatmap_mean_ratio" in result.stdout
    assert "--sam3_prompt_mask_head_oracle_prompt" in result.stdout
    assert "--allow_sam3_prompt_mask_head_oracle_diagnostic" in result.stdout
    assert "low_confidence_and_proposal_consensus" in result.stdout
    assert "--proposal_consensus_threshold" in result.stdout
    assert "raster_adjoint" in result.stdout
    assert "rgb_grabcut_score_component_guard" in result.stdout
    assert "--score_component_guard_min_mass_fraction" in result.stdout
    assert "--text_encoder" in result.stdout
    assert "openclip" in result.stdout


def test_gaussian_subset_alpha_proxy_physically_removes_unselected_primitives():
    class _Model:
        xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        rotation = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        scaling = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        opacity = torch.tensor([[0.1], [0.2], [0.3], [0.4]])

        def get_xyz(self):
            return self.xyz

        def get_rotation(self):
            return self.rotation

        def get_scaling(self):
            return self.scaling

        def get_opacity(self):
            return self.opacity

    proxy = GaussianSubsetAlphaProxy(_Model(), torch.tensor([False, True, False, True]))

    assert proxy.get_xyz().shape == (2, 3)
    assert torch.equal(proxy.get_xyz(), _Model.xyz[[1, 3]])
    assert torch.equal(proxy.get_opacity(), _Model.opacity[[1, 3]])
    assert torch.equal(proxy.get_features(), torch.ones(2, 1))


def test_gaussian_subset_alpha_proxy_supports_empty_selection():
    class _Model:
        xyz = torch.zeros(2, 3)
        rotation = torch.zeros(2, 4)
        scaling = torch.ones(2, 3)
        opacity = torch.ones(2, 1)

        def get_xyz(self):
            return self.xyz

        def get_rotation(self):
            return self.rotation

        def get_scaling(self):
            return self.scaling

        def get_opacity(self):
            return self.opacity

    proxy = GaussianSubsetAlphaProxy(_Model(), torch.zeros(2, dtype=torch.bool))

    assert proxy.get_xyz().shape == (0, 3)
    assert proxy.get_opacity().shape == (0, 1)
    assert proxy.get_features().shape == (0, 1)


def test_openclip_text_projection_head_is_identity():
    head = load_text_projection_head(
        text_encoder="openclip",
        summary_head_weights="missing-does-not-matter.pth",
        device=torch.device("cpu"),
    )
    x = torch.randn(2, 3, 4)

    y = head(x)

    assert isinstance(head, torch.nn.Identity)
    assert torch.equal(y, x)


def test_direct_3d_cli_rejects_oracle_prompt_without_diagnostic_flag():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "radio_gs/scripts/eval_lerf_direct_3d_selection.py",
            "--config",
            "missing.yaml",
            "--checkpoint",
            "missing.pth",
            "--scene",
            "ramen",
            "--mask_refinement",
            "sam3_prompt_mask_head",
            "--sam3_prompt_mask_head_oracle_prompt",
            "gt_mask",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "allow_sam3_prompt_mask_head_oracle_diagnostic" in result.stderr


def test_project_points_to_image_uses_lerf_w2c_pose_and_intrinsics():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    pose = torch.eye(4)

    pixels, visible = _project_points_to_image(
        xyz,
        pose,
        fx=10.0,
        fy=10.0,
        cx=5.0,
        cy=6.0,
        image_width=32,
        image_height=32,
    )

    assert torch.allclose(pixels[0], torch.tensor([5.0, 6.0]))
    assert torch.allclose(pixels[1], torch.tensor([15.0, 6.0]))
    assert visible.tolist() == [1.0, 1.0, 0.0]


def test_refine_mask_with_sam3_adaptor_features_snaps_overwide_boundary():
    features = torch.zeros((2, 16, 16), dtype=torch.float32)
    features[0].fill_(-1.0)
    features[1].fill_(1.0)
    target = np.zeros((32, 32), dtype=bool)
    target[8:24, 10:22] = True
    features[0, 4:12, 5:11] = 1.0
    features[1, 4:12, 5:11] = -1.0
    features = torch.nn.functional.normalize(features, dim=0)

    initial = np.zeros((32, 32), dtype=bool)
    initial[5:27, 7:25] = True
    initial_iou = np.logical_and(initial, target).sum() / np.logical_or(initial, target).sum()

    refined, report = refine_mask_with_sam3_adaptor_features(
        features,
        initial,
        support_dilate_pixels=2,
        inner_erode_pixels=2,
        score_std_scale=0.0,
        min_area_scale=0.30,
        max_area_scale=1.05,
        min_initial_iou=0.01,
    )
    refined_iou = np.logical_and(refined, target).sum() / np.logical_or(refined, target).sum()

    assert report["attempted"] is True
    assert report["accepted"] is True
    assert refined_iou > initial_iou + 0.20


def test_keep_largest_mask_component_if_dominant_preserves_multicomponent_support():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True

    guarded, report = keep_largest_mask_component_if_dominant(
        mask,
        min_largest_fraction=0.65,
    )

    assert guarded.tolist() == mask.tolist()
    assert report["component_guard_kept_largest"] is False
    assert report["component_guard_component_count"] == 2
    assert report["component_guard_largest_fraction"] == 0.5


def test_keep_largest_mask_component_if_dominant_removes_small_fragments():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:7, 1:7] = True
    mask[8:10, 8:10] = True

    guarded, report = keep_largest_mask_component_if_dominant(
        mask,
        min_largest_fraction=0.65,
    )

    assert int(guarded.sum()) == 36
    assert report["component_guard_kept_largest"] is True
    assert report["component_guard_component_count"] == 2
    assert report["component_guard_largest_fraction"] >= 0.9


def test_keep_largest_mask_component_if_dominant_uses_small_support_floor():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True

    guarded, report = keep_largest_mask_component_if_dominant(
        mask,
        min_largest_fraction=0.65,
        min_total_pixels_for_multicomponent=12,
    )

    assert int(guarded.sum()) == 4
    assert report["component_guard_kept_largest"] is True
    assert report["component_guard_kept_largest_due_to_small_support"] is True
    assert report["component_guard_min_total_pixels_for_multicomponent"] == 12


def test_keep_mask_components_by_heatmap_score_keeps_high_score_support():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:7, 1:7] = True
    mask[8:10, 8:10] = True
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[1:7, 1:7] = 0.02
    heatmap[8:10, 8:10] = 1.0

    guarded, report = keep_mask_components_by_heatmap_score(
        mask,
        heatmap,
        min_mass_fraction=0.5,
    )

    assert int(guarded.sum()) == 4
    assert guarded[8:10, 8:10].all()
    assert report["score_component_guard_component_count"] == 2
    assert report["score_component_guard_kept_components"] == 1


def test_keep_mask_components_by_heatmap_score_can_preserve_multiple_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True
    mask[6:9, 6:9] = True
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[1:4, 1:4] = 1.0
    heatmap[6:9, 6:9] = 0.8

    guarded, report = keep_mask_components_by_heatmap_score(
        mask,
        heatmap,
        min_mass_fraction=0.5,
        min_total_pixels_for_multicomponent=0,
    )

    assert guarded.tolist() == mask.tolist()
    assert report["score_component_guard_kept_components"] == 2


def test_keep_mask_components_by_heatmap_score_can_recover_tiny_support_from_heatmap():
    mask = np.zeros((8, 8), dtype=bool)
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[3:5, 3:5] = 1.0

    guarded, report = keep_mask_components_by_heatmap_score(
        mask,
        heatmap,
        min_recovery_pixels=4,
    )

    assert int(guarded.sum()) == 4
    assert guarded[3:5, 3:5].all()
    assert report["score_component_guard_heatmap_recovered"] is True


def test_smooth_scores_with_voxel_proposals_recovers_noisy_object_member():
    scores = torch.tensor(
        [
            [4.0, 0.0],
            [0.2, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    xyz = torch.tensor(
        [
            [0.01, 0.01, 0.01],
            [0.09, 0.02, 0.01],
            [0.30, 0.01, 0.01],
        ],
        dtype=torch.float32,
    )

    smoothed, stats = smooth_scores_with_voxel_proposals(
        scores,
        xyz,
        voxel_size=0.1,
        alpha=0.5,
        min_count=2,
        gate="low_margin",
        margin_threshold=0.5,
    )

    assert stats["enabled"] is True
    assert stats["num_proposals"] == 2
    assert torch.allclose(smoothed[0], scores[0])
    assert torch.allclose(smoothed[1], torch.tensor([1.15, 0.0]))
    assert torch.allclose(smoothed[2], scores[2])


def test_smooth_scores_with_voxel_proposals_can_require_consensus():
    scores = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [3.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    xyz = torch.tensor(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.01, 0.01],
            [0.30, 0.01, 0.01],
            [0.31, 0.01, 0.01],
        ],
        dtype=torch.float32,
    )

    smoothed, stats = smooth_scores_with_voxel_proposals(
        scores,
        xyz,
        voxel_size=0.1,
        alpha=0.5,
        min_count=2,
        gate="proposal_consensus",
        proposal_consensus_threshold=0.75,
    )

    assert stats["num_consensus_proposals"] == 1
    assert stats["num_assigned"] == 2
    assert torch.allclose(smoothed[0], scores[0])
    assert torch.allclose(smoothed[1], scores[1])
    assert torch.allclose(smoothed[2], torch.tensor([2.5, 0.0]))
    assert torch.allclose(smoothed[3], torch.tensor([1.5, 0.0]))


def test_refine_mask_with_sam3_adaptor_features_empty_mask_falls_back():
    features = torch.nn.functional.normalize(torch.rand((4, 8, 8)), dim=0)
    initial = np.zeros((16, 16), dtype=bool)

    refined, report = refine_mask_with_sam3_adaptor_features(features, initial)

    assert not refined.any()
    assert report["accepted"] is False
    assert report["fallback_reason"] == "empty_initial_mask"


def test_refine_mask_with_sam3_adaptor_features_can_skip_large_layout_masks():
    features = torch.nn.functional.normalize(torch.rand((4, 8, 8)), dim=0)
    initial = np.ones((16, 16), dtype=bool)

    refined, report = refine_mask_with_sam3_adaptor_features(
        features,
        initial,
        max_initial_area_fraction=0.5,
    )

    assert np.array_equal(refined, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "initial_mask_too_large"


def test_refine_mask_with_sam3_adaptor_features_box_support_can_expand_inside_prompt_box():
    features = torch.zeros((2, 16, 16), dtype=torch.float32)
    features[0].fill_(-1.0)
    features[1].fill_(1.0)
    target = np.zeros((32, 32), dtype=bool)
    target[8:24, 8:24] = True
    features[0, 4:12, 4:12] = 1.0
    features[1, 4:12, 4:12] = -1.0
    features = torch.nn.functional.normalize(features, dim=0)

    initial = np.zeros((32, 32), dtype=bool)
    initial[10:22, 10:22] = True
    initial_iou = np.logical_and(initial, target).sum() / np.logical_or(initial, target).sum()

    refined, report = refine_mask_with_sam3_adaptor_features(
        features,
        initial,
        support_mode="box",
        prototype_mode="box",
        support_dilate_pixels=4,
        inner_erode_pixels=0,
        score_std_scale=0.0,
        min_area_scale=1.0,
        max_area_scale=2.0,
        background_weight=0.0,
        min_initial_iou=0.01,
    )
    refined_iou = np.logical_and(refined, target).sum() / np.logical_or(refined, target).sum()

    assert report["accepted"] is True
    assert report["support_mode"] == "box"
    assert refined_iou > initial_iou + 0.20


def test_choose_sam3_mask_head_refined_mask_selects_candidate_by_initial_overlap():
    initial = np.zeros((16, 16), dtype=bool)
    initial[4:12, 4:12] = True
    logits = torch.full((3, 8, 8), -8.0)
    logits[0, 1:3, 1:3] = 8.0
    logits[1, 2:6, 2:6] = 8.0
    logits[2, 5:8, 5:8] = 8.0

    refined, report = choose_sam3_mask_head_refined_mask_with_report(
        initial,
        logits,
        logit_threshold=0.0,
        min_initial_iou=0.1,
    )

    assert report["attempted"] is True
    assert report["accepted"] is True
    assert report["selected_index"] == 1
    assert refined.sum() == 64


def test_choose_sam3_mask_head_refined_mask_falls_back_on_low_overlap():
    initial = np.zeros((16, 16), dtype=bool)
    initial[0:3, 0:3] = True
    logits = torch.full((1, 8, 8), -8.0)
    logits[0, 5:8, 5:8] = 8.0

    refined, report = choose_sam3_mask_head_refined_mask_with_report(
        initial,
        logits,
        logit_threshold=0.0,
        min_initial_iou=0.1,
    )

    assert np.array_equal(refined, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "low_initial_overlap"


def test_choose_sam3_mask_head_refined_mask_rejects_area_shrinkage():
    initial = np.zeros((16, 16), dtype=bool)
    initial[3:13, 3:13] = True
    logits = torch.full((1, 8, 8), -8.0)
    logits[0, 3:5, 3:5] = 8.0

    refined, report = choose_sam3_mask_head_refined_mask_with_report(
        initial,
        logits,
        logit_threshold=0.0,
        min_initial_iou=0.01,
        min_refined_area_ratio=0.5,
    )

    assert np.array_equal(refined, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "refined_mask_too_small"


def test_choose_sam3_mask_head_refined_mask_clips_to_support_band():
    initial = np.zeros((16, 16), dtype=bool)
    initial[4:10, 4:10] = True
    logits = torch.full((1, 8, 8), -8.0)
    logits[0, 2:5, 2:5] = 8.0
    logits[0, 6:8, 6:8] = 8.0

    refined, report = choose_sam3_mask_head_refined_mask_with_report(
        initial,
        logits,
        logit_threshold=0.0,
        min_initial_iou=0.01,
        max_refined_area_ratio=2.0,
        support_dilate=1,
    )

    assert report["accepted"] is True
    assert refined[12:16, 12:16].sum() == 0


def test_refine_mask_with_prompt_conditioned_sam3_head_uses_prompt_and_coarse_mask():
    class FakePromptHead(torch.nn.Module):
        def forward(self, features, prompts, coarse_masks):
            logits = torch.full((1, 1, 8, 8), -8.0, device=features.device)
            logits[:, :, 2:6, 2:6] = 8.0 + prompts.sum() * 0.0 + coarse_masks.sum() * 0.0
            return logits

    initial = np.zeros((16, 16), dtype=bool)
    initial[3:13, 3:13] = True
    gt_like = np.zeros((16, 16), dtype=bool)
    gt_like[4:12, 4:12] = True
    initial_iou = np.logical_and(initial, gt_like).sum() / np.logical_or(initial, gt_like).sum()

    refined, report = refine_mask_with_prompt_conditioned_sam3_head(
        feature_map=torch.zeros(1, 4, 8, 8),
        prompt_embedding=torch.ones(6),
        coarse_mask=initial,
        head=FakePromptHead(),
        logit_threshold=0.0,
        min_initial_iou=0.1,
    )
    refined_iou = np.logical_and(refined, gt_like).sum() / np.logical_or(refined, gt_like).sum()

    assert report["attempted"] is True
    assert report["accepted"] is True
    assert refined_iou > initial_iou


def test_prompt_conditioned_sam3_head_quality_gate_falls_back_on_low_quality():
    class LowQualityPromptHead(torch.nn.Module):
        def forward(self, features, prompts, coarse_masks):
            logits = torch.full((1, 1, 8, 8), -8.0, device=features.device)
            logits[:, :, 2:6, 2:6] = 8.0 + prompts.sum() * 0.0 + coarse_masks.sum() * 0.0
            return logits

        def forward_with_quality(self, features, prompts, coarse_masks):
            logits = self.forward(features, prompts, coarse_masks)
            quality = torch.full((1, 1), -6.0, device=features.device)
            return logits, quality

    initial = np.zeros((16, 16), dtype=bool)
    initial[3:13, 3:13] = True

    refined, report = refine_mask_with_prompt_conditioned_sam3_head(
        feature_map=torch.zeros(1, 4, 8, 8),
        prompt_embedding=torch.ones(6),
        coarse_mask=initial,
        head=LowQualityPromptHead(),
        logit_threshold=0.0,
        min_initial_iou=0.1,
        min_quality=0.8,
    )

    assert np.array_equal(refined, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "low_predicted_quality"


def test_direct3d_prompt_initial_mask_can_keep_heatmap_peak_component():
    coarse = np.zeros((8, 8), dtype=bool)
    coarse[1:3, 1:3] = True
    coarse[5:7, 5:7] = True
    heatmap = torch.zeros((8, 8), dtype=torch.float32)
    heatmap[5, 5] = 1.0

    refined = build_direct3d_prompt_initial_mask(
        coarse,
        heatmap,
        initial_refinement="peak_component",
    )

    expected = np.zeros_like(coarse)
    expected[5:7, 5:7] = True
    assert np.array_equal(refined, expected)


def test_direct3d_oracle_prompt_initial_mask_supports_gt_mask_and_box():
    coarse = np.zeros((8, 8), dtype=bool)
    coarse[0:2, 0:2] = True
    gt = np.zeros((8, 8), dtype=bool)
    gt[2:5, 3] = True
    gt[4, 3:7] = True

    gt_prompt = build_direct3d_oracle_prompt_initial_mask(coarse, gt, mode="gt_mask")
    box_prompt = build_direct3d_oracle_prompt_initial_mask(coarse, gt, mode="gt_box")
    no_oracle = build_direct3d_oracle_prompt_initial_mask(coarse, gt, mode="none")

    expected_box = np.zeros_like(gt)
    expected_box[2:5, 3:7] = True
    assert np.array_equal(gt_prompt, gt)
    assert np.array_equal(box_prompt, expected_box)
    assert np.array_equal(no_oracle, coarse)


def test_sam3_prompt_heatmap_guard_rejects_refinement_that_drops_peak():
    initial = np.zeros((8, 8), dtype=bool)
    initial[2:6, 2:6] = True
    refined = initial.copy()
    refined[4, 4] = False
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[4, 4] = 1.0

    guarded, report = apply_sam3_prompt_heatmap_guard(
        initial,
        refined,
        heatmap,
        min_mean_ratio=0.0,
        min_mass_ratio=0.0,
        require_peak_in_refined=True,
    )

    assert np.array_equal(guarded, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "heatmap_peak_outside_refined"


def test_prompt_conditioned_sam3_fallback_keeps_original_coarse_mask():
    coarse = np.zeros((8, 8), dtype=bool)
    coarse[1:7, 1:7] = True
    prompt_initial = np.zeros((8, 8), dtype=bool)
    prompt_initial[3:5, 3:5] = True
    rejected_candidate = np.zeros((8, 8), dtype=bool)
    rejected_candidate[0:2, 0:2] = True

    report = {"attempted": True, "accepted": True, "fallback_reason": ""}
    heatmap_report = {"accepted": False, "fallback_reason": "missing_heatmap_peak"}

    final, final_report = finalize_prompt_conditioned_sam3_mask(
        coarse,
        prompt_initial,
        rejected_candidate,
        report,
        heatmap_guard_report=heatmap_report,
    )

    assert np.array_equal(final, coarse)
    assert not np.array_equal(final, prompt_initial)
    assert final_report["accepted"] is False
    assert final_report["fallback_reason"] == "missing_heatmap_peak"


def test_normalize_score_heatmap_features_scales_each_query_independently():
    scores = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 20.0],
        ]
    )

    norm = normalize_score_heatmap_features(scores)

    assert torch.allclose(norm[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(norm[:, 1], torch.tensor([0.0, 1.0, 1.0]))


def test_refine_mask_with_sam3_feature_grabcut_uses_feature_boundaries():
    features = torch.zeros((3, 16, 16), dtype=torch.float32)
    features[0].fill_(0.0)
    features[2].fill_(1.0)
    target = np.zeros((32, 32), dtype=bool)
    target[8:24, 8:24] = True
    features[0, 4:12, 4:12] = 1.0
    features[2, 4:12, 4:12] = 0.0

    initial = np.zeros((32, 32), dtype=bool)
    initial[5:27, 5:27] = True
    initial_iou = np.logical_and(initial, target).sum() / np.logical_or(initial, target).sum()

    refined, report = refine_mask_with_sam3_feature_grabcut(
        features,
        initial,
        iterations=2,
        dilate_pixels=2,
        erode_pixels=2,
    )
    refined_iou = np.logical_and(refined, target).sum() / np.logical_or(refined, target).sum()

    assert report["attempted"] is True
    assert report["accepted"] is True
    assert refined_iou > initial_iou + 0.20


def test_refine_mask_with_sam3_feature_grabcut_rejects_unstable_area_ratio():
    features = torch.zeros((3, 16, 16), dtype=torch.float32)
    features[0].fill_(0.0)
    features[2].fill_(1.0)
    features[0, 4:12, 4:12] = 1.0
    features[2, 4:12, 4:12] = 0.0
    initial = np.zeros((32, 32), dtype=bool)
    initial[5:27, 5:27] = True

    refined, report = refine_mask_with_sam3_feature_grabcut(
        features,
        initial,
        iterations=2,
        dilate_pixels=2,
        erode_pixels=2,
        max_refined_area_ratio=0.4,
    )

    assert np.array_equal(refined, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "refined_area_too_large"


def test_sam3_adaptor_refiner_exposes_feature_grabcut_readout():
    assert hasattr(Sam3AdaptorMaskRefiner, "refine_grabcut_from_state_with_report")


def test_choose_refined_mask_by_geometry_accepts_boundary_aligned_candidate():
    initial = np.zeros((16, 16), dtype=bool)
    initial[3:13, 3:13] = True
    refined = np.zeros((16, 16), dtype=bool)
    refined[4:12, 4:12] = True
    alpha = np.zeros((16, 16), dtype=np.float32)
    alpha[4:12, 4:12] = 1.0
    depth = np.zeros_like(alpha)

    chosen, report = choose_refined_mask_by_geometry_with_report(
        initial,
        refined,
        alpha,
        depth,
        min_area_ratio=0.5,
        max_area_ratio=1.5,
        min_boundary_gain=-1e-6,
    )

    assert np.array_equal(chosen, refined)
    assert report["geometry_gate_accepted"] is True


def test_choose_refined_mask_by_geometry_rejects_area_outlier():
    initial = np.zeros((16, 16), dtype=bool)
    initial[4:12, 4:12] = True
    refined = np.ones((16, 16), dtype=bool)

    chosen, report = choose_refined_mask_by_geometry_with_report(
        initial,
        refined,
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((16, 16), dtype=np.float32),
        min_area_ratio=0.5,
        max_area_ratio=1.5,
    )

    assert np.array_equal(chosen, initial)
    assert report["geometry_gate_accepted"] is False
    assert report["geometry_gate_reason"] == "area_ratio_out_of_range"


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


def test_vala_knn_minmax_scores_matches_released_readout_shape_and_scale():
    scores = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32)
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    normalized = vala_knn_minmax_scores(scores, xyz, k=2, chunk_size=2)

    assert normalized.shape == scores.shape
    assert normalized[0, 0] == pytest.approx(0.0)
    assert normalized[2, 0] == pytest.approx(1.0)
    # The released evaluator remaps min-max scores as clip(2*x-1, 0, 1).
    assert normalized[1, 0] == pytest.approx(0.0, abs=1e-6)


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
    scales = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]])
    opacities = torch.tensor([[0.8], [0.2]])
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
        scales=scales,
        rotations=rotations,
        opacities=opacities,
    )

    payload = torch.load(cache_path, map_location="cpu")
    assert payload["version"] == 1
    assert payload["feature_space"] == "siglip_summary"
    assert payload["feature_key"] == "summary_features"
    assert payload["geometry_fingerprint"]["num_gaussians"] == 2
    assert isinstance(payload["geometry_fingerprint"]["xyz_sha256"], str)
    assert len(payload["geometry_fingerprint"]["xyz_sha256"]) == 64
    assert payload["geometry_fingerprint"]["scales_shape"] == [2, 3]
    assert len(payload["geometry_fingerprint"]["scales_sha256"]) == 64
    assert payload["geometry_fingerprint"]["rotations_shape"] == [2, 4]
    assert len(payload["geometry_fingerprint"]["rotations_sha256"]) == 64
    assert payload["geometry_fingerprint"]["opacities_shape"] == [2, 1]
    assert len(payload["geometry_fingerprint"]["opacities_sha256"]) == 64
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


def test_point_summary_adapter_valid_mask_uses_fallback_teacher_cache(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pth"
    teacher_cache_path = tmp_path / "teacher_cache.pt"
    valid = torch.tensor([True, False, True])

    torch.save({}, ckpt_path)
    torch.save({"valid": valid}, teacher_cache_path)

    loaded = _load_point_summary_adapter_valid_mask(
        str(ckpt_path),
        expected_count=3,
        device=torch.device("cpu"),
        fallback_teacher_cache=str(teacher_cache_path),
    )

    assert loaded is not None
    assert torch.equal(loaded.cpu(), valid)


def test_point_summary_adapter_valid_mask_rejects_misaligned_teacher_cache(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pth"
    teacher_cache_path = tmp_path / "teacher_cache.pt"
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)

    torch.save({}, ckpt_path)
    save_registered_feature_cache(
        teacher_cache_path,
        xyz=xyz,
        summary_features=torch.nn.functional.normalize(torch.ones(2, 3), dim=-1),
        valid=torch.tensor([True, False]),
        view_counts=torch.tensor([1.0, 0.0]),
        metadata={"scene": "toy"},
    )

    with pytest.raises(ValueError, match="teacher cache geometry mismatch"):
        _load_point_summary_adapter_valid_mask(
            str(ckpt_path),
            expected_count=2,
            device=torch.device("cpu"),
            fallback_teacher_cache=str(teacher_cache_path),
            expected_xyz=torch.flip(xyz, dims=[0]),
        )


def test_prompt_conditioned_sam3_head_resizes_coarse_prompt_before_dilation():
    seen: dict[str, torch.Tensor] = {}

    class _Head(torch.nn.Module):
        def forward(self, feature_map, prompt_embedding, coarse_prompt):
            seen["coarse"] = coarse_prompt.detach().cpu()
            return coarse_prompt[:, :1].float() * 2.0 - 1.0

    coarse = np.array([[1, 0], [0, 0]], dtype=bool)
    refined, report = refine_mask_with_prompt_conditioned_sam3_head(
        feature_map=torch.zeros(1, 4, 4, 4),
        prompt_embedding=torch.zeros(3),
        coarse_mask=coarse,
        head=_Head(),
        logit_threshold=0.0,
        min_initial_iou=0.0,
        coarse_dilate=1,
        coarse_threshold=0.0,
    )

    assert report["coarse_prompt_resized"] is True
    assert report["coarse_prompt_input_shape"] == [2, 2]
    assert report["coarse_prompt_shape"] == [4, 4]
    assert tuple(seen["coarse"].shape[-2:]) == (4, 4)
    assert refined.shape == coarse.shape


def test_prompt_conditioned_sam3_resize_uses_nearest_without_halo():
    seen: dict[str, torch.Tensor] = {}

    class _CaptureHead(torch.nn.Module):
        def forward(self, feature_map, prompt_embedding, coarse_prompt):
            seen["coarse"] = coarse_prompt.detach().cpu()
            return torch.zeros((1, 1, *feature_map.shape[-2:]), device=feature_map.device)

    coarse = np.zeros((4, 4), dtype=bool)
    coarse[1, 1] = True
    refine_mask_with_prompt_conditioned_sam3_head(
        feature_map=torch.zeros(1, 2, 8, 8),
        prompt_embedding=torch.zeros(3),
        coarse_mask=coarse,
        head=_CaptureHead(),
        logit_threshold=0.0,
        min_initial_iou=0.0,
        coarse_dilate=0,
        coarse_threshold=0.5,
    )

    assert tuple(seen["coarse"].shape[-2:]) == (8, 8)
    assert int(seen["coarse"].sum().item()) == 4


def test_initial_iou_bucket_summary_reports_delta_by_support_quality():
    details = [
        {"initial_iou": 0.10, "iou": 0.20, "delta_iou": 0.10, "sam3_accepted": True},
        {"initial_iou": 0.40, "iou": 0.35, "delta_iou": -0.05, "sam3_accepted": False},
        {"initial_iou": 0.80, "iou": 0.82, "delta_iou": 0.02, "sam3_accepted": True},
    ]

    summary = summarize_initial_iou_buckets(details)

    assert summary["lt_0p25"]["n"] == 1
    assert summary["lt_0p25"]["delta_miou"] == pytest.approx(0.10)
    assert summary["0p25_0p50"]["sam3_accept_rate"] == pytest.approx(0.0)
    assert summary["0p50_0p75"]["n"] == 0
    assert summary["gte_0p75"]["miou"] == pytest.approx(0.82)


def test_direct_head_contract_rejects_valid_mask_mismatch():
    checkpoint = {
        "point_summary_adapter_state_dict": {},
        "point_summary_adapter_metadata": {
            "direct_head_contract": {
                "compact_feature_key": "features",
                "direct_readout_mode": "gaussian",
                "point_summary_adapter_blend_alpha": 1.0,
                "point_summary_adapter_valid_mask_mode": "teacher_cache",
            }
        },
    }

    status = build_direct_head_eval_status(
        checkpoint,
        score_source="direct",
        use_point_summary_adapter=True,
        adapter_loaded=True,
        compact_feature_key="features",
        direct_readout_mode="gaussian",
        point_summary_adapter_blend_alpha=1.0,
        point_summary_adapter_valid_mask_mode="opacity",
    )

    assert any("valid_mask_mode_mismatch" in warning for warning in status["warnings"])
    with pytest.raises(ValueError, match="valid_mask_mode_mismatch"):
        enforce_direct_head_eval_consistency(status, strict=True)


def test_direct_head_contract_rejects_teacher_space_and_query_mode_mismatch():
    checkpoint = {
        "point_summary_adapter_state_dict": {},
        "point_summary_adapter_metadata": {
            "direct_head_contract": {
                "compact_feature_key": "features",
                "direct_readout_mode": "gaussian",
                "point_summary_adapter_blend_alpha": 1.0,
                "point_summary_adapter_valid_mask_mode": "teacher_cache",
                "point_summary_adapter_context_features": "opacity view_count",
                "teacher_feature_space": "siglip_summary",
                "teacher_cache_feature_key": "summary_features",
                "direct_point_query_mode": "gaussian_index",
                "direct_point_gaussian_position_mode": "gaussian_center",
            }
        },
    }

    status = build_direct_head_eval_status(
        checkpoint,
        score_source="direct",
        use_point_summary_adapter=True,
        adapter_loaded=True,
        compact_feature_key="features",
        direct_readout_mode="gaussian",
        point_summary_adapter_blend_alpha=1.0,
        point_summary_adapter_valid_mask_mode="teacher_cache",
        point_summary_adapter_context_features="opacity view_count",
        teacher_feature_space="radio",
        teacher_cache_feature_key="features",
        direct_point_query_mode="knn",
        direct_point_gaussian_position_mode="label_point",
    )

    warnings = " ".join(status["warnings"])
    assert "teacher_feature_space_mismatch" in warnings
    assert "teacher_cache_feature_key_mismatch" in warnings
    assert "direct_point_query_mode_mismatch" in warnings
    assert "direct_point_gaussian_position_mode_mismatch" in warnings
    with pytest.raises(ValueError, match="teacher_feature_space_mismatch"):
        enforce_direct_head_eval_consistency(status, strict=True)


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


def test_opacity_primitive_confidence_thresholds_and_normalizes():
    opacity = torch.tensor([[0.0], [0.02], [0.50], [1.00]])

    confidence = build_opacity_primitive_confidence(
        opacity,
        mode="opacity",
        threshold=0.05,
    )

    assert torch.allclose(confidence, torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_apply_direct_primitive_confidence_scales_rows():
    scores = torch.ones(3, 2, dtype=torch.float32)
    confidence = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    calibrated = apply_direct_primitive_confidence(scores, confidence, blend=0.5)

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


def test_normalize_registered_feature_sums_uses_subunit_weights():
    registered_sum = torch.tensor(
        [
            [0.2, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    registered_counts = torch.tensor([0.2, 0.0], dtype=torch.float32)

    normalized = normalize_registered_feature_sums(registered_sum, registered_counts)

    assert torch.allclose(normalized[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(normalized[1], torch.zeros(2))


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


def test_geometry_discontinuity_maps_detect_alpha_and_depth_edges():
    alpha = np.zeros((16, 16), dtype=np.float32)
    alpha[:, 8:] = 1.0
    depth = np.ones((16, 16), dtype=np.float32)
    depth[8:, :] = 3.0

    maps = geometry_discontinuity_maps(alpha, depth)

    assert set(maps) == {"alpha_edge", "depth_edge", "discontinuity"}
    assert maps["alpha_edge"][:, 7:9].mean() > maps["alpha_edge"][:, :3].mean()
    assert maps["depth_edge"][7:9, :].mean() > maps["depth_edge"][:3, :].mean()
    assert maps["discontinuity"].max() <= 1.0


def test_compute_geometry_boundary_alignment_scores_query_boundaries():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[:, 16:] = 1.0
    depth = np.ones((32, 32), dtype=np.float32)
    depth[16:, :] = 4.0
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 8:24] = 1
    pred = np.zeros_like(gt)
    pred[8:24, 12:28] = 1

    metrics = compute_geometry_boundary_alignment(pred, gt, alpha, depth)

    assert metrics["geometry_valid"] == 1
    assert metrics["gt_boundary_pixels"] > 0
    assert metrics["pred_boundary_pixels"] > 0
    assert metrics["discontinuity_gt_boundary_mean"] > 0.0
    assert metrics["discontinuity_pred_boundary_mean"] > 0.0
    assert metrics["discontinuity_error_boundary_mean"] >= 0.0


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


def test_score_cache_rejects_mismatched_geometry_fingerprint(tmp_path):
    cache_path = tmp_path / "scores.pt"
    metadata = {
        "scene": "figurines",
        "score_source": "registered_view",
    }
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    save_score_cache(
        cache_path,
        torch.ones(2, 1),
        metadata=metadata,
        registration_stats={},
        xyz=xyz,
    )

    with pytest.raises(ValueError, match="score cache geometry mismatch"):
        load_score_cache(
            cache_path,
            expected_metadata=metadata,
            expected_xyz=torch.flip(xyz, dims=[0]),
        )


def test_score_cache_requires_geometry_when_expected(tmp_path):
    cache_path = tmp_path / "legacy_scores.pt"
    metadata = {
        "scene": "figurines",
        "score_source": "registered_view",
    }
    save_score_cache(
        cache_path,
        torch.ones(2, 1),
        metadata=metadata,
        registration_stats={},
    )

    with pytest.raises(ValueError, match="missing geometry_fingerprint"):
        load_score_cache(
            cache_path,
            expected_metadata=metadata,
            expected_xyz=torch.zeros(2, 3),
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


def test_score_cache_accepts_legacy_default_direct_confidence_metadata(tmp_path):
    cache_path = tmp_path / "scores.pt"
    legacy_metadata = {
        "scene": "waldo_kitchen",
        "score_source": "direct",
        "registration_assignment_mode": "center",
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
            "direct_primitive_confidence_mode": "none",
            "direct_primitive_confidence_blend": 0.0,
            "direct_primitive_opacity_threshold": 0.02,
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


def test_choose_sam3_box_refined_mask_reports_feature_prompt_acceptance():
    initial = np.zeros((16, 16), dtype=np.uint8)
    initial[4:12, 4:12] = 1
    candidate = np.zeros_like(initial)
    candidate[3:13, 3:13] = 1

    refined, report = choose_sam3_box_refined_mask_with_report(
        initial,
        np.stack([candidate], axis=0),
        scores=np.asarray([0.7], dtype=np.float32),
        min_initial_iou=0.05,
    )

    assert np.array_equal(refined, candidate.astype(bool))
    assert report["accepted"] is True
    assert report["fallback_reason"] == "accepted"
    assert report["candidate_count"] == 1
    assert report["selected_index"] == 0
    assert report["best_initial_overlap"] > 0.05
    assert report["selected_score"] == pytest.approx(0.7)


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


def test_choose_sam3_box_refined_mask_reports_low_overlap_fallback():
    initial = np.zeros((16, 16), dtype=np.uint8)
    initial[4:12, 4:12] = 1
    candidate = np.zeros_like(initial)
    candidate[0:2, 0:2] = 1

    refined, report = choose_sam3_box_refined_mask_with_report(
        initial,
        np.stack([candidate], axis=0),
        scores=np.asarray([1.0], dtype=np.float32),
        min_initial_iou=0.2,
    )

    assert np.array_equal(refined, initial.astype(bool))
    assert report["accepted"] is False
    assert report["fallback_reason"] == "low_initial_overlap"
    assert report["candidate_count"] == 1
    assert report["selected_index"] == 0


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
