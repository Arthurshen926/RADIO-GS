import numpy as np
import torch

from radio_gs.models.prompt_conditioned_mask_refinement import (
    filter_refined_mask_by_heatmap_support,
    refine_mask_with_prompt_conditioned_sam3_head,
)
from radio_gs.scripts.eval_lerf_grounding import build_sam3_prompt_initial_mask


def test_heatmap_support_filter_rejects_refinement_that_drops_peak():
    initial = np.zeros((5, 5), dtype=bool)
    initial[1:4, 1:4] = True
    refined = initial.copy()
    refined[2, 2] = False
    heatmap = np.zeros((5, 5), dtype=np.float32)
    heatmap[2, 2] = 1.0

    pred, report = filter_refined_mask_by_heatmap_support(
        initial,
        refined,
        heatmap,
        require_peak_in_refined=True,
    )

    assert np.array_equal(pred, initial)
    assert report["accepted"] is False
    assert report["fallback_reason"] == "heatmap_peak_outside_refined"


def test_heatmap_support_filter_accepts_boundary_tightening_that_keeps_peak():
    initial = np.zeros((5, 5), dtype=bool)
    initial[1:4, 1:4] = True
    refined = np.zeros((5, 5), dtype=bool)
    refined[1:3, 1:3] = True
    heatmap = np.zeros((5, 5), dtype=np.float32)
    heatmap[1:3, 1:3] = 1.0
    heatmap[3, 3] = 0.1

    pred, report = filter_refined_mask_by_heatmap_support(
        initial,
        refined,
        heatmap,
        min_mean_ratio=0.9,
        min_mass_ratio=0.3,
        require_peak_in_refined=True,
    )

    assert np.array_equal(pred, refined)
    assert report["accepted"] is True
    assert report["fallback_reason"] == "accepted"


def test_sam3_prompt_initial_mask_can_keep_peak_component():
    heatmap = torch.zeros(5, 5)
    heatmap[1, 1] = 1.0
    heatmap[3, 3] = 0.9

    pred = build_sam3_prompt_initial_mask(
        heatmap,
        threshold_ratio=0.5,
        threshold_mode="fixed",
        threshold_mean_std_k=1.0,
        threshold_min_ratio=0.0,
        threshold_max_ratio=1.0,
        target_shape=(5, 5),
        initial_refinement="peak_component",
    )

    assert pred[1, 1]
    assert not pred[3, 3]


def test_model_prompt_conditioned_sam3_head_resizes_coarse_prompt_to_feature_shape():
    seen: dict[str, torch.Tensor] = {}

    class _CaptureHead(torch.nn.Module):
        def forward(self, feature_map, prompt_embedding, coarse_prompt):
            seen["coarse"] = coarse_prompt.detach().cpu()
            return torch.zeros((1, 1, *feature_map.shape[-2:]), device=feature_map.device)

    coarse = np.zeros((4, 4), dtype=bool)
    coarse[1, 1] = True

    refined, report = refine_mask_with_prompt_conditioned_sam3_head(
        feature_map=torch.zeros(1, 2, 8, 8),
        prompt_embedding=torch.zeros(3),
        coarse_mask=coarse,
        head=_CaptureHead(),
        logit_threshold=0.0,
        min_initial_iou=0.0,
        coarse_dilate=0,
        coarse_threshold=0.5,
    )

    assert refined.shape == coarse.shape
    assert report["coarse_prompt_resized"] is True
    assert report["coarse_prompt_input_shape"] == [4, 4]
    assert report["coarse_prompt_shape"] == [8, 8]
    assert tuple(seen["coarse"].shape[-2:]) == (8, 8)
    assert int(seen["coarse"].sum().item()) == 4
