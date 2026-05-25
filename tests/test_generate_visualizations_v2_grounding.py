from pathlib import Path

import cv2
import numpy as np
import torch

from radio_gs.scripts import generate_visualizations_v2 as viz
from radio_gs.scripts.eval_lerf_grounding import (
    build_sam3_prompt_initial_mask,
    compute_iou,
    compute_relevancy_heatmap,
    heatmap_peak_in_shape,
    keep_peak_connected_component,
    project_to_siglip2,
    refine_mask_with_rgb_edges,
    resolve_heatmap_threshold_ratio,
    save_heatmap_vis,
)


class _MarkerProjection(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return x


def test_grounding_projection_loader_prefers_summary_head(monkeypatch, tmp_path):
    summary_path = tmp_path / "summary.pth"
    summary_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        viz.SigLIP2SummaryHead,
        "from_extracted_weights",
        classmethod(lambda cls, path: _MarkerProjection(f"summary:{Path(path).name}")),
    )

    projection = viz.load_siglip2_projection(
        "unused_projection.pth",
        target_device=torch.device("cpu"),
        use_summary_head=True,
        summary_head_weights=str(summary_path),
    )

    assert projection.mode == "summary:summary.pth"


def test_grounding_projection_loader_can_use_spatial_projection(monkeypatch, tmp_path):
    projection_path = tmp_path / "projection.pth"
    projection_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        viz.SigLIP2FeatureProjection,
        "from_extracted_weights",
        classmethod(lambda cls, path: _MarkerProjection(f"projection:{Path(path).name}")),
    )

    projection = viz.load_siglip2_projection(
        str(projection_path),
        target_device=torch.device("cpu"),
        use_summary_head=False,
    )

    assert projection.mode == "projection:projection.pth"


def test_eval_compatible_selection_uses_sorted_scene_categories():
    selection = viz.build_eval_compatible_grounding_selection(
        categories=["banana", "apple", "cup", "donut"],
        scene_categories=["donut", "apple", "missing"],
        requested_queries=["donut", "banana", "apple", "missing"],
    )

    assert selection.active_queries == ["apple", "donut"]
    assert selection.scene_categories == ["apple", "donut"]
    assert selection.active_indices == [1, 3]
    assert selection.scene_indices == [1, 3]
    assert selection.active_scene_indices == [0, 1]


def test_eval_compatible_grounding_heatmaps_match_formal_scene_softmax():
    class FixedProjection(torch.nn.Module):
        def forward(self, features):
            out = torch.zeros(features.shape[0], features.shape[1], 3)
            out[..., 0] = 1.0
            return out

    features = torch.zeros(1, 1280, 1, 1)
    text = torch.eye(3)
    selection = viz.build_eval_compatible_grounding_selection(
        categories=["apple", "banana", "cup"],
        scene_categories=["apple", "banana", "cup"],
        requested_queries=["banana"],
    )

    heatmaps = viz.compute_eval_compatible_grounding_heatmaps(
        features,
        FixedProjection(),
        text,
        selection,
        temperature=1.0,
        scoring="softmax_scene",
        target_device=torch.device("cpu"),
    )

    siglip_feat = project_to_siglip2(features, FixedProjection())
    expected = compute_relevancy_heatmap(
        siglip_feat,
        text[selection.active_indices],
        temperature=1.0,
        scoring="softmax_scene",
        all_scene_emb=text[selection.scene_indices],
        active_scene_indices=selection.active_scene_indices,
    )

    assert torch.allclose(heatmaps, expected)
    assert heatmaps.shape == (1, 1, 1)
    assert heatmaps[0, 0, 0].item() < 0.25


def test_adaptive_sam3_prompt_initial_mask_keeps_raw_when_peak_loses_heatmap_mass():
    heatmap = torch.zeros(8, 8)
    heatmap[1:3, 1:3] = 0.95
    heatmap[5:7, 5:7] = 0.90
    heatmap[1, 1] = 1.0

    mask = build_sam3_prompt_initial_mask(
        heatmap,
        threshold_ratio=0.5,
        threshold_mode="fixed",
        threshold_mean_std_k=0.0,
        threshold_min_ratio=0.0,
        threshold_max_ratio=1.0,
        target_shape=(8, 8),
        initial_refinement="adaptive_peak",
    )

    assert mask[1:3, 1:3].all()
    assert mask[5:7, 5:7].all()


def test_adaptive_sam3_prompt_initial_mask_uses_peak_when_distractor_has_low_support():
    heatmap = torch.zeros(8, 8)
    heatmap[1:4, 1:4] = 0.95
    heatmap[1, 1] = 1.0
    heatmap[6, 6] = 0.51

    mask = build_sam3_prompt_initial_mask(
        heatmap,
        threshold_ratio=0.5,
        threshold_mode="fixed",
        threshold_mean_std_k=0.0,
        threshold_min_ratio=0.0,
        threshold_max_ratio=1.0,
        target_shape=(8, 8),
        initial_refinement="adaptive_peak",
    )

    assert mask[1:4, 1:4].all()
    assert not bool(mask[6, 6])


def test_project_to_siglip2_accepts_half_features_on_cpu():
    projection = torch.nn.Linear(1280, 3).cpu().float()
    features = torch.zeros(1, 1280, 1, 1, dtype=torch.float16)

    projected = project_to_siglip2(features, projection)

    assert projected.device.type == "cpu"
    assert projected.dtype == torch.float32
    assert projected.shape == (1, 3, 1, 1)


def test_refine_mask_with_rgb_edges_snaps_grounding_mask_to_boundary():
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


def test_compute_iou_can_use_rgb_edge_refinement():
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[:, :] = (20, 20, 180)
    rgb[16:32, 16:32] = (180, 20, 20)
    gt = np.zeros((48, 48), dtype=np.uint8)
    gt[16:32, 16:32] = 1
    heatmap = torch.zeros(48, 48)
    heatmap[10:38, 10:38] = 1.0

    plain_iou = compute_iou(heatmap, gt, threshold_ratio=0.5)
    refined_iou = compute_iou(
        heatmap,
        gt,
        threshold_ratio=0.5,
        rgb_image=rgb,
        mask_refinement="rgb_grabcut",
        mask_refinement_iters=2,
        mask_refinement_dilate=4,
        mask_refinement_erode=3,
    )

    assert refined_iou > plain_iou


def test_keep_peak_connected_component_removes_disconnected_heatmap_false_positive():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask[10:15, 10:15] = 1

    kept = keep_peak_connected_component(mask, (3, 3))

    assert kept.sum() == 16
    assert kept[2:6, 2:6].all()
    assert kept[10:15, 10:15].sum() == 0


def test_compute_iou_peak_component_readout_keeps_query_peak_region():
    heatmap = torch.zeros(16, 16)
    heatmap[2:6, 2:6] = 1.0
    heatmap[10:15, 10:15] = 0.8
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[2:6, 2:6] = 1

    plain_iou = compute_iou(heatmap, gt, threshold_ratio=0.5)
    refined_iou = compute_iou(
        heatmap,
        gt,
        threshold_ratio=0.5,
        mask_refinement="peak_component",
    )

    assert heatmap_peak_in_shape(heatmap, gt.shape) == (2, 2)
    assert refined_iou == 1.0
    assert refined_iou > plain_iou


def test_resolve_heatmap_threshold_ratio_supports_mean_std_mode():
    heatmap = torch.tensor(
        [
            [0.0, 0.1, 0.1, 0.2],
            [0.0, 0.1, 0.8, 1.0],
        ],
        dtype=torch.float32,
    )

    fixed = resolve_heatmap_threshold_ratio(heatmap, 0.6, mode="fixed")
    adaptive = resolve_heatmap_threshold_ratio(
        heatmap,
        0.6,
        mode="mean_std",
        mean_std_k=1.0,
        min_ratio=0.3,
        max_ratio=0.9,
    )

    assert fixed == 0.6
    assert 0.3 <= adaptive <= 0.9
    assert adaptive != fixed


def test_compute_iou_accepts_adaptive_threshold_mode():
    heatmap = torch.zeros(16, 16)
    heatmap[4:12, 4:12] = 0.4
    heatmap[6:10, 6:10] = 1.0
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[6:10, 6:10] = 1

    adaptive_iou = compute_iou(
        heatmap,
        gt,
        threshold_ratio=0.5,
        threshold_mode="mean_std",
        threshold_mean_std_k=1.0,
        threshold_min_ratio=0.3,
        threshold_max_ratio=0.9,
    )

    assert adaptive_iou > 0.0


def test_lerf_overlay_preserves_rgb_aspect_ratio(tmp_path):
    heatmaps = {"apple": np.ones((2, 2), dtype=np.float32)}
    masks = {"apple": np.array([[0, 1], [1, 0]], dtype=np.uint8)}
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)

    save_heatmap_vis(
        heatmaps,
        masks,
        frame_id=7,
        out_dir=tmp_path,
        tag="rendered",
        rgb_image=rgb,
    )

    image = cv2.imread(str(tmp_path / "lerf_grounding_frame_00007_rendered.png"))

    assert image is not None
    assert image.shape[1] == 6 * rgb.shape[1]
    assert image.shape[0] == 28 + rgb.shape[0]


def test_lerf_overlay_can_write_per_query_files(tmp_path):
    heatmaps = {
        "apple": np.ones((2, 2), dtype=np.float32),
        "red cup": np.zeros((2, 2), dtype=np.float32),
    }
    masks = {
        "apple": np.ones((2, 2), dtype=np.uint8),
        "red cup": np.ones((2, 2), dtype=np.uint8),
    }
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)

    save_heatmap_vis(
        heatmaps,
        masks,
        frame_id=7,
        out_dir=tmp_path,
        tag="rendered",
        rgb_image=rgb,
        save_per_query=True,
    )

    assert (tmp_path / "lerf_grounding_frame_00007_rendered_apple.png").exists()
    assert (tmp_path / "lerf_grounding_frame_00007_rendered_red_cup.png").exists()
