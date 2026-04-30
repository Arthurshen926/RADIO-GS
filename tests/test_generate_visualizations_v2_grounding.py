from pathlib import Path

import torch

from radio_gs.scripts import generate_visualizations_v2 as viz
from radio_gs.scripts.eval_lerf_grounding import compute_relevancy_heatmap, project_to_siglip2


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
