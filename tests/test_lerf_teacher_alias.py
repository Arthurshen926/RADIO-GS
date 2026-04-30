import torch

from radio_gs.scripts.eval_lerf_grounding import (
    canonical_lerf_mode,
    display_lerf_mode,
    iter_lerf_report_modes,
    lerf_mode_tag,
)
from radio_gs.scripts.generate_visualizations_v2 import compute_grounding_heatmaps


def test_teacher_mode_preserves_gt_compatibility_alias() -> None:
    assert canonical_lerf_mode("gt") == "teacher"
    assert canonical_lerf_mode("teacher") == "teacher"
    assert lerf_mode_tag("gt") == "teacher"
    assert display_lerf_mode("gt") == "TEACHER RADIO features"


def test_report_modes_emit_teacher_before_rendered() -> None:
    assert iter_lerf_report_modes({"gt": {}, "rendered": {}}) == ["teacher", "rendered"]
    assert iter_lerf_report_modes({"teacher": {}, "rendered": {}}) == ["teacher", "rendered"]


def test_visualization_grounding_temperature_matches_eval_logit_scale() -> None:
    class FixedProjection(torch.nn.Module):
        def forward(self, features):
            out = torch.zeros(features.shape[0], features.shape[1], 1536)
            out[..., 0] = 1.0
            return out

    features = torch.zeros(1, 1280, 1, 1)
    text = torch.zeros(2, 1536)
    text[0, 0] = 1.0
    text[1, 1] = 1.0

    _, probs = compute_grounding_heatmaps(
        features,
        FixedProjection(),
        text,
        temperature=50.0,
        target_device=torch.device("cpu"),
    )

    assert probs[0, 0, 0].item() > 0.99
    assert probs[1, 0, 0].item() < 0.01
