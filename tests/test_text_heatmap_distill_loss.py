import torch
import torch.nn.functional as F

from radio_gs.losses.text_heatmap_distill_loss import compute_text_heatmap_distill_loss


def test_text_heatmap_distill_loss_is_zero_for_identical_projected_features():
    features = F.normalize(torch.randn(2, 4, 3, 5), dim=1)
    text = F.normalize(torch.randn(6, 4), dim=1)

    loss, stats = compute_text_heatmap_distill_loss(features, features.clone(), text)

    assert loss.item() == 0.0
    assert stats["num_queries"] == 6


def test_text_heatmap_distill_loss_detects_changed_query_responses():
    teacher = F.normalize(torch.randn(1, 4, 2, 2), dim=1)
    rendered = teacher.clone()
    rendered[:, 0] *= -1.0
    rendered = F.normalize(rendered, dim=1)
    text = F.normalize(torch.randn(5, 4), dim=1)

    loss, _ = compute_text_heatmap_distill_loss(
        rendered,
        teacher,
        text,
        temperature=10.0,
    )

    assert loss.item() > 1.0e-4


def test_text_heatmap_distill_loss_can_downsample_spatial_maps():
    teacher = F.normalize(torch.randn(1, 4, 8, 8), dim=1)
    rendered = teacher + 0.1 * torch.randn_like(teacher)
    text = F.normalize(torch.randn(3, 4), dim=1)

    loss, stats = compute_text_heatmap_distill_loss(
        rendered,
        teacher,
        text,
        downsample=2,
    )

    assert torch.isfinite(loss)
    assert stats["height"] == 4
    assert stats["width"] == 4


def test_spatial_text_heatmap_distill_detects_peak_shift_with_single_query():
    teacher = torch.tensor(
        [
            [
                [[1.0, -1.0], [-1.0, -1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    rendered = torch.tensor(
        [
            [
                [[-1.0, -1.0], [-1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    text = torch.tensor([[1.0, 0.0]])

    query_loss, _ = compute_text_heatmap_distill_loss(
        rendered,
        teacher,
        text,
        temperature=10.0,
        mode="query",
    )
    spatial_loss, stats = compute_text_heatmap_distill_loss(
        rendered,
        teacher,
        text,
        temperature=10.0,
        mode="spatial",
    )

    assert query_loss.item() == 0.0
    assert spatial_loss.item() > 1.0
    assert stats["mode"] == "spatial"


def test_text_heatmap_distill_rejects_unknown_mode():
    features = F.normalize(torch.randn(1, 4, 3, 3), dim=1)
    text = F.normalize(torch.randn(2, 4), dim=1)

    try:
        compute_text_heatmap_distill_loss(features, features, text, mode="unknown")
    except ValueError as exc:
        assert "mode must be one of" in str(exc)
    else:
        raise AssertionError("expected ValueError")
