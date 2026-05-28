import torch

from radio_gs.models.point_summary_adapter import (
    append_point_summary_context,
    parse_point_summary_context_features,
    point_summary_context_dim,
)


def test_parse_point_summary_context_features_accepts_common_separators():
    assert parse_point_summary_context_features("opacity, scale_log_mean view_count") == (
        "opacity",
        "scale_log_mean",
        "view_count",
    )


def test_point_summary_context_dim_counts_enabled_scalars():
    assert point_summary_context_dim("opacity scale_log_mean scale_log_max view_count") == 4


def test_append_point_summary_context_adds_bounded_geometry_and_visibility():
    compact = torch.zeros(3, 2)
    opacity = torch.tensor([[0.0], [0.5], [2.0]])
    scales = torch.tensor(
        [
            [0.01, 0.02, 0.04],
            [0.10, 0.20, 0.40],
            [1.00, 2.00, 4.00],
        ]
    )
    view_counts = torch.tensor([0.0, 3.0, 15.0])

    out = append_point_summary_context(
        compact,
        context_features="opacity scale_log_mean scale_log_max view_count",
        opacity=opacity,
        scales=scales,
        view_counts=view_counts,
        view_count_max=15.0,
    )

    assert out.shape == (3, 6)
    assert torch.allclose(out[:, :2], compact)
    assert torch.all(out[:, 2:] <= 1.0)
    assert torch.all(out[:, 2:] >= -1.0)
    assert out[0, -1] == 0.0
    assert out[-1, -1] == 1.0


def test_append_point_summary_context_requires_requested_inputs():
    compact = torch.zeros(2, 4)
    try:
        append_point_summary_context(compact, context_features="view_count")
    except ValueError as exc:
        assert "view_counts" in str(exc)
    else:
        raise AssertionError("expected missing view_counts to raise")
