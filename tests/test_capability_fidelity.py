import pytest
import torch

from radio_gs.evaluation.capability_fidelity import (
    dense_cosine_values,
    local_affinity_pairs,
    relation_fidelity_summary,
    select_query_free_compositor,
)


def test_identical_dense_features_have_perfect_cosine_and_relations():
    torch.manual_seed(1)
    features = torch.randn(4, 3, 5)
    valid = torch.ones(3, 5, dtype=torch.bool)

    cosine = dense_cosine_values(features, features, valid)
    affinity = local_affinity_pairs(features, valid)
    relation = relation_fidelity_summary(affinity, affinity)

    torch.testing.assert_close(cosine, torch.ones_like(cosine))
    assert relation["affinity_mae"] == 0.0
    assert relation["affinity_pearson"] == pytest.approx(1.0)
    assert relation["boundary_margin_retention"] == pytest.approx(1.0)


def test_local_affinity_respects_valid_edges():
    features = torch.ones(2, 2, 2)
    valid = torch.tensor([[True, True], [False, True]])

    affinity = local_affinity_pairs(features, valid)

    assert affinity.numel() == 2
    torch.testing.assert_close(affinity, torch.ones(2))


def _compositor_report(mean=0.8, p05=0.6, pearson=0.2, retention=0.3):
    report = {
        space: {
            "mean_cosine": mean,
            "p05_cosine": p05,
            "local_relation": {
                "affinity_pearson": pearson,
                "boundary_margin_retention": retention,
            },
        }
        for space in ("raw_radio", "official_dino_v3", "official_sam3")
    }
    report["support_fraction_on_visible"] = 1.0
    return report


def test_compositor_selection_accepts_relation_gain_with_dense_guard():
    reports = {
        "alpha_mean": _compositor_report(),
        "gamma_1.5": _compositor_report(
            mean=0.797, p05=0.595, pearson=0.23, retention=0.33
        ),
        "top1": _compositor_report(
            mean=0.78, p05=0.55, pearson=0.4, retention=0.5
        ),
    }
    decision = select_query_free_compositor(reports)
    assert decision["selected_variant"] == "gamma_1.5"
    assert not decision["candidates"]["top1"]["eligible"]


def test_compositor_selection_requires_explicit_visible_support() -> None:
    report = _compositor_report()
    report.pop("support_fraction_on_visible")

    with pytest.raises(ValueError, match="visible support"):
        select_query_free_compositor(
            {
                "alpha_mean": _compositor_report(),
                "missing": report,
            }
        )


def test_compositor_selection_fails_closed_when_all_support_is_too_low() -> None:
    reports = {
        "alpha_mean": _compositor_report(),
        "candidate": _compositor_report(
            pearson=0.4,
            retention=0.5,
        ),
    }
    for report in reports.values():
        report["support_fraction_on_visible"] = 0.90

    decision = select_query_free_compositor(reports)

    assert decision["selected_variant"] is None
    assert not decision["promotion_allowed"]
    assert (
        decision["selection_status"]
        == "support_gate_failed_no_promotion"
    )
    assert all(
        not candidate["support_guard_passed"]
        for candidate in decision["candidates"].values()
    )


def test_compositor_selection_rejects_nonfinite_metrics() -> None:
    report = _compositor_report()
    report["official_dino_v3"]["mean_cosine"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        select_query_free_compositor(
            {
                "alpha_mean": _compositor_report(),
                "invalid": report,
            }
        )
