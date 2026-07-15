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
    return {
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
