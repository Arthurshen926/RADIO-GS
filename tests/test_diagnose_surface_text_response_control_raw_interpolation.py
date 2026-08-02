from __future__ import annotations

import math

import pytest
import torch

from radio_gs.scripts import (
    diagnose_surface_text_response_control_raw_interpolation as diagnostic,
)


class _ScalarModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        del geometry
        return radio_features * self.weight


class _IdentityHead(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _response_metrics(alpha: float) -> dict:
    scene = {
        "profile_cosine_mean": 0.8,
        "profile_cosine_p05": 0.7,
        "ranking_spearman_mean": 0.6,
        "ranking_spearman_p05": 0.5,
        "top_decile_overlap_mean": 0.7,
        "top_decile_overlap_p05": 0.6,
        "smooth_l1": 0.10 - alpha,
        "mae": 0.20 - alpha,
    }
    return {
        "text_support_top1_agreement": 0.75,
        "text_support_valid_query_ratio": 1.0,
        "text_response_smooth_l1": scene["smooth_l1"],
        "text_response_mae": scene["mae"],
        "text_response_profile_cosine_mean": 0.8,
        "text_response_profile_cosine_p05": 0.7,
        "text_response_ranking_spearman_mean": 0.6,
        "text_response_ranking_spearman_p05": 0.5,
        "text_response_top_decile_overlap_mean": 0.7,
        "text_response_top_decile_overlap_p05": 0.6,
        "text_response_scene_worst_smooth_l1": scene["smooth_l1"],
        "text_response_scene_worst_mae": scene["mae"],
        "text_response_scene_worst_profile_cosine_mean": 0.8,
        "text_response_scene_worst_profile_cosine_p05": 0.7,
        "text_response_scene_worst_ranking_spearman_mean": 0.6,
        "text_response_scene_worst_ranking_spearman_p05": 0.5,
        "text_response_scene_worst_top_decile_overlap_mean": 0.7,
        "text_response_scene_worst_top_decile_overlap_p05": 0.6,
        "text_response_scene_metrics": {"scene-a": scene},
        "descriptor_relation_smooth_l1": 0.02,
    }


def _required_cli(seed: str) -> list[str]:
    return [
        "--train-caches", "train*.pt",
        "--validation-caches", "validation*.pt",
        "--fit-text-bank", "fit.pt",
        "--fit-text-bank-manifest", "fit.json",
        "--calibration-manifest", "calibration.json",
        "--surface-control-checkpoint", "surface.pt",
        "--surface-control-checkpoint-sha256", "a" * 64,
        "--output", "result.json",
        "--seed", seed,
    ]


def test_seed_contract_is_explicit_and_fail_closed() -> None:
    assert tuple(diagnostic.validated_seed(seed) for seed in (0, 1, 2)) == (
        0,
        1,
        2,
    )
    for invalid in (-1, 3, True, "0"):
        with pytest.raises(ValueError, match="one of 0/1/2"):
            diagnostic.validated_seed(invalid)
    assert diagnostic._parser().parse_args(_required_cli("2")).seed == 2
    with pytest.raises(SystemExit):
        diagnostic._parser().parse_args(_required_cli("3"))
    assert "seed0" not in diagnostic.ARTIFACT_TYPE
    assert "seed0" not in diagnostic.ALGORITHM_VERSION


def test_interpolate_state_dict_interpolates_float_and_rejects_discrete_drift() -> None:
    control = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    raw = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    value = diagnostic.interpolate_state_dict(control, raw, 0.005)
    assert torch.allclose(value["weight"], torch.tensor([1.02, 3.02]))
    assert value["counter"].item() == 2
    assert diagnostic.state_dict_binding(value)["tensor_count"] == 2
    displacement = diagnostic.parameter_displacement_binding(
        control, raw, ("weight",)
    )
    assert displacement["overall"]["delta_l2"] == pytest.approx(
        math.sqrt(32.0)
    )
    assert displacement["overall"]["max_abs"] == 4.0
    assert set(displacement["top_level_modules"]) == {"weight"}

    forged = dict(raw)
    forged["counter"] = torch.tensor(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="non-floating"):
        diagnostic.interpolate_state_dict(control, forged, 0.005)


def test_grid_really_evaluates_every_alpha_and_reuses_robust_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ScalarModel(0.0)
    evaluations: list[float] = []

    def evaluate(*args: object, **kwargs: object) -> tuple[dict, dict]:
        del args, kwargs
        alpha = float(model.weight.detach())
        evaluations.append(alpha)
        surface = {
            "summary_token_cosine": 0.9,
            "mean_descriptor_cosine": 0.8,
            "all_view_descriptor_cosine": 0.8,
        }
        return surface, _response_metrics(alpha)

    monkeypatch.setattr(diagnostic.trainer, "_evaluate_response_aware", evaluate)
    history, best_index, best_score = diagnostic.evaluate_interpolation_grid(
        model,
        _IdentityHead(),
        {},
        device=torch.device("cpu"),
        batch_size=2,
        text_bank=torch.ones(1, 2),
        control_state={"weight": torch.tensor(0.0)},
        raw_state={"weight": torch.tensor(1.0)},
    )

    assert evaluations == pytest.approx(list(diagnostic.ALPHA_GRID), abs=1e-8)
    assert len(history) == len(diagnostic.ALPHA_GRID)
    assert best_index == len(diagnostic.ALPHA_GRID) - 1
    assert history[best_index]["alpha"] == diagnostic.ALPHA_GRID[-1]
    assert history[best_index]["response_selection_feasible"] is True
    assert history[best_index]["selection_score"] == best_score
    assert history[best_index]["control_to_candidate_parameter_radius"][
        "delta_l2"
    ] == pytest.approx(diagnostic.ALPHA_GRID[-1])
    assert all(
        "fit_response_control_deltas" in row
        and "text_response_scene_metrics" in row
        for row in history
    )


def test_single_raw_proposal_runs_one_complete_adamw_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ScalarModel(0.5)
    train_data = {
        "scene_ids": ["scene-a", "scene-a"],
        "radio_features": torch.ones(2, 2),
        "geometry": torch.zeros(2, 1),
        "anchor_index": torch.zeros(2, dtype=torch.long),
        "token_mask": torch.ones(2, 2, dtype=torch.bool),
        "reliability": torch.ones(2, 2),
    }

    def targets(data: object, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        del data
        count = len(rows)
        return (
            torch.zeros(count, 2),
            torch.zeros(count, 2),
            torch.zeros(count, 1, 2),
            torch.ones(count, 1, dtype=torch.bool),
        )

    def losses(predicted: torch.Tensor, *args: object, **kwargs: object) -> dict:
        del args, kwargs
        total = predicted.square().mean()
        return {
            "total": total,
            "token": total,
            "descriptor": total,
            "relation": total,
            "independent_response": total,
            "scene_response": total,
            "scene_profile": total.detach(),
            "scene_ranking": total.detach(),
        }

    monkeypatch.setattr(diagnostic.trainer, "_targets", targets)
    monkeypatch.setattr(
        diagnostic.trainer,
        "inject_tangent_direction_noise",
        lambda features, mask, angle_degrees: features,
    )
    monkeypatch.setattr(diagnostic.trainer, "compute_training_losses", losses)
    before = float(model.weight.detach())
    result = diagnostic.train_single_raw_proposal(
        model,
        _IdentityHead(),
        train_data,
        device=torch.device("cpu"),
        text_bank=torch.ones(1, 2),
        response_lambdas={"independent_response": 1.0, "scene_response": 1.0},
        generator=torch.Generator().manual_seed(0),
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        token_weight=0.25,
        relation_weight=0.1,
        canonical_noise_degrees=0.0,
    )

    assert result["epoch_count"] == 1
    assert result["complete_scene_batch_count"] == 1
    assert set(result["mean_losses"]) == {
        "total",
        "token",
        "descriptor",
        "relation",
        "independent_response",
        "scene_response",
        "scene_profile",
        "scene_ranking",
    }
    assert math.isfinite(result["mean_losses"]["total"])
    assert float(model.weight.detach()) != before
