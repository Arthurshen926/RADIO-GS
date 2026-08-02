from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from radio_gs.models.surface_text_response_adapter import (
    LowRankTangentSummaryAdapter,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_pilot as pilot,
)


def _sha(character: str) -> str:
    return character * 64


def test_pilot_contract_is_fixed_seed0_fit_only_and_continuous() -> None:
    contract = pilot._training_contract()

    assert contract["seed"] == 0
    assert contract["adapter"] == {
        "rank": 32,
        "max_angle_degrees": 0.1,
        "location": "before_frozen_official_siglip_summary_head",
    }
    assert contract["base_surface_readout"] == "frozen_exact_seed0_control"
    assert contract["official_siglip_summary_head"] == "frozen"
    assert contract["objective"]["vocabulary"] == "target_blind_fit_only"
    assert (
        contract["objective"]["teacher_side_autograd"]
        == "detached_teacher_variance_weights_text_bank"
    )
    assert contract["selector"]["tail_fraction"] == 0.10
    assert contract["selector"]["discrete_rank_metrics_are_feasibility_gates"] is False


def test_pilot_cli_has_no_seed_or_external_benchmark_selection() -> None:
    parser = pilot.build_arg_parser()
    destinations = {action.dest for action in parser._actions}

    assert "seed" not in destinations
    assert "canonical_task_id" not in destinations
    assert "registry_row" not in destinations
    assert "benchmark" not in destinations


def test_combined_state_hash_binds_all_three_components() -> None:
    value = pilot.combined_state_sha256(_sha("a"), _sha("b"), _sha("c"))
    expected = hashlib.sha256(
        (
            '{"base_surface_state_dict_sha256":"'
            + _sha("a")
            + '","response_adapter_architecture_digest":"'
            + _sha("c")
            + '","response_adapter_state_dict_sha256":"'
            + _sha("b")
            + '"}'
        ).encode("utf-8")
    ).hexdigest()
    assert value == expected
    assert value != pilot.combined_state_sha256(_sha("a"), _sha("d"), _sha("c"))
    with pytest.raises(ValueError):
        pilot.combined_state_sha256("bad", _sha("b"), _sha("c"))


def test_adapter_angle_statistics_reports_mean_max_and_saturation() -> None:
    angle = torch.deg2rad(torch.tensor(0.1, dtype=torch.float64))
    base = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    adapted = torch.tensor(
        [
            [1.0, 0.0],
            [torch.cos(angle), torch.sin(angle)],
        ],
        dtype=torch.float64,
    )

    stats = pilot.adapter_angle_statistics(
        base, adapted, max_angle_degrees=0.1
    )

    assert stats["mean_degrees"] == pytest.approx(0.05, abs=1e-8)
    assert stats["max_degrees"] == pytest.approx(0.1, abs=1e-8)
    assert stats["saturation_ratio_at_99pct_cap"] == 0.5


def test_uncertainty_weight_statistics_are_query_free_and_formula_bound() -> None:
    teacher = torch.tensor(
        [[1.0, 0.0], [0.5, 0.75**0.5], [0.0, 1.0]]
    )
    views = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.8, 0.6], [0.2, 0.96**0.5]],
            [[0.0, 1.0], [0.0, -1.0]],
        ]
    )
    mask = torch.ones(3, 2, dtype=torch.bool)

    stats = pilot.uncertainty_weight_statistics(
        teacher,
        views,
        mask,
        torch.tensor([[1.0, 0.0]]),
        ["fit_scene"] * 3,
    )

    noisy_weight = 0.5 / (0.5 + 2.0 * 0.3 + 1e-6)
    stable_weight = 1.0 / (1.0 + 1e-6)
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(
        (2.0 * noisy_weight + stable_weight) / 3.0, rel=1e-5
    )
    assert stats["fraction_below_0p5"] == pytest.approx(2.0 / 3.0)


def test_continuous_selector_uses_paired_global_and_per_scene_cvar() -> None:
    control = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]])
    candidate = control - 0.1
    valid = torch.ones_like(control, dtype=torch.bool)

    selector, delta = pilot.continuous_selector_metrics(
        candidate,
        valid,
        control,
        valid,
        ["scene_a", "scene_b"],
    )

    expected_delta = -0.1 / control.mean().item()
    assert torch.allclose(delta, torch.full_like(delta, expected_delta))
    assert selector["normalized_mean_delta"] == pytest.approx(expected_delta)
    assert selector["normalized_upper_cvar10_delta"] == pytest.approx(
        expected_delta
    )
    assert selector["worst_scene_upper_cvar10_delta"] == pytest.approx(
        expected_delta
    )
    assert set(selector["per_scene"]) == {"scene_a", "scene_b"}


def test_selector_requires_surface_and_continuous_feasibility_then_selects_best() -> None:
    surface = {
        "summary_token_cosine": 0.9,
        "mean_descriptor_cosine": 0.9,
        "all_view_descriptor_cosine": 0.9,
        "surface_selection_score": 0.9,
        "text_response_smooth_l1": 0.1,
        "text_response_mae": 0.2,
    }
    control_selector = {
        "candidate_mean_unit_loss": 1.0,
        "normalized_mean_delta": 0.0,
        "normalized_upper_cvar10_delta": 0.0,
        "worst_scene_mean_delta": 0.0,
        "worst_scene_upper_cvar10_delta": 0.0,
    }
    control = pilot.annotate_selection_record(
        {"epoch": 0, **surface},
        control_record=surface,
        selector=control_selector,
    )
    improved_selector = {
        **control_selector,
        "candidate_mean_unit_loss": 0.8,
        "normalized_mean_delta": -0.2,
        "normalized_upper_cvar10_delta": -0.1,
        "worst_scene_mean_delta": -0.1,
        "worst_scene_upper_cvar10_delta": -0.05,
    }
    candidate = pilot.annotate_selection_record(
        {"epoch": 1, **surface, "text_response_smooth_l1": 0.08},
        control_record=control,
        selector=improved_selector,
    )

    assert control["selection_feasible"] is True
    assert candidate["selection_feasible"] is True
    assert pilot.select_best_epoch([control, candidate]) == 1

    damaged = pilot.annotate_selection_record(
        {
            "epoch": 1,
            **surface,
            "summary_token_cosine": 0.897,
            "text_response_smooth_l1": 0.01,
        },
        control_record=control,
        selector=improved_selector,
    )
    assert damaged["surface_control_feasible"] is False
    assert pilot.select_best_epoch([control, damaged]) == 0


class _IdentitySummaryHead(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def test_cpu_evaluation_emits_continuous_units_surface_and_angle_stats() -> None:
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=2,
        rank=1,
        max_angle_degrees=0.1,
    )
    base_tokens = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]]
    )
    views = torch.stack(
        [
            torch.stack([value, value])
            for value in base_tokens
        ]
    )
    data = {
        "radio_features": torch.zeros(4, 1, 2),
        "official_summary_tokens": views.clone(),
        "official_crop_summaries": views.clone(),
        "teacher_mask": torch.ones(4, 2, dtype=torch.bool),
        "scene_ids": ["a", "a", "b", "b"],
    }

    metrics, units, valid = pilot._evaluate(
        adapter,
        _IdentitySummaryHead(),
        base_tokens,
        data,
        torch.eye(2),
    )

    assert metrics["summary_token_cosine"] == pytest.approx(1.0)
    assert metrics["mean_descriptor_cosine"] == pytest.approx(1.0)
    assert metrics["adapter_angle"]["max_degrees"] < 1e-6
    assert units.shape == (2, 2)
    assert valid.shape == units.shape
    assert bool(valid.all())
    assert torch.equal(units, torch.zeros_like(units))

