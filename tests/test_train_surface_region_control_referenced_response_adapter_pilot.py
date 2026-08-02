from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from radio_gs.models.surface_text_response_adapter import (
    LowRankTangentSummaryAdapter,
)
from radio_gs.scripts import (
    train_surface_region_control_referenced_response_adapter_pilot as v3,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_cvar_pilot as v2,
)


class _IdentitySummaryHead(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def _synthetic_data() -> tuple[torch.Tensor, dict[str, object]]:
    base = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]])
    teacher = torch.tensor([[0.98, 0.20], [0.72, 0.69], [0.20, 0.98], [-0.50, 0.86]])
    views = torch.stack([torch.stack([value, value]) for value in teacher])
    data: dict[str, object] = {
        "radio_features": torch.zeros(4, 1, 2),
        "official_summary_tokens": views.clone(),
        "official_crop_summaries": views.clone(),
        "teacher_mask": torch.ones(4, 2, dtype=torch.bool),
        "scene_ids": ["a", "a", "b", "b"],
    }
    return base, data


def test_v3_contract_is_full_fit_control_referenced_and_fixed() -> None:
    contract = v3.training_contract_v3()
    objective = contract["objective"]

    assert contract["seed"] == 0
    assert contract["adapter"] == {
        "rank": 32,
        "max_angle_degrees": 0.1,
        "location": "before_frozen_official_siglip_summary_head",
    }
    assert objective["reduction_domain"] == (
        "one_complete_32_scene_fit_set_before_each_optimizer_proposal"
    )
    assert objective["constraint_penalty"] == {
        "kind": "unsmoothed_l1_hinge",
        "fixed_weight": 1.0,
        "mathematical_exact_penalty_guarantee": False,
        "reason": (
            "the fixed empirical multiplier is preregistered but no unknown "
            "optimal Lagrange-multiplier bound is claimed"
        ),
    }
    assert objective["surface_term_in_training_objective"] is False
    assert objective["surface_role"] == ("unchanged_validation_noninferiority_gate")
    assert contract["trust_backtracking"]["maximum_saturation_ratio"] == 0.05
    assert contract["trust_backtracking"]["feasibility_is_mandatory"] is True
    assert contract["optimizer_state_on_full_rejection"].startswith("restore")
    assert contract["v3_boundaries_preserved"]["surface_gate_unchanged"] is True
    assert contract["v3_boundaries_preserved"]["scope"] == (
        "external_benchmarks_unopened"
    )
    assert len(contract["advance_gate"]["eight_checks"]) == 8
    assert contract["advance_gate"]["seed1_required_only_after_seed0_pass"]


def test_v3_cli_is_seed0_fit_only_without_benchmark_selector() -> None:
    destinations = {action.dest for action in v3.build_arg_parser()._actions}
    assert "seed" not in destinations
    assert "canonical_task_id" not in destinations
    assert "registry_row" not in destinations
    assert "benchmark" not in destinations
    assert "device" in destinations


def test_full_fit_objective_uses_all_scenes_and_matches_identity_control() -> None:
    base, data = _synthetic_data()
    adapter = LowRankTangentSummaryAdapter(feature_dim=2, rank=1, max_angle_degrees=0.1)
    head = _IdentitySummaryHead()
    text = torch.eye(2)
    control_metrics, control_units, control_valid = v2._evaluate_v2(
        adapter, head, base, data, text
    )
    assert control_metrics["text_response_smooth_l1"] > 0.0

    objective, stats, units, valid, unary = (
        v3.compute_full_fit_control_referenced_objective(
            adapter,
            head,
            base,
            data,
            text,
            control_units,
            control_valid,
            control_metrics["text_response_smooth_l1"],
        )
    )

    assert units.shape == valid.shape == (2, 2)
    torch.testing.assert_close(units.detach().cpu(), control_units)
    assert torch.equal(valid.detach().cpu(), control_valid)
    assert stats["scene_count"].item() == 2
    assert stats["global_mean_delta"].item() == pytest.approx(0.0)
    assert stats["independent_unary_delta"].item() == pytest.approx(0.0)
    assert objective.item() == pytest.approx(0.0)
    assert unary.item() == pytest.approx(control_metrics["text_response_smooth_l1"])
    objective.backward()
    assert adapter.up.weight.grad is not None
    assert torch.isfinite(adapter.up.weight.grad).all()


def test_fit_trust_backtracking_accepts_only_saturation_feasible_state() -> None:
    torch.manual_seed(7)
    adapter = LowRankTangentSummaryAdapter(feature_dim=4, rank=2, max_angle_degrees=0.1)
    base = torch.randn(32, 4)
    old = {name: value.detach().clone() for name, value in adapter.state_dict().items()}
    with torch.no_grad():
        adapter.up.weight.fill_(1.0)
    proposed = {
        name: value.detach().clone() for name, value in adapter.state_dict().items()
    }

    trust = v3.apply_fit_trust_backtracking(adapter, base, old, proposed)

    assert trust["unbacktracked_proposal_angle"]["saturation_ratio_at_99pct_cap"] > 0.05
    assert 0.0 < trust["accepted_parameter_displacement_fraction"] < 1.0
    assert trust["accepted_angle"]["saturation_ratio_at_99pct_cap"] <= 0.05
    assert trust["feasible"] is True
    assert trust["fully_rejected"] is False


def test_full_trust_rejection_restores_optimizer_state() -> None:
    torch.manual_seed(11)
    adapter = LowRankTangentSummaryAdapter(feature_dim=4, rank=2, max_angle_degrees=0.1)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    preproposal_optimizer = copy.deepcopy(optimizer.state_dict())
    old = {name: value.detach().clone() for name, value in adapter.state_dict().items()}

    adapter.up.weight.sum().backward()
    optimizer.step()
    assert optimizer.state_dict()["state"]
    with torch.no_grad():
        adapter.up.weight.fill_(1.0)
    proposed = {
        name: value.detach().clone() for name, value in adapter.state_dict().items()
    }
    base = torch.randn(16, 4)
    trust = v3.apply_fit_trust_backtracking(
        adapter,
        base,
        old,
        proposed,
        maximum_backtracking_steps=0,
    )
    assert trust["fully_rejected"] is True
    assert trust["accepted_parameter_displacement_fraction"] == 0.0

    action = v3.reconcile_optimizer_state_after_trust(
        optimizer, preproposal_optimizer, trust
    )

    assert action == "restored_exact_preproposal_state_after_full_rejection"
    assert optimizer.state_dict()["state"] == preproposal_optimizer["state"]
    for name, value in adapter.state_dict().items():
        torch.testing.assert_close(value, old[name])


def _metrics(*, smooth: float, mae: float) -> dict[str, object]:
    angle = {
        "mean_degrees": 0.01,
        "max_degrees": 0.02,
        "saturation_ratio_at_99pct_cap": 0.0,
    }
    return {
        "summary_token_cosine": 0.90,
        "mean_descriptor_cosine": 0.91,
        "all_view_descriptor_cosine": 0.92,
        "surface_selection_score": 0.915,
        "text_response_smooth_l1": smooth,
        "text_response_mae": mae,
        "adapter_angle": angle,
    }


def _selector(*, mean: float, cvar: float, worst_mean: float, worst_cvar: float):
    return {
        "control_scale_mean_unit_loss": 1.0,
        "candidate_mean_unit_loss": 1.0 + mean,
        "normalized_mean_delta": mean,
        "normalized_upper_cvar10_delta": cvar,
        "worst_scene_mean_delta": worst_mean,
        "worst_scene_upper_cvar10_delta": worst_cvar,
        "per_scene": {"fit": {"mean_delta": mean, "upper_cvar10_delta": cvar}},
        "unit_count": 8,
    }


def test_v3_eight_check_gate_selects_strict_candidate() -> None:
    fit_angle = _metrics(smooth=1.0, mae=1.0)["adapter_angle"]
    control_metrics = _metrics(smooth=1.0, mae=1.0)
    control = v3.annotate_v3_selection_record(
        {"epoch": 0, **control_metrics},
        control_record=control_metrics,
        selector=_selector(mean=0.0, cvar=0.0, worst_mean=0.0, worst_cvar=0.0),
        fit_angle=fit_angle,
    )
    candidate = v3.annotate_v3_selection_record(
        {"epoch": 1, **_metrics(smooth=0.99, mae=0.99)},
        control_record=control,
        selector=_selector(mean=-0.003, cvar=0.004, worst_mean=0.009, worst_cvar=0.009),
        fit_angle=fit_angle,
    )

    assert len(candidate["v3_advance_gate_checks"]) == 8
    assert candidate["v3_constraint_feasible"] is True
    assert candidate["v3_advance_gate_passed"] is True
    assert v3.select_best_epoch_v3([control, candidate]) == 1


def test_v3_unary_regression_is_infeasible_even_when_pairwise_mean_improves() -> None:
    fit_angle = _metrics(smooth=1.0, mae=1.0)["adapter_angle"]
    control_metrics = _metrics(smooth=1.0, mae=1.0)
    control = v3.annotate_v3_selection_record(
        {"epoch": 0, **control_metrics},
        control_record=control_metrics,
        selector=_selector(mean=0.0, cvar=0.0, worst_mean=0.0, worst_cvar=0.0),
        fit_angle=fit_angle,
    )
    candidate = v3.annotate_v3_selection_record(
        {"epoch": 1, **_metrics(smooth=1.001, mae=0.99)},
        control_record=control,
        selector=_selector(mean=-0.01, cvar=0.0, worst_mean=0.0, worst_cvar=0.0),
        fit_angle=fit_angle,
    )

    assert candidate["v3_advance_gate_checks"]["unary_smooth_l1_and_mae"] is False
    assert candidate["v3_constraint_feasible"] is False
    assert v3.select_best_epoch_v3([control, candidate]) == 0
