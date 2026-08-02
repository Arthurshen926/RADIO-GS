from __future__ import annotations

import pytest
import torch
from torch import nn

from radio_gs.models.surface_text_response_adapter import (
    LowRankTangentSummaryAdapter,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_cvar_pilot as v2,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_pilot as v1,
)


def test_v2_contract_preserves_v1_boundaries_and_matches_selector_tail() -> None:
    contract = v2.training_contract_v2()
    primary = contract["objective"]["primary_pairwise_risk"]

    assert contract["seed"] == 0
    assert contract["adapter"] == {
        "rank": 32,
        "max_angle_degrees": 0.1,
        "location": "before_frozen_official_siglip_summary_head",
    }
    assert contract["base_surface_readout"] == "frozen_exact_seed0_control"
    assert contract["official_siglip_summary_head"] == "frozen"
    assert primary["within_scene_mean_weight"] == 0.5
    assert primary["within_scene_fractional_upper_cvar_weight"] == 0.5
    assert primary["within_scene_fractional_upper_cvar_tail"] == 0.10
    assert primary["across_scene_reduction"] == "equal_scene_mean"
    assert primary["student_scene_query_unit_autograd"] == "retained"
    assert primary["teacher_variance_weights_text_bank_autograd"] == "detached"
    assert contract["selector"] == v1._selector_contract()
    assert contract["v1_boundaries_preserved"]["scope"] == (
        "external_benchmarks_unopened"
    )
    assert contract["v1_boundaries_preserved"][
        "evaluation_protocol_freeze_id"
    ] == "evaluation_protocols_20260801_v1"


def test_v2_cli_is_seed0_only_and_has_no_benchmark_selection() -> None:
    destinations = {
        action.dest for action in v2.build_arg_parser()._actions
    }
    assert "seed" not in destinations
    assert "canonical_task_id" not in destinations
    assert "registry_row" not in destinations
    assert "benchmark" not in destinations


class _IdentitySummaryHead(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def test_v2_cpu_evaluation_reports_mean_cvar_primary_risk() -> None:
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=2,
        rank=1,
        max_angle_degrees=0.1,
    )
    base_tokens = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]]
    )
    views = torch.stack(
        [torch.stack([value, value]) for value in base_tokens]
    )
    data = {
        "radio_features": torch.zeros(4, 1, 2),
        "official_summary_tokens": views.clone(),
        "official_crop_summaries": views.clone(),
        "teacher_mask": torch.ones(4, 2, dtype=torch.bool),
        "scene_ids": ["a", "a", "b", "b"],
    }

    metrics, units, valid = v2._evaluate_v2(
        adapter,
        _IdentitySummaryHead(),
        base_tokens,
        data,
        torch.eye(2),
    )

    assert metrics["primary_equal_scene_mean_cvar_risk"] == pytest.approx(0.0)
    assert metrics["primary_scene_mean"] == pytest.approx([0.0, 0.0])
    assert metrics["primary_scene_upper_fractional_cvar10"] == pytest.approx(
        [0.0, 0.0]
    )
    assert metrics["primary_scene_risk"] == pytest.approx([0.0, 0.0])
    assert units.shape == valid.shape == (2, 2)


def test_v2_objective_differs_from_v1_only_in_primary_pairwise_risk() -> None:
    v1_contract = v1._training_contract()
    v2_contract = v2.training_contract_v2()

    for field in (
        "seed",
        "epochs",
        "patience",
        "target_batch_rows",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "base_surface_readout",
        "official_siglip_summary_head",
        "adapter",
        "selector",
    ):
        assert v2_contract[field] == v1_contract[field]
    assert v2_contract["objective"] != v1_contract["objective"]
    assert v2_contract["objective"]["independent_response_weight"] == (
        v1_contract["objective"]["independent_response_weight"]
    )
    assert v2_contract["objective"]["surface_descriptor_weight"] == (
        v1_contract["objective"]["surface_descriptor_weight"]
    )

