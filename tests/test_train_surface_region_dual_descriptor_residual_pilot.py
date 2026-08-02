from __future__ import annotations

import argparse
import inspect

import pytest
import torch
from torch import nn

from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.models.surface_region_dual_descriptor import SurfaceRegionDualDescriptor
from radio_gs.scripts import (
    train_surface_region_dual_descriptor_residual_pilot as pilot,
)
from radio_gs.scripts.train_surface_region_text_response_distill import (
    state_dict_sha256,
)


def _selector(
    *, mean: float, cvar: float = 0.0, worst_mean: float = 0.0, worst_cvar: float = 0.0
) -> dict[str, object]:
    return {
        "control_scale_mean_unit_loss": 1.0,
        "candidate_mean_unit_loss": 1.0 + mean,
        "normalized_mean_delta": mean,
        "normalized_upper_cvar10_delta": cvar,
        "worst_scene_mean_delta": worst_mean,
        "worst_scene_upper_cvar10_delta": worst_cvar,
        "per_scene": {"scene": {"mean_delta": mean, "upper_cvar10_delta": cvar}},
        "unit_count": 8,
    }


def _record(epoch: int, mean: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "adapter_state_dict_sha256": "a" * 64,
        "fit_constraints_feasible": True,
        "fit_constraint_checks": {name: True for name in pilot.FIT_CONSTRAINT_NAMES},
        "official_token_bitwise_equal": True,
        "official_descriptor_bitwise_equal": True,
        "dev_control_referenced_selector": _selector(
            mean=mean, cvar=0.004, worst_mean=0.009, worst_cvar=0.009
        ),
        "validation_unary_relative_deltas": {
            "text_response_smooth_l1": -0.01,
            "text_response_mae": -0.01,
        },
        "validation_descriptor_deltas": {
            "summary_token_cosine": 0.0,
            "mean_descriptor_cosine": 0.0,
            "all_view_descriptor_cosine": 0.0,
        },
        "dev": {"surface_selection_score": 0.9},
    }


def test_contract_preserves_actual_promoted_hidden256_lineage() -> None:
    contract = pilot.training_contract()

    assert contract["seed"] == 0
    assert contract["architecture"]["one_architecture_only"] is True
    assert contract["architecture"]["summary_hidden_dim"] == 256
    assert contract["architecture"]["trainable_parameter_count"] == 856_065
    assert contract["architecture"]["parameter_count_lineage_correction"] == {
        "earlier_estimate": 823_041,
        "earlier_estimate_assumed_hidden_dim": 128,
        "frozen_promoted_control_hidden_dim": 256,
        "correct_count_for_frozen_control": 856_065,
        "decision": "preserve_frozen_control_instead_of_changing_base",
    }
    assert contract["immutable_controls"] == {
        "surface_region_summary_readout_v2": True,
        "official_siglip2_summary_head": True,
        "official_summary_token": True,
        "official_descriptor": True,
        "primitive_feature_field": True,
    }
    assert contract["data"]["benchmark_vocabulary_images_masks_or_targets"] is False
    assert (
        contract["epoch0_gradient_calibration"]["weights_fixed_after_epoch_zero"]
        is True
    )
    assert contract["selection"]["metric_continuation"] is False
    assert contract["selection"]["additional_seed_after_failure"] is False
    assert contract["advance_gate"]["single_conjunction"] is True
    assert contract["advance_gate"]["missing_replay_evidence"] == (
        "fail_closed_pending_finalization"
    )


def test_actual_hidden256_model_has_only_declared_adapter_parameters() -> None:
    torch.manual_seed(701)
    base = SurfaceRegionSummaryReadoutV2(feature_dim=4, hidden_dim=256)
    head = nn.Linear(4, 1536)
    model = SurfaceRegionDualDescriptor(base, head)

    parameters = pilot._trainable_parameters(model)
    state = pilot._adapter_state(model)

    assert sum(parameter.numel() for parameter in parameters) == 856_065
    assert sum(value.numel() for value in state.values()) == 856_065
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(not parameter.requires_grad for parameter in head.parameters())
    assert set(name.split(".", 1)[0] for name in state) == {
        "context_norm",
        "context_projection",
        "film",
        "gate",
    }


def test_adapter_state_round_trip_excludes_frozen_controls() -> None:
    torch.manual_seed(709)
    model = SurfaceRegionDualDescriptor(
        SurfaceRegionSummaryReadoutV2(feature_dim=4, hidden_dim=256),
        nn.Linear(4, 1536),
    )
    state = pilot._adapter_state(model)
    digest = state_dict_sha256(state)
    with torch.no_grad():
        model.film.bias.fill_(0.25)

    pilot._load_adapter_state(model, state)

    assert state_dict_sha256(pilot._adapter_state(model)) == digest
    assert torch.count_nonzero(model.film.bias) == 0


def test_cpu_fit_graph_uses_generic_crop_teachers_and_target_blind_bank() -> None:
    torch.manual_seed(719)
    model = SurfaceRegionDualDescriptor(
        SurfaceRegionSummaryReadoutV2(feature_dim=4, hidden_dim=8),
        nn.Linear(4, 2),
        descriptor_dim=2,
        bottleneck_dim=4,
    )
    teacher_tokens = torch.randn(4, 2, 4)
    teacher_descriptors = torch.nn.functional.normalize(torch.randn(4, 2, 2), dim=-1)
    data = {
        "radio_features": torch.randn(4, 3, 4),
        "geometry": torch.randn(4, 3, 14),
        "anchor_index": torch.zeros(4, dtype=torch.long),
        "token_mask": torch.ones(4, 3, dtype=torch.bool),
        "reliability": torch.ones(4, 3, 1),
        "official_summary_tokens": teacher_tokens,
        "official_crop_summaries": teacher_descriptors,
        "teacher_mask": torch.ones(4, 2, dtype=torch.bool),
        "scene_ids": ["fit_a", "fit_a", "fit_b", "fit_b"],
    }
    text_bank = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)

    metrics, units, valid, controls = pilot.evaluate_split(
        model, data, text_bank, torch.device("cpu")
    )

    assert units.shape == valid.shape == (2, 3)
    assert valid.dtype == torch.bool
    assert set(metrics) == {
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
        "surface_selection_score",
        "text_response_smooth_l1",
        "text_response_mae",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert set(controls) == {
        "official_token_sha256",
        "official_descriptor_sha256",
    }


def test_selection_uses_only_fit_feasible_frozen_dev_aggregates() -> None:
    control = _record(0, 0.0)
    infeasible = _record(1, -0.10)
    infeasible["fit_constraints_feasible"] = False
    valid = _record(2, -0.003)

    assert pilot.select_best_epoch([control, infeasible, valid]) == 2

    with pytest.raises(ValueError, match="contiguous"):
        pilot.select_best_epoch([control, valid])


def test_missing_point_render_replay_is_explicit_and_fail_closed() -> None:
    record = _record(1, -0.003)

    gate = pilot.build_pilot_gate(record)

    assert gate["passed"] is False
    assert gate["checks"]["point_render_max_abs_error_le_1e_minus_6"] is False
    assert gate["finalization_status"] == "pending"
    evidence = gate["point_render_replay_evidence"]
    assert evidence["evidence_present"] is False
    assert evidence["status"] == "missing_pending_independent_materializer_replay"


def test_state_bound_independent_replay_can_finalize_pure_gate() -> None:
    record = _record(1, -0.003)
    evidence = {
        "schema_version": 1,
        "artifact_type": "dual_descriptor_point_render_replay_evidence",
        "candidate_adapter_state_dict_sha256": "a" * 64,
        "independent_materializer_replay": True,
        "frozen_scalar_compositor_replay": True,
        "point_render_replay_max_abs_error": 1e-7,
    }

    gate = pilot.build_pilot_gate(record, point_render_replay_evidence=evidence)

    assert gate["passed"] is True
    assert gate["finalization_status"] == "finalized"
    assert gate["point_render_replay_evidence"]["evidence_present"] is True

    wrong = dict(evidence)
    wrong["candidate_adapter_state_dict_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="evidence differs"):
        pilot.build_pilot_gate(record, point_render_replay_evidence=wrong)


def test_cli_is_fixed_seed0_and_has_no_benchmark_or_search_inputs() -> None:
    destinations = {action.dest for action in pilot.build_arg_parser()._actions}

    assert "seed" not in destinations
    assert "epochs" not in destinations
    assert "learning_rate" not in destinations
    assert "point_render_replay_evidence" not in destinations
    assert not any(
        fragment in destination
        for destination in destinations
        for fragment in ("benchmark", "query", "mask", "target", "image")
    )
    source = inspect.getsource(pilot.train)
    assert 'seed1_executed": False' in source
    assert 'metric_continuation": False' in source
    assert "build_pilot_gate(history[best_epoch])" in source


def test_existing_output_fails_before_loading_any_training_input(tmp_path) -> None:
    output = tmp_path / "occupied.pt"
    output.write_bytes(b"occupied")
    args = argparse.Namespace(output=output)

    with pytest.raises(FileExistsError, match="must be new"):
        pilot.train(args)


def test_threshold_constants_are_identical_to_single_conjunction() -> None:
    assert pilot.DEV_MEAN_DELTA_MAX == -0.0025
    assert pilot.DEV_GLOBAL_CVAR_DELTA_MAX == 0.005
    assert pilot.DEV_WORST_SCENE_MEAN_DELTA_MAX == 0.010
    assert pilot.DEV_WORST_SCENE_CVAR_DELTA_MAX == 0.010
    assert pilot.POINT_RENDER_MAX_ABS_ERROR == 1e-6
