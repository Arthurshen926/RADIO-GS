#!/usr/bin/env python3
"""Run the sole fit-only seed-0 control-referenced adapter v3 pilot.

This additive pilot does not alter v1/v2, the frozen Surface readout, the
official SigLIP2 summary head, the target-blind fit bank, or the evaluation
protocol.  Every epoch forms one full-fit ``[scene,query]`` paired-delta
objective before making one AdamW proposal.  Validation/dev/benchmark data
never enter the optimizer.

The proposal is a trust step: if more than five percent of *fit* tokens use
at least 99 percent of the 0.1-degree hard angular cap, its parameter
displacement is repeatedly halved.  A fully rejected proposal restores both
parameters and optimizer state.  A partially accepted proposal retains the
full-fit gradient moments but records the accepted displacement fraction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.control_referenced_uncertainty_response_risk import (
    compute_control_referenced_exact_hinge_risk,
)
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
)
from radio_gs.losses.uncertainty_response_risk import (
    compute_uncertainty_weighted_scene_query_pairwise_gap_units,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.models.surface_text_response_adapter import (
    LowRankTangentSummaryAdapter,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_cvar_pilot as v2,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_pilot as v1,
)
from radio_gs.scripts.bind_evaluation_protocol_freeze import (
    UNOPENED_SCOPE,
    build_binding,
)
from radio_gs.scripts.train_surface_region_summary_readout import (
    _load,
    _paths,
    _seed_training,
    _targets,
)
from radio_gs.scripts.train_surface_region_text_response_distill import (
    _cache_binding,
    _fit_bank_binding,
    _validate_train_validation_contracts,
    _verify_radio_checkpoint,
    load_fit_text_embedding_bank,
    load_surface_control_checkpoint,
    state_dict_sha256,
)
from radio_gs.utils.immutable_artifacts import (
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_region_control_referenced_response_adapter_seed0_pilot"
ALGORITHM_VERSION = (
    "frozen_surface_low_rank_tangent_adapter_full_fit_control_referenced_"
    "paired_mean_l1_hinge_trust_v3"
)

FIT_CVAR_TAIL_FRACTION = 0.10
FIT_GLOBAL_CVAR_TOLERANCE = 0.005
FIT_WORST_SCENE_MEAN_TOLERANCE = 0.010
FIT_WORST_SCENE_CVAR_TOLERANCE = 0.010
FIT_UNARY_DELTA_TOLERANCE = 0.0
# This is a fixed, dimensionless empirical L1-hinge weight.  It does not
# claim the unknown multiplier bound required for a mathematical exact-
# penalty guarantee.
FIT_L1_HINGE_WEIGHT = 1.0

MAX_SATURATION_RATIO_AT_99PCT_CAP = 0.05
TRUST_BACKTRACK_FACTOR = 0.5
TRUST_MAX_BACKTRACK_STEPS = 24


def _objective_contract_v3() -> dict[str, Any]:
    return {
        "formula": (
            "mean(d)+1.0*(relu(global_cvar10(d)-0.005)+"
            "relu(max_scene_mean(d)-0.010)+"
            "relu(max_scene_cvar10(d)-0.010)+"
            "relu(independent_unary_delta-0.0))"
        ),
        "paired_unit_delta": (
            "d=(candidate_scene_query_unit-control_scene_query_unit)/"
            "mean(control_valid_scene_query_unit)"
        ),
        "scene_query_unit": ("uncertainty_weighted_normalized_pairwise_gap_smooth_l1"),
        "primary": "global_mean_of_all_valid_full_fit_scene_query_deltas",
        "constraints": {
            "global_fractional_upper_cvar": {
                "tail_fraction": FIT_CVAR_TAIL_FRACTION,
                "tolerance": FIT_GLOBAL_CVAR_TOLERANCE,
            },
            "worst_scene_mean": {"tolerance": FIT_WORST_SCENE_MEAN_TOLERANCE},
            "worst_scene_fractional_upper_cvar": {
                "tail_fraction": FIT_CVAR_TAIL_FRACTION,
                "tolerance": FIT_WORST_SCENE_CVAR_TOLERANCE,
            },
            "independent_unary_delta": {
                "definition": "candidate_smooth_l1/control_smooth_l1-1",
                "tolerance": FIT_UNARY_DELTA_TOLERANCE,
            },
        },
        "constraint_penalty": {
            "kind": "unsmoothed_l1_hinge",
            "fixed_weight": FIT_L1_HINGE_WEIGHT,
            "mathematical_exact_penalty_guarantee": False,
            "reason": (
                "the fixed empirical multiplier is preregistered but no "
                "unknown optimal Lagrange-multiplier bound is claimed"
            ),
        },
        "reduction_domain": (
            "one_complete_32_scene_fit_set_before_each_optimizer_proposal"
        ),
        "teacher_variance_weights_text_bank_autograd": "detached",
        "student_scene_query_unit_autograd": "retained_until_full_fit_risk",
        "surface_term_in_training_objective": False,
        "surface_role": "unchanged_validation_noninferiority_gate",
        "vocabulary": "target_blind_fit_only",
    }


def _advance_gate_contract_v3() -> dict[str, Any]:
    return {
        "eight_checks": [
            "selected_epoch_gt_zero",
            "normalized_mean_delta_le_negative_0p0025",
            "global_cvar10_delta_le_0p005",
            "worst_scene_mean_delta_le_0p010",
            "worst_scene_cvar10_delta_le_0p010",
            "unary_smooth_l1_and_mae_relative_deltas_le_zero",
            "all_three_surface_deltas_ge_negative_0p002",
            "fit_and_validation_saturation_at_99pct_cap_le_0p05_and_"
            "max_angle_le_0p10001_degrees",
        ],
        "selected_epoch_gt_zero": True,
        "required_mean_improvement": v1.PILOT_REQUIRED_MEAN_IMPROVEMENT,
        "global_cvar_tolerance": FIT_GLOBAL_CVAR_TOLERANCE,
        "worst_scene_mean_tolerance": FIT_WORST_SCENE_MEAN_TOLERANCE,
        "worst_scene_cvar_tolerance": FIT_WORST_SCENE_CVAR_TOLERANCE,
        "unary_relative_delta_tolerance": 0.0,
        "surface_noninferiority_tolerance": (v1.SURFACE_NONINFERIORITY_TOLERANCE),
        "maximum_saturation_ratio_at_99pct_cap": (MAX_SATURATION_RATIO_AT_99PCT_CAP),
        "adapter_max_angle_degrees": v1.ADAPTER_MAX_ANGLE_DEGREES,
        "adapter_angle_audit_absolute_tolerance_degrees": (
            v1.ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES
        ),
        "seed0_only_in_this_artifact": True,
        "seed1_required_only_after_seed0_pass": True,
    }


def training_contract_v3() -> dict[str, Any]:
    contract = v2.training_contract_v2()
    contract.update(
        {
            "objective": _objective_contract_v3(),
            "optimizer": (
                "persistent_adamw_one_full_fit_gradient_proposal_per_epoch_"
                "with_parameter_displacement_backtracking"
            ),
            "optimizer_state_on_partial_backtrack": (
                "retain_full_fit_gradient_adam_moments_and_step_count;only_"
                "parameter_displacement_is_scaled"
            ),
            "optimizer_state_on_full_rejection": (
                "restore_exact_preproposal_parameters_and_optimizer_state"
            ),
            "trust_backtracking": {
                "measurement_split": "fit_train_tokens_only",
                "saturation_definition": "angle_ge_0.99_times_hard_cap",
                "maximum_saturation_ratio": (MAX_SATURATION_RATIO_AT_99PCT_CAP),
                "parameter_displacement_factor": TRUST_BACKTRACK_FACTOR,
                "maximum_backtracking_steps": TRUST_MAX_BACKTRACK_STEPS,
                "hard_cap_degrees_unchanged": v1.ADAPTER_MAX_ANGLE_DEGREES,
                "feasibility_is_mandatory": True,
            },
            "advance_gate": _advance_gate_contract_v3(),
            "v3_boundaries_preserved": {
                "seed": v1.PILOT_SEED,
                "adapter_rank": v1.ADAPTER_RANK,
                "adapter_max_angle_degrees": (v1.ADAPTER_MAX_ANGLE_DEGREES),
                "fit_text_bank": "frozen_target_blind_fit_only",
                "surface_selector": v1._selector_contract(),
                "surface_gate_unchanged": True,
                "official_siglip_summary_head": "frozen",
                "evaluation_protocol_freeze_id": v1.EXPECTED_FREEZE_ID,
                "evaluation_protocol_freeze_sha256": (v1.EXPECTED_FREEZE_SHA256),
                "scope": UNOPENED_SCOPE,
            },
        }
    )
    return contract


def _scene_groups_in_order(
    scene_ids: Sequence[str],
) -> tuple[list[str], list[torch.Tensor]]:
    groups: dict[str, list[int]] = {}
    for row, raw_scene in enumerate(scene_ids):
        scene = str(raw_scene)
        if not scene:
            raise ValueError("fit scene IDs must be non-empty")
        groups.setdefault(scene, []).append(row)
    if not groups or any(len(rows) < 2 for rows in groups.values()):
        raise ValueError("every fit scene must contain at least two rows")
    return list(groups), [
        torch.tensor(rows, dtype=torch.long) for rows in groups.values()
    ]


def _scalarize_fit_statistics(
    statistics: Mapping[str, Any],
    *,
    scene_names: Sequence[str],
    candidate_unary_loss: torch.Tensor,
) -> dict[str, Any]:
    scene_mean = torch.as_tensor(statistics["scene_mean_delta"]).cpu()
    scene_cvar = torch.as_tensor(statistics["scene_upper_fractional_cvar_delta"]).cpu()
    if len(scene_names) != len(scene_mean) or len(scene_names) != len(scene_cvar):
        raise RuntimeError("fit risk scene statistics are misaligned")
    normalized_delta = torch.as_tensor(statistics["normalized_delta"]).cpu()
    return {
        "objective": float(torch.as_tensor(statistics["objective"]).cpu()),
        "global_mean_delta": float(
            torch.as_tensor(statistics["global_mean_delta"]).cpu()
        ),
        "global_upper_fractional_cvar10_delta": float(
            torch.as_tensor(statistics["global_upper_fractional_cvar_delta"]).cpu()
        ),
        "worst_scene_mean_delta": float(
            torch.as_tensor(statistics["worst_scene_mean_delta"]).cpu()
        ),
        "worst_scene_upper_fractional_cvar10_delta": float(
            torch.as_tensor(statistics["worst_scene_upper_fractional_cvar_delta"]).cpu()
        ),
        "independent_unary_loss": float(candidate_unary_loss.detach().cpu()),
        "independent_unary_delta": float(
            torch.as_tensor(statistics["independent_unary_delta"]).cpu()
        ),
        "exact_hinge_penalty": float(
            torch.as_tensor(statistics["exact_hinge_penalty"]).cpu()
        ),
        "exact_hinge_violations": {
            name: float(torch.as_tensor(value).cpu())
            for name, value in statistics["exact_hinge_violations"].items()
        },
        "control_scale_mean_unit_loss": float(
            torch.as_tensor(statistics["control_scale_mean_unit_loss"]).cpu()
        ),
        "per_scene": {
            str(scene): {
                "mean_delta": float(scene_mean[index]),
                "upper_fractional_cvar10_delta": float(scene_cvar[index]),
            }
            for index, scene in enumerate(scene_names)
        },
        "scene_count": int(torch.as_tensor(statistics["scene_count"]).cpu()),
        "valid_scene_query_count": int(
            torch.as_tensor(statistics["valid_scene_query_count"]).cpu()
        ),
        "normalized_delta_sha256": tensor_sha256(normalized_delta.float()),
    }


def compute_full_fit_control_referenced_objective(
    adapter: LowRankTangentSummaryAdapter,
    head: torch.nn.Module,
    base_tokens: torch.Tensor,
    data: Mapping[str, Any],
    text_bank: torch.Tensor,
    control_unit_loss: torch.Tensor,
    control_valid: torch.Tensor,
    control_unary_loss: float,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build one differentiable risk over the complete fit scene set."""

    if "scene_ids" not in data:
        raise ValueError("full-fit risk requires exact row-to-scene bindings")
    scene_names, scene_rows = _scene_groups_in_order(data["scene_ids"])
    candidate_units: list[torch.Tensor] = []
    candidate_validity: list[torch.Tensor] = []
    student_parts: list[torch.Tensor] = []
    teacher_parts: list[torch.Tensor] = []
    for scene, cpu_rows in zip(scene_names, scene_rows):
        _target_token, teacher, all_descriptors, teacher_mask = _targets(data, cpu_rows)
        device_rows = cpu_rows.to(base_tokens.device)
        adapted = adapter(base_tokens.index_select(0, device_rows))
        student = F.normalize(head(adapted[:, None])[:, 0].float(), dim=-1, eps=1e-8)
        teacher = teacher.to(base_tokens.device)
        all_descriptors = all_descriptors.to(base_tokens.device)
        teacher_mask = teacher_mask.to(base_tokens.device)
        units, validity, _unit_stats = (
            compute_uncertainty_weighted_scene_query_pairwise_gap_units(
                student,
                teacher,
                all_descriptors,
                teacher_mask,
                text_bank,
                [scene] * len(cpu_rows),
                standard_error_multiplier=v1.STANDARD_ERROR_MULTIPLIER,
                tie_tolerance=v1.TIE_TOLERANCE,
                eps=v1.EPS,
            )
        )
        if units.shape[0] != 1:
            raise RuntimeError("one fit scene produced multiple risk rows")
        candidate_units.append(units[0])
        candidate_validity.append(validity[0])
        student_parts.append(student)
        teacher_parts.append(teacher)

    units = torch.stack(candidate_units)
    validity = torch.stack(candidate_validity)
    student = torch.cat(student_parts)
    teacher = torch.cat(teacher_parts)
    candidate_unary = compute_independent_normalized_cosine_response_smooth_l1_loss(
        student, teacher, text_bank
    )
    objective, statistics = compute_control_referenced_exact_hinge_risk(
        units,
        validity,
        control_unit_loss,
        control_valid,
        candidate_unary,
        control_unary_loss,
        cvar_tail_fraction=FIT_CVAR_TAIL_FRACTION,
        global_cvar_tolerance=FIT_GLOBAL_CVAR_TOLERANCE,
        worst_scene_mean_tolerance=FIT_WORST_SCENE_MEAN_TOLERANCE,
        worst_scene_cvar_tolerance=FIT_WORST_SCENE_CVAR_TOLERANCE,
        unary_delta_tolerance=FIT_UNARY_DELTA_TOLERANCE,
        exact_penalty_weight=FIT_L1_HINGE_WEIGHT,
        eps=1e-12,
    )
    return objective, statistics, units, validity, candidate_unary


@torch.no_grad()
def fit_adapter_angle_statistics(
    adapter: LowRankTangentSummaryAdapter,
    base_tokens: torch.Tensor,
) -> dict[str, float]:
    adapted_parts = []
    for start in range(0, len(base_tokens), v1.TARGET_BATCH_ROWS):
        stop = min(start + v1.TARGET_BATCH_ROWS, len(base_tokens))
        adapted_parts.append(adapter(base_tokens[start:stop]))
    adapted = torch.cat(adapted_parts)
    return v1.adapter_angle_statistics(
        base_tokens,
        adapted,
        max_angle_degrees=v1.ADAPTER_MAX_ANGLE_DEGREES,
    )


def _interpolate_adapter_state(
    adapter: torch.nn.Module,
    old_state: Mapping[str, torch.Tensor],
    proposed_state: Mapping[str, torch.Tensor],
    fraction: float,
) -> None:
    if set(old_state) != set(proposed_state):
        raise ValueError("old/proposed adapter states are misaligned")
    if not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("trust interpolation fraction must lie in [0,1]")
    interpolated: dict[str, torch.Tensor] = {}
    for name in old_state:
        old = old_state[name]
        proposed = proposed_state[name].to(device=old.device, dtype=old.dtype)
        if old.shape != proposed.shape:
            raise ValueError("old/proposed adapter tensor shapes differ")
        if old.is_floating_point():
            interpolated[name] = old + float(fraction) * (proposed - old)
        elif float(fraction) == 0.0:
            interpolated[name] = old.clone()
        else:
            interpolated[name] = proposed.clone()
    adapter.load_state_dict(interpolated, strict=True)


@torch.no_grad()
def apply_fit_trust_backtracking(
    adapter: LowRankTangentSummaryAdapter,
    base_tokens: torch.Tensor,
    old_state: Mapping[str, torch.Tensor],
    proposed_state: Mapping[str, torch.Tensor],
    *,
    maximum_saturation_ratio: float = MAX_SATURATION_RATIO_AT_99PCT_CAP,
    backtrack_factor: float = TRUST_BACKTRACK_FACTOR,
    maximum_backtracking_steps: int = TRUST_MAX_BACKTRACK_STEPS,
) -> dict[str, Any]:
    """Accept the largest fit-token-feasible proposal displacement."""

    if (
        not math.isfinite(float(maximum_saturation_ratio))
        or not 0.0 <= float(maximum_saturation_ratio) <= 1.0
        or not math.isfinite(float(backtrack_factor))
        or not 0.0 < float(backtrack_factor) < 1.0
        or int(maximum_backtracking_steps) < 0
    ):
        raise ValueError("trust backtracking contract is invalid")

    _interpolate_adapter_state(adapter, old_state, proposed_state, 0.0)
    old_angle = fit_adapter_angle_statistics(adapter, base_tokens)
    if (
        old_angle["saturation_ratio_at_99pct_cap"]
        > float(maximum_saturation_ratio) + 1e-12
    ):
        raise RuntimeError("preproposal adapter is outside the fit trust set")

    fraction = 1.0
    _interpolate_adapter_state(adapter, old_state, proposed_state, fraction)
    proposal_angle = fit_adapter_angle_statistics(adapter, base_tokens)
    accepted_angle = proposal_angle
    steps = 0
    while accepted_angle["saturation_ratio_at_99pct_cap"] > float(
        maximum_saturation_ratio
    ) + 1e-12 and steps < int(maximum_backtracking_steps):
        fraction *= float(backtrack_factor)
        steps += 1
        _interpolate_adapter_state(adapter, old_state, proposed_state, fraction)
        accepted_angle = fit_adapter_angle_statistics(adapter, base_tokens)

    fully_rejected = False
    if (
        accepted_angle["saturation_ratio_at_99pct_cap"]
        > float(maximum_saturation_ratio) + 1e-12
    ):
        fraction = 0.0
        fully_rejected = True
        _interpolate_adapter_state(adapter, old_state, proposed_state, fraction)
        accepted_angle = fit_adapter_angle_statistics(adapter, base_tokens)
    if (
        accepted_angle["saturation_ratio_at_99pct_cap"]
        > float(maximum_saturation_ratio) + 1e-12
    ):
        raise RuntimeError("fit trust backtracking failed closed")

    return {
        "maximum_saturation_ratio_at_99pct_cap": float(maximum_saturation_ratio),
        "backtrack_factor": float(backtrack_factor),
        "maximum_backtracking_steps": int(maximum_backtracking_steps),
        "backtracking_steps": steps,
        "accepted_parameter_displacement_fraction": fraction,
        "fully_rejected": fully_rejected,
        "preproposal_angle": old_angle,
        "unbacktracked_proposal_angle": proposal_angle,
        "accepted_angle": accepted_angle,
        "feasible": True,
    }


def reconcile_optimizer_state_after_trust(
    optimizer: torch.optim.Optimizer,
    preproposal_optimizer_state: Mapping[str, Any],
    trust_record: Mapping[str, Any],
) -> str:
    """Restore optimizer state iff the parameter proposal was fully rejected."""

    fraction = float(trust_record["accepted_parameter_displacement_fraction"])
    fully_rejected = bool(trust_record["fully_rejected"])
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("trust record has an invalid accepted fraction")
    if fully_rejected != (fraction == 0.0):
        raise ValueError("trust rejection flag and accepted fraction disagree")
    if fully_rejected:
        optimizer.load_state_dict(dict(preproposal_optimizer_state))
        return "restored_exact_preproposal_state_after_full_rejection"
    return (
        "retained_full_fit_gradient_moments_for_partial_or_full_" "parameter_acceptance"
    )


def _relative_delta(candidate: float, control: float, label: str) -> float:
    if (
        not math.isfinite(float(candidate))
        or not math.isfinite(float(control))
        or float(control) <= 1e-12
    ):
        raise ValueError(f"{label} control-relative metric is invalid")
    return float(candidate) / float(control) - 1.0


def annotate_v3_selection_record(
    record: Mapping[str, Any],
    *,
    control_record: Mapping[str, Any],
    selector: Mapping[str, Any],
    fit_angle: Mapping[str, float],
) -> dict[str, Any]:
    value = v1.annotate_selection_record(
        record, control_record=control_record, selector=selector
    )
    unary_deltas = {
        "text_response_smooth_l1": _relative_delta(
            float(value["text_response_smooth_l1"]),
            float(control_record["text_response_smooth_l1"]),
            "text_response_smooth_l1",
        ),
        "text_response_mae": _relative_delta(
            float(value["text_response_mae"]),
            float(control_record["text_response_mae"]),
            "text_response_mae",
        ),
    }
    angle_limit = (
        v1.ADAPTER_MAX_ANGLE_DEGREES + v1.ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES
    )
    validation_angle = value["adapter_angle"]
    checks = {
        "selected_epoch_gt_zero": int(value["epoch"]) > 0,
        "normalized_mean_delta": (
            float(selector["normalized_mean_delta"])
            <= -v1.PILOT_REQUIRED_MEAN_IMPROVEMENT + 1e-12
        ),
        "global_cvar10_delta": (
            float(selector["normalized_upper_cvar10_delta"])
            <= FIT_GLOBAL_CVAR_TOLERANCE + 1e-12
        ),
        "worst_scene_mean_delta": (
            float(selector["worst_scene_mean_delta"])
            <= FIT_WORST_SCENE_MEAN_TOLERANCE + 1e-12
        ),
        "worst_scene_cvar10_delta": (
            float(selector["worst_scene_upper_cvar10_delta"])
            <= FIT_WORST_SCENE_CVAR_TOLERANCE + 1e-12
        ),
        "unary_smooth_l1_and_mae": all(
            delta <= 1e-12 for delta in unary_deltas.values()
        ),
        "three_surface_fields": value["surface_control_feasible"] is True,
        "fit_and_validation_angle_trust": (
            float(fit_angle["saturation_ratio_at_99pct_cap"])
            <= MAX_SATURATION_RATIO_AT_99PCT_CAP + 1e-12
            and float(validation_angle["saturation_ratio_at_99pct_cap"])
            <= MAX_SATURATION_RATIO_AT_99PCT_CAP + 1e-12
            and float(fit_angle["max_degrees"]) <= angle_limit
            and float(validation_angle["max_degrees"]) <= angle_limit
        ),
    }
    constraint_names = (
        "global_cvar10_delta",
        "worst_scene_mean_delta",
        "worst_scene_cvar10_delta",
        "unary_smooth_l1_and_mae",
        "three_surface_fields",
        "fit_and_validation_angle_trust",
    )
    value.update(
        {
            "validation_unary_control_relative_deltas": unary_deltas,
            "fit_adapter_angle": dict(fit_angle),
            "v3_advance_gate_checks": checks,
            "v3_constraint_feasible": all(checks[name] for name in constraint_names),
            "v3_advance_gate_passed": all(checks.values()),
        }
    )
    return value


def select_best_epoch_v3(history: Sequence[Mapping[str, Any]]) -> int:
    if not history or [row.get("epoch") for row in history] != list(
        range(len(history))
    ):
        raise ValueError("v3 pilot history must be contiguous from epoch zero")
    eligible = [row for row in history if row.get("v3_constraint_feasible") is True]
    if not eligible:
        raise RuntimeError("v3 selector has no constraint-feasible control")

    def rank(row: Mapping[str, Any]) -> tuple[float, ...]:
        selector = row["continuous_selector"]
        unary = row["validation_unary_control_relative_deltas"]
        return (
            -float(selector["candidate_mean_unit_loss"]),
            -float(selector["normalized_upper_cvar10_delta"]),
            -float(selector["worst_scene_upper_cvar10_delta"]),
            -float(unary["text_response_smooth_l1"]),
            -float(unary["text_response_mae"]),
            float(row["surface_selection_score"]),
        )

    return int(max(eligible, key=rank)["epoch"])


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("v3 pilot checkpoint/report output must be new")
    repo_root = Path(__file__).resolve().parents[2]
    protocol_binding = build_binding(
        Path(args.evaluation_protocol_freeze),
        scope=UNOPENED_SCOPE,
        repo_root=repo_root,
    )
    if (
        protocol_binding["scope"] != UNOPENED_SCOPE
        or protocol_binding["task"] is not None
        or protocol_binding["freeze"]["freeze_id"] != v1.EXPECTED_FREEZE_ID
        or protocol_binding["freeze"]["sha256"] != v1.EXPECTED_FREEZE_SHA256
    ):
        raise ValueError("v3 pilot evaluation-protocol freeze binding differs")

    train_paths = _paths(args.train_caches)
    validation_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    validation_data, validation_meta = _load(validation_paths, "validation")
    _validate_train_validation_contracts(train_meta, validation_meta)
    if set(train_meta["scenes"]) & set(validation_meta["scenes"]):
        raise ValueError("v3 pilot train/validation scenes overlap")
    if "scene_ids" not in train_data or "scene_ids" not in validation_data:
        raise ValueError("v3 pilot caches require exact row-to-scene bindings")
    train_scene_names = v1._scene_order(train_data["scene_ids"])
    validation_scene_names = v1._scene_order(validation_data["scene_ids"])
    if len(train_scene_names) != len(train_meta["scenes"]):
        raise ValueError("v3 train scene order/count differs")
    if len(validation_scene_names) != len(validation_meta["scenes"]):
        raise ValueError("v3 validation scene order/count differs")

    radio_path = Path(args.radio_checkpoint).resolve()
    radio_sha = _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    base_model, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=v1.PILOT_SEED,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=256,
        reliability_attention_mode="log_prior",
        context_pooling_mode="joint_attention_v1",
    )

    device = torch.device(args.device)
    _seed_training(v1.PILOT_SEED, device=device)
    base_model = base_model.to(device).eval().requires_grad_(False)
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).to(device)
    head.eval().requires_grad_(False)
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=1280,
        rank=v1.ADAPTER_RANK,
        max_angle_degrees=v1.ADAPTER_MAX_ANGLE_DEGREES,
    ).to(device)
    text_bank = fit_bank["embeddings"].to(device)
    base_train = v1._precompute_base_tokens(base_model, train_data, device)
    base_validation = v1._precompute_base_tokens(base_model, validation_data, device)
    base_state = v1._clone_state(base_model)
    base_state_sha = state_dict_sha256(base_state)
    if any(parameter.requires_grad for parameter in base_model.parameters()) or any(
        parameter.requires_grad for parameter in head.parameters()
    ):
        raise RuntimeError("v3 pilot base readout/head freeze failed")

    train_teacher, train_views, train_teacher_mask = v1._all_teacher_targets(train_data)
    validation_teacher, validation_views, validation_teacher_mask = (
        v1._all_teacher_targets(validation_data)
    )
    uncertainty_statistics = {
        "train": v1.uncertainty_weight_statistics(
            train_teacher,
            train_views,
            train_teacher_mask,
            fit_bank["embeddings"],
            train_data["scene_ids"],
        ),
        "validation": v1.uncertainty_weight_statistics(
            validation_teacher,
            validation_views,
            validation_teacher_mask,
            fit_bank["embeddings"],
            validation_data["scene_ids"],
        ),
    }

    adapter.eval()
    control_fit_metrics, control_fit_units, control_fit_valid = v2._evaluate_v2(
        adapter, head, base_train, train_data, text_bank
    )
    control_metrics, control_units, control_valid = v2._evaluate_v2(
        adapter, head, base_validation, validation_data, text_bank
    )
    control_selector, _ = v1.continuous_selector_metrics(
        control_units,
        control_valid,
        control_units,
        control_valid,
        validation_scene_names,
    )
    control_fit_angle = fit_adapter_angle_statistics(adapter, base_train)
    control_record = annotate_v3_selection_record(
        {
            "epoch": 0,
            "initialization": "zero_up_identity_adapter",
            **control_metrics,
        },
        control_record={**control_metrics},
        selector=control_selector,
        fit_angle=control_fit_angle,
    )
    control_record["selection_updated_best"] = True
    initial_adapter_state = v1._clone_state(adapter)
    initial_adapter_sha = state_dict_sha256(initial_adapter_state)
    architecture = adapter.architecture()
    control_record.update(
        {
            "base_surface_state_dict_sha256": base_state_sha,
            "response_adapter_state_dict_sha256": initial_adapter_sha,
            "combined_state_sha256": v1.combined_state_sha256(
                base_state_sha, initial_adapter_sha, architecture["digest"]
            ),
            "scene_query_unit_loss_sha256": tensor_sha256(control_units.float()),
            "scene_query_unit_valid_sha256": tensor_sha256(control_valid),
        }
    )
    history: list[dict[str, Any]] = [control_record]
    selector_units: list[dict[str, torch.Tensor | int]] = [
        {"epoch": 0, "loss": control_units.float(), "valid": control_valid}
    ]
    best_epoch = 0
    best_state = initial_adapter_state
    stale = 0
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=v1.LEARNING_RATE, weight_decay=v1.WEIGHT_DECAY
    )

    for epoch in range(1, v1.EPOCHS + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        old_adapter_state = v1._clone_state(adapter)
        old_optimizer_state = copy.deepcopy(optimizer.state_dict())
        objective, fit_stats, _fit_units, _fit_valid, fit_unary = (
            compute_full_fit_control_referenced_objective(
                adapter,
                head,
                base_train,
                train_data,
                text_bank,
                control_fit_units,
                control_fit_valid,
                float(control_fit_metrics["text_response_smooth_l1"]),
            )
        )
        gradient_origin = _scalarize_fit_statistics(
            fit_stats,
            scene_names=train_scene_names,
            candidate_unary_loss=fit_unary,
        )
        objective.backward()
        optimizer.step()
        proposed_adapter_state = v1._clone_state(adapter)
        trust = apply_fit_trust_backtracking(
            adapter,
            base_train,
            old_adapter_state,
            proposed_adapter_state,
        )
        trust["optimizer_state_action"] = reconcile_optimizer_state_after_trust(
            optimizer, old_optimizer_state, trust
        )

        adapter.eval()
        with torch.no_grad():
            (
                accepted_objective,
                accepted_fit_stats,
                accepted_fit_units,
                accepted_fit_valid,
                accepted_fit_unary,
            ) = compute_full_fit_control_referenced_objective(
                adapter,
                head,
                base_train,
                train_data,
                text_bank,
                control_fit_units,
                control_fit_valid,
                float(control_fit_metrics["text_response_smooth_l1"]),
            )
        accepted_fit = _scalarize_fit_statistics(
            accepted_fit_stats,
            scene_names=train_scene_names,
            candidate_unary_loss=accepted_fit_unary,
        )
        if not math.isclose(
            accepted_fit["objective"],
            float(accepted_objective.detach().cpu()),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise RuntimeError("accepted fit objective audit differs")
        if (
            trust["accepted_angle"]["saturation_ratio_at_99pct_cap"]
            > MAX_SATURATION_RATIO_AT_99PCT_CAP + 1e-12
        ):
            raise RuntimeError("accepted v3 proposal violated fit trust gate")
        accepted_fit.update(
            {
                "scene_query_unit_loss_sha256": tensor_sha256(
                    accepted_fit_units.detach().cpu().float()
                ),
                "scene_query_unit_valid_sha256": tensor_sha256(
                    accepted_fit_valid.detach().cpu()
                ),
            }
        )

        metrics, units, valid = v2._evaluate_v2(
            adapter, head, base_validation, validation_data, text_bank
        )
        selector, _delta = v1.continuous_selector_metrics(
            units,
            valid,
            control_units,
            control_valid,
            validation_scene_names,
        )
        state = v1._clone_state(adapter)
        adapter_sha = state_dict_sha256(state)
        record = annotate_v3_selection_record(
            {
                "epoch": epoch,
                "fit_training": {
                    "gradient_origin_before_proposal": gradient_origin,
                    "accepted_after_trust_backtracking": accepted_fit,
                    "trust_backtracking": trust,
                },
                **metrics,
            },
            control_record=history[0],
            selector=selector,
            fit_angle=trust["accepted_angle"],
        )
        record.update(
            {
                "base_surface_state_dict_sha256": base_state_sha,
                "response_adapter_state_dict_sha256": adapter_sha,
                "combined_state_sha256": v1.combined_state_sha256(
                    base_state_sha, adapter_sha, architecture["digest"]
                ),
                "scene_query_unit_loss_sha256": tensor_sha256(units.float()),
                "scene_query_unit_valid_sha256": tensor_sha256(valid),
            }
        )
        selected_epoch = select_best_epoch_v3([*history, record])
        best_updated = selected_epoch == epoch
        record["selection_updated_best"] = best_updated
        if best_updated:
            best_epoch = epoch
            best_state = state
            stale = 0
        else:
            if selected_epoch != best_epoch:
                raise RuntimeError("v3 pilot best epoch changed retroactively")
            stale += 1
        record["patience_stale"] = stale
        record["patience_stop"] = stale >= v1.PATIENCE
        history.append(record)
        selector_units.append({"epoch": epoch, "loss": units.float(), "valid": valid})
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= v1.PATIENCE:
            break

    if select_best_epoch_v3(history) != best_epoch:
        raise RuntimeError("v3 pilot online/final selection differs")
    adapter.load_state_dict(best_state, strict=True)
    final_metrics, final_units, final_valid = v2._evaluate_v2(
        adapter, head, base_validation, validation_data, text_bank
    )
    if (
        tensor_sha256(final_units.float())
        != history[best_epoch]["scene_query_unit_loss_sha256"]
        or tensor_sha256(final_valid)
        != history[best_epoch]["scene_query_unit_valid_sha256"]
    ):
        raise RuntimeError("selected v3 adapter replay differs")
    best_adapter_sha = state_dict_sha256(best_state)
    best_combined_sha = v1.combined_state_sha256(
        base_state_sha, best_adapter_sha, architecture["digest"]
    )
    pilot_advance = history[best_epoch]["v3_advance_gate_passed"] is True

    implementation_paths = (
        Path(__file__),
        repo_root / "radio_gs/losses/control_referenced_uncertainty_response_risk.py",
        repo_root / "radio_gs/losses/uncertainty_response_risk.py",
        repo_root
        / "radio_gs/scripts/train_surface_region_uncertainty_response_adapter_cvar_pilot.py",
        repo_root
        / "radio_gs/scripts/train_surface_region_uncertainty_response_adapter_pilot.py",
        repo_root / "radio_gs/models/surface_text_response_adapter.py",
        repo_root / "radio_gs/losses/direct_point_query_logit_distill_loss.py",
        repo_root / "radio_gs/scripts/train_surface_region_summary_readout.py",
        repo_root / "radio_gs/scripts/train_surface_region_text_response_distill.py",
        repo_root / "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
    )
    fit_control = {
        "metrics": control_fit_metrics,
        "scene_order": train_scene_names,
        "scene_query_unit_loss": control_fit_units.float(),
        "scene_query_unit_valid": control_fit_valid,
        "scene_query_unit_loss_sha256": tensor_sha256(control_fit_units.float()),
        "scene_query_unit_valid_sha256": tensor_sha256(control_fit_valid),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "base_surface_state_dict": base_state,
        "base_surface_state_dict_sha256": base_state_sha,
        "response_adapter_architecture": architecture,
        "response_adapter_state_dict": best_state,
        "response_adapter_state_dict_sha256": best_adapter_sha,
        "combined_state_sha256": best_combined_sha,
        "best_epoch": best_epoch,
        "history": history,
        "selector_unit_losses": selector_units,
        "fit_control": fit_control,
        "pilot_advance_gate_passed": pilot_advance,
        "seed1_executed": False,
        "seed1_required_only_after_seed0_pass": True,
        "provenance": {
            "evaluation_protocol": protocol_binding,
            "scope": UNOPENED_SCOPE,
            "external_benchmarks_opened": False,
            "formal_authority": False,
            "pilot_only": True,
            "benchmark_queries_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_images_opened": False,
            "fit_text_bank_opened": True,
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "custom_text_projection": False,
            "official_siglip_summary_head_frozen": True,
            "fit_split_only": True,
            "surface_control": surface_control,
            "train_caches": _cache_binding(train_paths),
            "validation_caches": _cache_binding(validation_paths),
            "fit_text_bank": _fit_bank_binding(fit_bank),
            "radio_checkpoint": {"path": str(radio_path), "sha256": radio_sha},
            "train_contract": train_meta,
            "validation_contract": validation_meta,
            "implementation_sources": [
                v1._file_record(path) for path in implementation_paths
            ],
        },
        "training_contract": training_contract_v3(),
        "uncertainty_weight_statistics": uncertainty_statistics,
        "final_validation": final_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    checkpoint_sha = sha256_file(output)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": f"{ARTIFACT_TYPE}_report",
        "algorithm_version": ALGORITHM_VERSION,
        "output": str(output),
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_protocol": protocol_binding,
        "scope": UNOPENED_SCOPE,
        "external_benchmarks_opened": False,
        "formal_authority": False,
        "pilot_only": True,
        "base_surface_state_dict_sha256": base_state_sha,
        "response_adapter_architecture": architecture,
        "response_adapter_state_dict_sha256": best_adapter_sha,
        "combined_state_sha256": best_combined_sha,
        "best_epoch": best_epoch,
        "selected_history_record": history[best_epoch],
        "pilot_advance_gate_passed": pilot_advance,
        "pilot_advance_gate": _advance_gate_contract_v3(),
        "seed1_executed": False,
        "seed1_required_only_after_seed0_pass": True,
        "fit_control": {
            key: value
            for key, value in fit_control.items()
            if key not in {"scene_query_unit_loss", "scene_query_unit_valid"}
        },
        "uncertainty_weight_statistics": uncertainty_statistics,
        "history_length": len(history),
        "training_contract": training_contract_v3(),
    }
    write_frozen_json(report_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--fit-text-bank", type=Path, required=True)
    parser.add_argument("--fit-text-bank-manifest", type=Path, required=True)
    parser.add_argument("--surface-control-checkpoint", type=Path, required=True)
    parser.add_argument("--surface-control-checkpoint-sha256", required=True)
    parser.add_argument("--radio-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--evaluation-protocol-freeze",
        type=Path,
        default=Path("paper/artifacts/evaluation_protocol_freeze_20260801.yaml"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = train(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
