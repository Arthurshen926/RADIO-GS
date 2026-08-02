#!/usr/bin/env python3
"""Train the sole target-blind seed-0 dual-descriptor residual pilot.

The promoted Surface summary readout and RADIO's official SigLIP2 summary
head are immutable controls.  Only the 856,065 parameters of the semantic
descriptor branch are optimized, using the frozen fit vocabulary and the
scene-disjoint frozen dev caches.  No benchmark input is accepted by this
program.

Point/render consistency is deliberately not inferred from the training
caches.  The training artifact stops with a fail-closed pending gate until an
independent materializer replays the selected state through the frozen scalar
compositor and supplies state-bound evidence to a later finalizer.
"""

from __future__ import annotations

import argparse
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
from radio_gs.losses.dual_descriptor_response_risk import (
    DEV_GLOBAL_CVAR_DELTA_MAX,
    DEV_MEAN_DELTA_MAX,
    DEV_WORST_SCENE_CVAR_DELTA_MAX,
    DEV_WORST_SCENE_MEAN_DELTA_MAX,
    FIT_CONSTRAINT_NAMES,
    POINT_RENDER_MAX_ABS_ERROR,
    build_seed0_single_conjunction_gate,
    calibrate_epoch0_gradient_weights,
    compute_dual_descriptor_loss_components,
    compute_dual_descriptor_response_risk,
)
from radio_gs.losses.uncertainty_response_risk import (
    compute_uncertainty_weighted_scene_query_pairwise_gap_units,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionDualDescriptor,
)
from radio_gs.scripts import (
    train_surface_region_uncertainty_response_adapter_pilot as legacy_pilot,
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
ARTIFACT_TYPE = "surface_region_dual_descriptor_residual_seed0_pilot"
ALGORITHM_VERSION = (
    "frozen_surface_official_descriptor_plus_context_film_residual_"
    "full_fit_gradient_calibrated_control_risk_v1"
)

PILOT_SEED = 0
# The frozen promoted control has hidden_dim=256 (for example
# feature_encoder.1.weight is [256,1280]).  The earlier 823,041 estimate was
# for V2's constructor default hidden_dim=128 and is not this checkpoint
# lineage.  Preserve the actual control instead of changing it to hit an old
# parameter estimate.
SUMMARY_HIDDEN_DIM = 256
TRAINABLE_PARAMETER_COUNT = 856_065
DESCRIPTOR_DIM = 1536
BOTTLENECK_DIM = 256
INITIAL_GATE = 0.1
EPOCHS = 30
PATIENCE = 8
TARGET_BATCH_ROWS = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

FIT_CVAR_TAIL_FRACTION = 0.10
FIT_GLOBAL_CVAR_TOLERANCE = 0.005
FIT_WORST_SCENE_MEAN_TOLERANCE = 0.010
FIT_WORST_SCENE_CVAR_TOLERANCE = 0.010
FIT_UNARY_DELTA_TOLERANCE = 0.0
FIT_L1_HINGE_WEIGHT = 1.0

# A finite value is required by the pure conjunction builder.  This sentinel
# is strictly outside its admissible range and never purports to be a replay.
MISSING_POINT_RENDER_REPLAY_SENTINEL = 1.0


def training_contract() -> dict[str, Any]:
    """Return the single, fixed seed-0 training and stopping contract."""

    return {
        "schema_version": 1,
        "seed": PILOT_SEED,
        "architecture": {
            "name": "surface_region_dual_descriptor_v1",
            "one_architecture_only": True,
            "summary_hidden_dim": SUMMARY_HIDDEN_DIM,
            "descriptor_dim": DESCRIPTOR_DIM,
            "bottleneck_dim": BOTTLENECK_DIM,
            "initial_gate": INITIAL_GATE,
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "trainable_modules": [
                "context_norm",
                "context_projection",
                "film",
                "gate",
            ],
            "parameter_count_lineage_correction": {
                "earlier_estimate": 823_041,
                "earlier_estimate_assumed_hidden_dim": 128,
                "frozen_promoted_control_hidden_dim": SUMMARY_HIDDEN_DIM,
                "correct_count_for_frozen_control": TRAINABLE_PARAMETER_COUNT,
                "decision": "preserve_frozen_control_instead_of_changing_base",
            },
        },
        "immutable_controls": {
            "surface_region_summary_readout_v2": True,
            "official_siglip2_summary_head": True,
            "official_summary_token": True,
            "official_descriptor": True,
            "primitive_feature_field": True,
        },
        "data": {
            "fit_caches": "frozen_query_free_scene_disjoint",
            "dev_caches": "frozen_query_free_scene_disjoint",
            "text_bank": "frozen_target_blind_fit_only_vocabulary",
            "generic_crop_teachers": "frozen_official_crop_summaries",
            "benchmark_vocabulary_images_masks_or_targets": False,
        },
        "objective": {
            "formula": "L_struct+lambda_unary*L_unary+lambda_risk*L_risk",
            "structural": "L_all_view+0.1*L_relation_gram_smooth_l1",
            "risk": ("control_referenced_full_fit_mean_plus_unsmoothed_l1_hinges"),
            "mathematical_exact_penalty_guarantee": False,
            "one_complete_fit_set_per_optimizer_step": True,
        },
        "epoch0_gradient_calibration": {
            "kind": "deterministic_gradient_norm_calibration_not_search",
            "weighted_auxiliary_to_structural_gradient_ratio": 0.25,
            "weights_fixed_after_epoch_zero": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience": PATIENCE,
        },
        "selection": {
            "domain": "frozen_dev_aggregates_only",
            "fit_constraints_required": True,
            "primary": "minimum_dev_normalized_mean_delta",
            "metric_continuation": False,
            "additional_seed_after_failure": False,
        },
        "advance_gate": {
            "implementation": "build_seed0_single_conjunction_gate",
            "single_conjunction": True,
            "thresholds": {
                "dev_normalized_mean_delta_max": DEV_MEAN_DELTA_MAX,
                "dev_global_cvar10_delta_max": DEV_GLOBAL_CVAR_DELTA_MAX,
                "dev_worst_scene_mean_delta_max": DEV_WORST_SCENE_MEAN_DELTA_MAX,
                "dev_worst_scene_cvar10_delta_max": (DEV_WORST_SCENE_CVAR_DELTA_MAX),
                "point_render_max_abs_error_max": POINT_RENDER_MAX_ABS_ERROR,
            },
            "point_render_replay": (
                "explicit_independent_materializer_and_frozen_scalar_"
                "compositor_evidence_required"
            ),
            "missing_replay_evidence": "fail_closed_pending_finalization",
        },
        "evaluation_protocol": {
            "freeze_id": legacy_pilot.EXPECTED_FREEZE_ID,
            "freeze_sha256": legacy_pilot.EXPECTED_FREEZE_SHA256,
            "scope": UNOPENED_SCOPE,
        },
    }


def _scene_groups_in_order(
    scene_ids: Sequence[str],
) -> tuple[list[str], list[torch.Tensor]]:
    groups: dict[str, list[int]] = {}
    for row, raw_scene in enumerate(scene_ids):
        scene = str(raw_scene)
        if not scene:
            raise ValueError("scene IDs must be non-empty")
        groups.setdefault(scene, []).append(row)
    if not groups or any(len(rows) < 2 for rows in groups.values()):
        raise ValueError("every scene must contain at least two rows")
    return list(groups), [
        torch.tensor(rows, dtype=torch.long) for rows in groups.values()
    ]


def _trainable_parameters(
    model: SurfaceRegionDualDescriptor,
) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if sum(parameter.numel() for parameter in parameters) != TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("dual-descriptor trainable parameter inventory differs")
    if model.trainable_parameter_count() != TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("dual-descriptor architecture parameter count differs")
    if any(
        parameter.requires_grad for parameter in model.summary_readout.parameters()
    ) or any(
        parameter.requires_grad
        for parameter in model.official_summary_head.parameters()
    ):
        raise RuntimeError("official Surface controls are not frozen")
    return parameters


def _adapter_state(
    model: SurfaceRegionDualDescriptor,
) -> dict[str, torch.Tensor]:
    prefixes = ("context_norm.", "context_projection.", "film.", "gate.")
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }
    if not state or sum(value.numel() for value in state.values()) != (
        TRAINABLE_PARAMETER_COUNT
    ):
        raise RuntimeError("dual-descriptor adapter state inventory differs")
    return state


def _load_adapter_state(
    model: SurfaceRegionDualDescriptor,
    state: Mapping[str, torch.Tensor],
) -> None:
    current = model.state_dict()
    if set(state) != {
        name
        for name in current
        if name.startswith(("context_norm.", "context_projection.", "film.", "gate."))
    }:
        raise ValueError("adapter state fields differ")
    merged = dict(current)
    merged.update(state)
    model.load_state_dict(merged, strict=True)


def _forward_all(
    model: SurfaceRegionDualDescriptor,
    data: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    official_tokens: list[torch.Tensor] = []
    official_descriptors: list[torch.Tensor] = []
    semantic_descriptors: list[torch.Tensor] = []
    for start in range(0, len(data["radio_features"]), TARGET_BATCH_ROWS):
        stop = min(start + TARGET_BATCH_ROWS, len(data["radio_features"]))
        output = model(
            data["radio_features"][start:stop].to(device),
            data["geometry"][start:stop].to(device),
            anchor_index=data["anchor_index"][start:stop].to(device),
            token_mask=data["token_mask"][start:stop].to(device),
            reliability=data["reliability"][start:stop].to(device),
        )
        official_tokens.append(output.official_token)
        official_descriptors.append(output.official_descriptor)
        semantic_descriptors.append(output.semantic_descriptor)
    return (
        torch.cat(official_tokens),
        torch.cat(official_descriptors),
        torch.cat(semantic_descriptors),
    )


def _all_targets(
    data: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(len(data["radio_features"]))
    token, descriptor, views, valid = _targets(data, rows)
    return (
        token.to(device),
        descriptor.to(device),
        views.to(device),
        valid.to(device),
    )


def _scene_query_units(
    semantic: torch.Tensor,
    teacher: torch.Tensor,
    views: torch.Tensor,
    view_valid: torch.Tensor,
    text_bank: torch.Tensor,
    scene_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    scene_names, scene_rows = _scene_groups_in_order(scene_ids)
    units: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    for scene, cpu_rows in zip(scene_names, scene_rows):
        rows = cpu_rows.to(semantic.device)
        scene_units, scene_valid, _ = (
            compute_uncertainty_weighted_scene_query_pairwise_gap_units(
                semantic.index_select(0, rows),
                teacher.index_select(0, rows),
                views.index_select(0, rows),
                view_valid.index_select(0, rows),
                text_bank,
                [scene] * len(cpu_rows),
                standard_error_multiplier=legacy_pilot.STANDARD_ERROR_MULTIPLIER,
                tie_tolerance=legacy_pilot.TIE_TOLERANCE,
                eps=legacy_pilot.EPS,
            )
        )
        if scene_units.shape[0] != 1:
            raise RuntimeError("one scene produced multiple scene/query rows")
        units.append(scene_units[0])
        valid.append(scene_valid[0])
    return torch.stack(units), torch.stack(valid)


def _fit_graph(
    model: SurfaceRegionDualDescriptor,
    data: Mapping[str, Any],
    text_bank: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    official_token, official_descriptor, semantic = _forward_all(model, data, device)
    target_token, teacher, views, view_valid = _all_targets(data, device)
    units, valid = _scene_query_units(
        semantic, teacher, views, view_valid, text_bank, data["scene_ids"]
    )
    return {
        "official_token": official_token,
        "official_descriptor": official_descriptor,
        "semantic": semantic,
        "target_token": target_token,
        "teacher": teacher,
        "views": views,
        "view_valid": view_valid,
        "units": units,
        "valid": valid,
    }


def _relative_delta(candidate: float, control: float, label: str) -> float:
    if not math.isfinite(float(candidate)) or not math.isfinite(float(control)):
        raise ValueError(f"{label} is non-finite")
    if float(control) <= 0.0:
        if math.isclose(float(candidate), float(control), abs_tol=1e-12):
            return 0.0
        raise ValueError(f"{label} control must be positive")
    return float(candidate) / float(control) - 1.0


@torch.no_grad()
def evaluate_split(
    model: SurfaceRegionDualDescriptor,
    data: Mapping[str, Any],
    text_bank: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, str]]:
    graph = _fit_graph(model, data, text_bank, device)
    semantic = graph["semantic"]
    teacher = graph["teacher"]
    views = graph["views"]
    view_valid = graph["view_valid"]
    student_response = semantic @ F.normalize(text_bank.float(), dim=-1).T
    teacher_response = teacher @ F.normalize(text_bank.float(), dim=-1).T
    pair = torch.einsum("bd,bvd->bv", semantic, views)
    surface = {
        "summary_token_cosine": float(
            F.cosine_similarity(graph["official_token"], graph["target_token"], dim=-1)
            .mean()
            .cpu()
        ),
        "mean_descriptor_cosine": float(
            F.cosine_similarity(semantic, teacher, dim=-1).mean().cpu()
        ),
        "all_view_descriptor_cosine": float(pair[view_valid].mean().cpu()),
    }
    metrics: dict[str, Any] = {
        **surface,
        "surface_selection_score": 0.5
        * (surface["mean_descriptor_cosine"] + surface["all_view_descriptor_cosine"]),
        "text_response_smooth_l1": float(
            F.smooth_l1_loss(student_response, teacher_response).cpu()
        ),
        "text_response_mae": float(
            (student_response - teacher_response).abs().mean().cpu()
        ),
    }
    controls = {
        "official_token_sha256": tensor_sha256(graph["official_token"].cpu()),
        "official_descriptor_sha256": tensor_sha256(graph["official_descriptor"].cpu()),
    }
    return metrics, graph["units"].cpu(), graph["valid"].cpu(), controls


def _selector(
    units: torch.Tensor,
    valid: torch.Tensor,
    control_units: torch.Tensor,
    control_valid: torch.Tensor,
    scenes: Sequence[str],
) -> dict[str, Any]:
    selector, _ = legacy_pilot.continuous_selector_metrics(
        units, valid, control_units, control_valid, scenes
    )
    return selector


def _fit_constraint_checks(
    selector: Mapping[str, Any], unary_relative_delta: float
) -> dict[str, bool]:
    return {
        "global_cvar10_delta": float(selector["normalized_upper_cvar10_delta"])
        <= FIT_GLOBAL_CVAR_TOLERANCE + 1e-12,
        "worst_scene_mean_delta": float(selector["worst_scene_mean_delta"])
        <= FIT_WORST_SCENE_MEAN_TOLERANCE + 1e-12,
        "worst_scene_cvar10_delta": float(selector["worst_scene_upper_cvar10_delta"])
        <= FIT_WORST_SCENE_CVAR_TOLERANCE + 1e-12,
        "independent_unary_delta": float(unary_relative_delta)
        <= FIT_UNARY_DELTA_TOLERANCE + 1e-12,
    }


def _record(
    *,
    epoch: int,
    fit_metrics: Mapping[str, Any],
    dev_metrics: Mapping[str, Any],
    fit_selector: Mapping[str, Any],
    dev_selector: Mapping[str, Any],
    control_fit_metrics: Mapping[str, Any],
    control_dev_metrics: Mapping[str, Any],
    adapter_sha256: str,
    official_token_bitwise_equal: bool,
    official_descriptor_bitwise_equal: bool,
    training_objective: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fit_unary = _relative_delta(
        float(fit_metrics["text_response_smooth_l1"]),
        float(control_fit_metrics["text_response_smooth_l1"]),
        "fit unary smooth_l1",
    )
    unary_deltas = {
        name: _relative_delta(
            float(dev_metrics[name]), float(control_dev_metrics[name]), name
        )
        for name in ("text_response_smooth_l1", "text_response_mae")
    }
    descriptor_deltas = {
        name: float(dev_metrics[name]) - float(control_dev_metrics[name])
        for name in (
            "summary_token_cosine",
            "mean_descriptor_cosine",
            "all_view_descriptor_cosine",
        )
    }
    checks = _fit_constraint_checks(fit_selector, fit_unary)
    return {
        "epoch": int(epoch),
        "adapter_state_dict_sha256": adapter_sha256,
        "fit": dict(fit_metrics),
        "dev": dict(dev_metrics),
        "fit_control_referenced_selector": dict(fit_selector),
        "dev_control_referenced_selector": dict(dev_selector),
        "fit_independent_unary_relative_delta": fit_unary,
        "validation_unary_relative_deltas": unary_deltas,
        "validation_descriptor_deltas": descriptor_deltas,
        "fit_constraint_checks": checks,
        "fit_constraints_feasible": all(checks.values()),
        "official_token_bitwise_equal": bool(official_token_bitwise_equal),
        "official_descriptor_bitwise_equal": bool(official_descriptor_bitwise_equal),
        "training_objective_before_step": (
            None if training_objective is None else dict(training_objective)
        ),
    }


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    if not history or [row.get("epoch") for row in history] != list(
        range(len(history))
    ):
        raise ValueError("dual-descriptor history must be contiguous from epoch zero")
    eligible = [row for row in history if row.get("fit_constraints_feasible") is True]
    if not eligible:
        raise RuntimeError("no fit-constraint-feasible control exists")

    def rank(row: Mapping[str, Any]) -> tuple[float, ...]:
        selector = row["dev_control_referenced_selector"]
        unary = row["validation_unary_relative_deltas"]
        return (
            -float(selector["normalized_mean_delta"]),
            -float(selector["normalized_upper_cvar10_delta"]),
            -float(selector["worst_scene_mean_delta"]),
            -float(selector["worst_scene_upper_cvar10_delta"]),
            -float(unary["text_response_smooth_l1"]),
            -float(unary["text_response_mae"]),
            float(row["dev"]["surface_selection_score"]),
        )

    return int(max(eligible, key=rank)["epoch"])


def _validated_point_render_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    adapter_state_dict_sha256: str,
) -> tuple[float, dict[str, Any]]:
    if evidence is None:
        return MISSING_POINT_RENDER_REPLAY_SENTINEL, {
            "status": "missing_pending_independent_materializer_replay",
            "evidence_present": False,
            "candidate_adapter_state_dict_sha256": adapter_state_dict_sha256,
        }
    required = {
        "schema_version",
        "artifact_type",
        "candidate_adapter_state_dict_sha256",
        "independent_materializer_replay",
        "frozen_scalar_compositor_replay",
        "point_render_replay_max_abs_error",
    }
    if set(evidence) != required:
        raise ValueError("point/render replay evidence schema differs")
    value = float(evidence["point_render_replay_max_abs_error"])
    if (
        evidence["schema_version"] != 1
        or evidence["artifact_type"] != "dual_descriptor_point_render_replay_evidence"
        or evidence["candidate_adapter_state_dict_sha256"] != adapter_state_dict_sha256
        or evidence["independent_materializer_replay"] is not True
        or evidence["frozen_scalar_compositor_replay"] is not True
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError("point/render replay evidence differs")
    return value, {**dict(evidence), "evidence_present": True, "status": "validated"}


def build_pilot_gate(
    selected_record: Mapping[str, Any],
    *,
    point_render_replay_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adapter_sha = str(selected_record["adapter_state_dict_sha256"])
    point_error, evidence = _validated_point_render_evidence(
        point_render_replay_evidence,
        adapter_state_dict_sha256=adapter_sha,
    )
    dev_selector = selected_record["dev_control_referenced_selector"]
    gate = build_seed0_single_conjunction_gate(
        selected_epoch=int(selected_record["epoch"]),
        dev_normalized_mean_delta=float(dev_selector["normalized_mean_delta"]),
        dev_global_cvar10_delta=float(dev_selector["normalized_upper_cvar10_delta"]),
        dev_worst_scene_mean_delta=float(dev_selector["worst_scene_mean_delta"]),
        dev_worst_scene_cvar10_delta=float(
            dev_selector["worst_scene_upper_cvar10_delta"]
        ),
        validation_unary_relative_deltas=selected_record[
            "validation_unary_relative_deltas"
        ],
        validation_descriptor_deltas=selected_record["validation_descriptor_deltas"],
        official_token_bitwise_equal=bool(
            selected_record["official_token_bitwise_equal"]
        ),
        official_descriptor_bitwise_equal=bool(
            selected_record["official_descriptor_bitwise_equal"]
        ),
        fit_constraint_checks=selected_record["fit_constraint_checks"],
        point_render_max_abs_error=point_error,
    )
    gate["point_render_replay_evidence"] = evidence
    gate["finalization_status"] = (
        "finalized" if evidence["evidence_present"] is True else "pending"
    )
    return gate


def _objective_report(statistics: Mapping[str, Any]) -> dict[str, Any]:
    risk = statistics["control_referenced_risk_statistics"]
    return {
        "objective": float(torch.as_tensor(statistics["objective"]).cpu()),
        "structural_loss": float(torch.as_tensor(statistics["structural_loss"]).cpu()),
        "all_view_cosine_loss": float(
            torch.as_tensor(statistics["all_view_cosine_loss"]).cpu()
        ),
        "relation_gram_smooth_l1_loss": float(
            torch.as_tensor(statistics["relation_gram_smooth_l1_loss"]).cpu()
        ),
        "independent_unary_loss": float(
            torch.as_tensor(
                statistics["independent_normalized_cosine_response_smooth_l1_loss"]
            ).cpu()
        ),
        "control_referenced_risk": float(
            torch.as_tensor(statistics["control_referenced_risk"]).cpu()
        ),
        "global_mean_delta": float(torch.as_tensor(risk["global_mean_delta"]).cpu()),
        "global_upper_fractional_cvar10_delta": float(
            torch.as_tensor(risk["global_upper_fractional_cvar_delta"]).cpu()
        ),
        "worst_scene_mean_delta": float(
            torch.as_tensor(risk["worst_scene_mean_delta"]).cpu()
        ),
        "worst_scene_upper_fractional_cvar10_delta": float(
            torch.as_tensor(risk["worst_scene_upper_fractional_cvar_delta"]).cpu()
        ),
    }


def _calibrate(
    model: SurfaceRegionDualDescriptor,
    train_data: Mapping[str, Any],
    text_bank: torch.Tensor,
    device: torch.device,
    control_units: torch.Tensor,
    control_valid: torch.Tensor,
    control_unary_loss: float,
) -> tuple[float, float, dict[str, float]]:
    model.train()
    graph = _fit_graph(model, train_data, text_bank, device)
    structural, _all_view, _relation, unary = compute_dual_descriptor_loss_components(
        graph["semantic"],
        graph["teacher"],
        graph["views"],
        graph["view_valid"],
        text_bank,
    )
    risk, _ = compute_control_referenced_exact_hinge_risk(
        graph["units"],
        graph["valid"],
        control_units.to(device),
        control_valid.to(device),
        unary,
        control_unary_loss,
        cvar_tail_fraction=FIT_CVAR_TAIL_FRACTION,
        global_cvar_tolerance=FIT_GLOBAL_CVAR_TOLERANCE,
        worst_scene_mean_tolerance=FIT_WORST_SCENE_MEAN_TOLERANCE,
        worst_scene_cvar_tolerance=FIT_WORST_SCENE_CVAR_TOLERANCE,
        unary_delta_tolerance=FIT_UNARY_DELTA_TOLERANCE,
        exact_penalty_weight=FIT_L1_HINGE_WEIGHT,
        eps=1e-12,
    )
    return calibrate_epoch0_gradient_weights(
        structural, unary, risk, _trainable_parameters(model)
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if (
        output.exists()
        or output.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise FileExistsError("dual-descriptor checkpoint/report output must be new")
    repo_root = Path(__file__).resolve().parents[2]
    protocol_binding = build_binding(
        Path(args.evaluation_protocol_freeze),
        scope=UNOPENED_SCOPE,
        repo_root=repo_root,
    )
    if (
        protocol_binding["scope"] != UNOPENED_SCOPE
        or protocol_binding["task"] is not None
        or protocol_binding["freeze"]["freeze_id"] != legacy_pilot.EXPECTED_FREEZE_ID
        or protocol_binding["freeze"]["sha256"] != legacy_pilot.EXPECTED_FREEZE_SHA256
    ):
        raise ValueError("dual-descriptor evaluation-protocol binding differs")

    train_paths = _paths(args.train_caches)
    dev_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    dev_data, dev_meta = _load(dev_paths, "validation")
    _validate_train_validation_contracts(train_meta, dev_meta)
    if set(train_meta["scenes"]) & set(dev_meta["scenes"]):
        raise ValueError("fit/dev scenes overlap")
    if "scene_ids" not in train_data or "scene_ids" not in dev_data:
        raise ValueError("dual-descriptor caches require row-to-scene bindings")
    train_scenes, _ = _scene_groups_in_order(train_data["scene_ids"])
    dev_scenes, _ = _scene_groups_in_order(dev_data["scene_ids"])

    radio_path = Path(args.radio_checkpoint).resolve()
    radio_sha = _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    summary_readout, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=PILOT_SEED,
        train_paths=train_paths,
        validation_paths=dev_paths,
        train_meta=train_meta,
        validation_meta=dev_meta,
        hidden_dim=SUMMARY_HIDDEN_DIM,
        reliability_attention_mode="log_prior",
        context_pooling_mode="joint_attention_v1",
    )
    device = torch.device(args.device)
    _seed_training(PILOT_SEED, device=device)
    summary_readout = summary_readout.to(device).eval().requires_grad_(False)
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).to(device)
    head.eval().requires_grad_(False)
    model = SurfaceRegionDualDescriptor(
        summary_readout,
        head,
        descriptor_dim=DESCRIPTOR_DIM,
        bottleneck_dim=BOTTLENECK_DIM,
        initial_gate=INITIAL_GATE,
    ).to(device)
    _trainable_parameters(model)
    text_bank = fit_bank["embeddings"].to(device)
    base_summary_state = legacy_pilot._clone_state(summary_readout)
    base_summary_sha = state_dict_sha256(base_summary_state)
    initial_adapter_state = _adapter_state(model)
    initial_adapter_sha = state_dict_sha256(initial_adapter_state)
    architecture = model.architecture()
    if architecture["trainable_parameter_count"] != TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("dual-descriptor architecture digest count differs")

    model.eval()
    control_fit_metrics, control_fit_units, control_fit_valid, fit_official = (
        evaluate_split(model, train_data, text_bank, device)
    )
    control_dev_metrics, control_dev_units, control_dev_valid, dev_official = (
        evaluate_split(model, dev_data, text_bank, device)
    )
    control_fit_selector = _selector(
        control_fit_units,
        control_fit_valid,
        control_fit_units,
        control_fit_valid,
        train_scenes,
    )
    control_dev_selector = _selector(
        control_dev_units,
        control_dev_valid,
        control_dev_units,
        control_dev_valid,
        dev_scenes,
    )
    control_record = _record(
        epoch=0,
        fit_metrics=control_fit_metrics,
        dev_metrics=control_dev_metrics,
        fit_selector=control_fit_selector,
        dev_selector=control_dev_selector,
        control_fit_metrics=control_fit_metrics,
        control_dev_metrics=control_dev_metrics,
        adapter_sha256=initial_adapter_sha,
        official_token_bitwise_equal=True,
        official_descriptor_bitwise_equal=True,
        training_objective=None,
    )
    control_record.update(
        {
            "initialization": "exact_official_descriptor_zero_residual",
            "selection_updated_best": True,
            "fit_unit_loss_sha256": tensor_sha256(control_fit_units.float()),
            "fit_unit_valid_sha256": tensor_sha256(control_fit_valid),
            "dev_unit_loss_sha256": tensor_sha256(control_dev_units.float()),
            "dev_unit_valid_sha256": tensor_sha256(control_dev_valid),
        }
    )

    lambda_unary, lambda_risk, gradient_calibration = _calibrate(
        model,
        train_data,
        text_bank,
        device,
        control_fit_units,
        control_fit_valid,
        float(control_fit_metrics["text_response_smooth_l1"]),
    )
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, Any]] = [control_record]
    best_epoch = 0
    best_state = initial_adapter_state
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        graph = _fit_graph(model, train_data, text_bank, device)
        objective, objective_statistics = compute_dual_descriptor_response_risk(
            graph["units"],
            graph["valid"],
            control_fit_units.to(device),
            control_fit_valid.to(device),
            graph["semantic"],
            graph["teacher"],
            graph["views"],
            graph["view_valid"],
            text_bank,
            float(control_fit_metrics["text_response_smooth_l1"]),
            lambda_unary=lambda_unary,
            lambda_risk=lambda_risk,
            cvar_tail_fraction=FIT_CVAR_TAIL_FRACTION,
            global_cvar_tolerance=FIT_GLOBAL_CVAR_TOLERANCE,
            worst_scene_mean_tolerance=FIT_WORST_SCENE_MEAN_TOLERANCE,
            worst_scene_cvar_tolerance=FIT_WORST_SCENE_CVAR_TOLERANCE,
            unary_delta_tolerance=FIT_UNARY_DELTA_TOLERANCE,
            l1_hinge_weight=FIT_L1_HINGE_WEIGHT,
            eps=1e-12,
        )
        objective_report = _objective_report(objective_statistics)
        objective.backward()
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in _trainable_parameters(model)
        ):
            raise RuntimeError("dual-descriptor gradient is missing/non-finite")
        optimizer.step()

        model.eval()
        fit_metrics, fit_units, fit_valid, current_fit_official = evaluate_split(
            model, train_data, text_bank, device
        )
        dev_metrics, dev_units, dev_valid, current_dev_official = evaluate_split(
            model, dev_data, text_bank, device
        )
        official_token_equal = (
            current_fit_official["official_token_sha256"]
            == fit_official["official_token_sha256"]
            and current_dev_official["official_token_sha256"]
            == dev_official["official_token_sha256"]
        )
        official_descriptor_equal = (
            current_fit_official["official_descriptor_sha256"]
            == fit_official["official_descriptor_sha256"]
            and current_dev_official["official_descriptor_sha256"]
            == dev_official["official_descriptor_sha256"]
        )
        state = _adapter_state(model)
        adapter_sha = state_dict_sha256(state)
        fit_selector = _selector(
            fit_units,
            fit_valid,
            control_fit_units,
            control_fit_valid,
            train_scenes,
        )
        dev_selector = _selector(
            dev_units,
            dev_valid,
            control_dev_units,
            control_dev_valid,
            dev_scenes,
        )
        record = _record(
            epoch=epoch,
            fit_metrics=fit_metrics,
            dev_metrics=dev_metrics,
            fit_selector=fit_selector,
            dev_selector=dev_selector,
            control_fit_metrics=control_fit_metrics,
            control_dev_metrics=control_dev_metrics,
            adapter_sha256=adapter_sha,
            official_token_bitwise_equal=official_token_equal,
            official_descriptor_bitwise_equal=official_descriptor_equal,
            training_objective=objective_report,
        )
        record.update(
            {
                "fit_unit_loss_sha256": tensor_sha256(fit_units.float()),
                "fit_unit_valid_sha256": tensor_sha256(fit_valid),
                "dev_unit_loss_sha256": tensor_sha256(dev_units.float()),
                "dev_unit_valid_sha256": tensor_sha256(dev_valid),
            }
        )
        selected_epoch = select_best_epoch([*history, record])
        updated = selected_epoch == epoch
        record["selection_updated_best"] = updated
        if updated:
            best_epoch = epoch
            best_state = state
            stale = 0
        else:
            if selected_epoch != best_epoch:
                raise RuntimeError("best epoch changed retroactively")
            stale += 1
        record["patience_stale"] = stale
        record["patience_stop"] = stale >= PATIENCE
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= PATIENCE:
            break

    if select_best_epoch(history) != best_epoch:
        raise RuntimeError("online/final dual-descriptor selection differs")
    _load_adapter_state(model, best_state)
    final_dev_metrics, final_dev_units, final_dev_valid, final_dev_official = (
        evaluate_split(model, dev_data, text_bank, device)
    )
    if (
        final_dev_metrics != history[best_epoch]["dev"]
        or tensor_sha256(final_dev_units.float())
        != history[best_epoch]["dev_unit_loss_sha256"]
        or tensor_sha256(final_dev_valid)
        != history[best_epoch]["dev_unit_valid_sha256"]
        or final_dev_official != dev_official
    ):
        raise RuntimeError("selected dual-descriptor replay differs")
    best_adapter_sha = state_dict_sha256(best_state)
    if best_adapter_sha != history[best_epoch]["adapter_state_dict_sha256"]:
        raise RuntimeError("selected adapter digest differs")

    # Training cannot create independent point/render evidence for itself.
    # Therefore this invocation always stops with a fail-closed pending replay.
    gate = build_pilot_gate(history[best_epoch])
    if gate["passed"] is not False:
        raise RuntimeError("training-only artifact cannot pass replay-dependent gate")
    non_replay_checks_passed = all(
        value
        for name, value in gate["checks"].items()
        if name != "point_render_max_abs_error_le_1e_minus_6"
    )
    gate_status = (
        "training_complete_pending_point_render_replay"
        if non_replay_checks_passed
        else "training_complete_seed0_gate_failed"
    )

    implementation_paths = (
        Path(__file__),
        repo_root / "radio_gs/models/surface_region_dual_descriptor.py",
        repo_root / "radio_gs/losses/dual_descriptor_response_risk.py",
        repo_root / "radio_gs/losses/control_referenced_uncertainty_response_risk.py",
        repo_root / "radio_gs/losses/uncertainty_response_risk.py",
        repo_root / "radio_gs/losses/direct_point_query_logit_distill_loss.py",
        repo_root / "radio_gs/interfaces/surface_region_summary.py",
        repo_root / "radio_gs/scripts/train_surface_region_summary_readout.py",
        repo_root / "radio_gs/scripts/train_surface_region_text_response_distill.py",
        repo_root / "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm_version": ALGORITHM_VERSION,
        "training_complete": True,
        "gate_status": gate_status,
        "pilot_advance_gate_passed": False,
        "continuation_authorized": False,
        "seed1_executed": False,
        "additional_seed_or_architecture_authorized": False,
        "base_surface_state_dict": base_summary_state,
        "base_surface_state_dict_sha256": base_summary_sha,
        "dual_descriptor_architecture": architecture,
        "adapter_state_dict": best_state,
        "adapter_state_dict_sha256": best_adapter_sha,
        "best_epoch": best_epoch,
        "history": history,
        "gradient_calibration": gradient_calibration,
        "lambda_unary": lambda_unary,
        "lambda_risk": lambda_risk,
        "seed0_single_conjunction_gate": gate,
        "control": {
            "fit_metrics": control_fit_metrics,
            "dev_metrics": control_dev_metrics,
            "fit_unit_loss_sha256": tensor_sha256(control_fit_units.float()),
            "fit_unit_valid_sha256": tensor_sha256(control_fit_valid),
            "dev_unit_loss_sha256": tensor_sha256(control_dev_units.float()),
            "dev_unit_valid_sha256": tensor_sha256(control_dev_valid),
            "fit_official_output_hashes": fit_official,
            "dev_official_output_hashes": dev_official,
        },
        "provenance": {
            "evaluation_protocol": protocol_binding,
            "scope": UNOPENED_SCOPE,
            "formal_authority": False,
            "pilot_only": True,
            "external_benchmarks_opened": False,
            "benchmark_vocabulary_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_targets_opened": False,
            "metric_continuation": False,
            "fit_split_only_for_optimizer": True,
            "dev_split_used_for_selection_only": True,
            "surface_control": surface_control,
            "train_caches": _cache_binding(train_paths),
            "validation_caches": _cache_binding(dev_paths),
            "fit_text_bank": _fit_bank_binding(fit_bank),
            "radio_checkpoint": {"path": str(radio_path), "sha256": radio_sha},
            "train_contract": train_meta,
            "validation_contract": dev_meta,
            "implementation_sources": [
                legacy_pilot._file_record(path) for path in implementation_paths
            ],
        },
        "training_contract": training_contract(),
        "final_validation": final_dev_metrics,
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
        "training_complete": True,
        "gate_status": gate_status,
        "pilot_advance_gate_passed": False,
        "continuation_authorized": False,
        "seed1_executed": False,
        "additional_seed_or_architecture_authorized": False,
        "evaluation_protocol": protocol_binding,
        "scope": UNOPENED_SCOPE,
        "external_benchmarks_opened": False,
        "base_surface_state_dict_sha256": base_summary_sha,
        "dual_descriptor_architecture": architecture,
        "adapter_state_dict_sha256": best_adapter_sha,
        "best_epoch": best_epoch,
        "selected_history_record": history[best_epoch],
        "gradient_calibration": gradient_calibration,
        "lambda_unary": lambda_unary,
        "lambda_risk": lambda_risk,
        "seed0_single_conjunction_gate": gate,
        "history_length": len(history),
        "training_contract": training_contract(),
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
