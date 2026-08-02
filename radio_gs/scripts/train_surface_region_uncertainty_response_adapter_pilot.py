#!/usr/bin/env python3
"""Train the fit-only seed-0 uncertainty response-adapter v5 pilot.

This is deliberately separate from every registered Surface text-response
trainer and promotion path.  It opens only query-free Surface train/validation
caches and the frozen target-blind *fit* vocabulary.  The seed-0 Surface
readout and official SigLIP summary head remain frozen; only a bounded
low-rank adapter before the official head is optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
    compute_multiview_teacher_response_uncertainty,
    compute_scene_wise_uncertainty_weighted_text_response_pairwise_gap_smooth_l1_loss,
    fractional_upper_cvar,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.models.surface_text_response_adapter import (
    LowRankTangentSummaryAdapter,
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
    complete_scene_batches,
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
ARTIFACT_TYPE = "surface_region_uncertainty_response_adapter_seed0_pilot"
ALGORITHM_VERSION = (
    "frozen_surface_low_rank_tangent_adapter_multiview_uncertainty_cvar_v5"
)
PILOT_SEED = 0
EXPECTED_FREEZE_ID = "evaluation_protocols_20260801_v1"
EXPECTED_FREEZE_SHA256 = (
    "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916"
)

ADAPTER_RANK = 32
ADAPTER_MAX_ANGLE_DEGREES = 0.1
EPOCHS = 30
PATIENCE = 8
TARGET_BATCH_ROWS = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PAIRWISE_WEIGHT = 1.0
INDEPENDENT_RESPONSE_WEIGHT = 0.25
SURFACE_DESCRIPTOR_WEIGHT = 0.25
STANDARD_ERROR_MULTIPLIER = 2.0
TIE_TOLERANCE = 1e-6
EPS = 1e-6

SURFACE_NONINFERIORITY_TOLERANCE = 0.002
CVAR_TAIL_FRACTION = 0.10
AGGREGATE_MEAN_DELTA_TOLERANCE = 0.005
GLOBAL_CVAR_DELTA_TOLERANCE = 0.010
PER_SCENE_MEAN_DELTA_TOLERANCE = 0.010
PER_SCENE_CVAR_DELTA_TOLERANCE = 0.020
PILOT_REQUIRED_MEAN_IMPROVEMENT = 0.0025
PILOT_GLOBAL_CVAR_TOLERANCE = 0.005
PILOT_PER_SCENE_CVAR_TOLERANCE = 0.010
ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES = 1e-5

SURFACE_FIELDS = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
)


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_record(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve(strict=True)
    return {"path": str(source), "sha256": sha256_file(source)}


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def combined_state_sha256(
    base_state_sha256: str,
    adapter_state_sha256: str,
    adapter_architecture_digest: str,
) -> str:
    for label, value in (
        ("base_state_sha256", base_state_sha256),
        ("adapter_state_sha256", adapter_state_sha256),
        ("adapter_architecture_digest", adapter_architecture_digest),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} must be a lowercase SHA-256")
    return _canonical_json_sha256(
        {
            "base_surface_state_dict_sha256": base_state_sha256,
            "response_adapter_state_dict_sha256": adapter_state_sha256,
            "response_adapter_architecture_digest": (
                adapter_architecture_digest
            ),
        }
    )


def adapter_angle_statistics(
    base_tokens: torch.Tensor,
    adapted_tokens: torch.Tensor,
    *,
    max_angle_degrees: float,
) -> dict[str, float]:
    left = torch.as_tensor(base_tokens).detach().cpu().double()
    right = torch.as_tensor(adapted_tokens).detach().cpu().double()
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] == 0:
        raise ValueError("base/adapted tokens must be aligned non-empty [B,D]")
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("base/adapted tokens must be finite")
    left = F.normalize(left, dim=-1, eps=1e-12)
    right = F.normalize(right, dim=-1, eps=1e-12)
    angles = torch.rad2deg(
        torch.acos((left * right).sum(dim=-1).clamp(-1.0, 1.0))
    )
    cap = float(max_angle_degrees)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("max_angle_degrees must be finite and positive")
    return {
        "mean_degrees": float(angles.mean()),
        "max_degrees": float(angles.max()),
        "saturation_ratio_at_99pct_cap": float(
            (angles >= 0.99 * cap).double().mean()
        ),
    }


def _scene_order(scene_ids: Sequence[str]) -> list[str]:
    ordered = list(dict.fromkeys(str(value) for value in scene_ids))
    if any(not value for value in ordered):
        raise ValueError("scene IDs must be non-empty")
    return ordered


@torch.no_grad()
def uncertainty_weight_statistics(
    teacher_descriptors: torch.Tensor,
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    text_bank: torch.Tensor,
    scene_ids: Sequence[str],
) -> dict[str, float | int]:
    teacher = F.normalize(torch.as_tensor(teacher_descriptors).float(), dim=-1)
    text = F.normalize(torch.as_tensor(text_bank).float(), dim=-1)
    if len(scene_ids) != teacher.shape[0]:
        raise ValueError("uncertainty statistics scene IDs are misaligned")
    uncertainty = compute_multiview_teacher_response_uncertainty(
        teacher_view_descriptors,
        teacher_mask,
        text,
        eps=EPS,
    )
    response = teacher @ text.T
    standard_error = uncertainty["response_standard_error"].to(response.device)
    weights: list[torch.Tensor] = []
    for scene in _scene_order(scene_ids):
        indices = torch.tensor(
            [index for index, value in enumerate(scene_ids) if str(value) == scene],
            dtype=torch.long,
            device=response.device,
        )
        if indices.numel() < 2:
            raise ValueError("every pilot scene must have at least two regions")
        local_response = response.index_select(0, indices)
        local_standard_error = standard_error.index_select(0, indices)
        pairs = torch.triu_indices(
            len(indices), len(indices), offset=1, device=response.device
        )
        gaps = local_response[pairs[0]] - local_response[pairs[1]]
        pair_standard_error = torch.sqrt(
            local_standard_error[pairs[0]].square()
            + local_standard_error[pairs[1]].square()
        )
        span = local_response.amax(dim=0) - local_response.amin(dim=0)
        valid = (gaps.abs() > TIE_TOLERANCE) & (
            span > TIE_TOLERANCE
        ).unsqueeze(0)
        confidence = gaps.abs() / (
            gaps.abs()
            + STANDARD_ERROR_MULTIPLIER * pair_standard_error
            + EPS
        )
        if bool(valid.any()):
            weights.append(confidence[valid].cpu())
    if not weights:
        raise ValueError("pilot uncertainty statistics have no valid pair-query")
    values = torch.cat(weights).float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
        "fraction_below_0p5": float((values < 0.5).float().mean()),
        "teacher_response_variance_mean": float(
            uncertainty["response_variance"].mean()
        ),
    }


def continuous_selector_metrics(
    candidate_unit_loss: torch.Tensor,
    candidate_valid: torch.Tensor,
    control_unit_loss: torch.Tensor,
    control_valid: torch.Tensor,
    scene_names: Sequence[str],
) -> tuple[dict[str, Any], torch.Tensor]:
    candidate = torch.as_tensor(candidate_unit_loss).detach().cpu().float()
    control = torch.as_tensor(control_unit_loss).detach().cpu().float()
    candidate_mask = torch.as_tensor(candidate_valid).detach().cpu().bool()
    control_mask = torch.as_tensor(control_valid).detach().cpu().bool()
    if (
        candidate.shape != control.shape
        or candidate_mask.shape != candidate.shape
        or not torch.equal(candidate_mask, control_mask)
        or candidate.ndim != 2
        or candidate.shape[0] != len(scene_names)
        or not bool(candidate_mask.any())
        or not bool(torch.isfinite(candidate).all())
        or not bool(torch.isfinite(control).all())
    ):
        raise ValueError("continuous selector units are invalid or misaligned")
    denominator = float(control[control_mask].mean())
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("continuous selector control scale is degenerate")
    delta = (candidate - control) / denominator
    active_delta = delta[candidate_mask]
    global_mean = float(active_delta.mean())
    global_cvar = float(
        fractional_upper_cvar(active_delta, CVAR_TAIL_FRACTION)
    )
    per_scene: dict[str, dict[str, float]] = {}
    for index, raw_scene in enumerate(scene_names):
        scene = str(raw_scene)
        values = delta[index][candidate_mask[index]]
        if values.numel() == 0 or scene in per_scene:
            raise ValueError("continuous selector scenes are empty or repeated")
        per_scene[scene] = {
            "mean_delta": float(values.mean()),
            "upper_cvar10_delta": float(
                fractional_upper_cvar(values, CVAR_TAIL_FRACTION)
            ),
        }
    return (
        {
            "control_scale_mean_unit_loss": denominator,
            "candidate_mean_unit_loss": float(candidate[candidate_mask].mean()),
            "normalized_mean_delta": global_mean,
            "normalized_upper_cvar10_delta": global_cvar,
            "worst_scene_mean_delta": max(
                value["mean_delta"] for value in per_scene.values()
            ),
            "worst_scene_upper_cvar10_delta": max(
                value["upper_cvar10_delta"] for value in per_scene.values()
            ),
            "per_scene": per_scene,
            "unit_count": int(candidate_mask.sum()),
        },
        delta,
    )


def annotate_selection_record(
    record: Mapping[str, Any],
    *,
    control_record: Mapping[str, Any],
    selector: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(record)
    surface_deltas = {
        field: float(value[field]) - float(control_record[field])
        for field in SURFACE_FIELDS
    }
    surface_feasible = all(
        delta >= -SURFACE_NONINFERIORITY_TOLERANCE - 1e-12
        for delta in surface_deltas.values()
    )
    continuous_feasible = (
        float(selector["normalized_mean_delta"])
        <= AGGREGATE_MEAN_DELTA_TOLERANCE + 1e-12
        and float(selector["normalized_upper_cvar10_delta"])
        <= GLOBAL_CVAR_DELTA_TOLERANCE + 1e-12
        and float(selector["worst_scene_mean_delta"])
        <= PER_SCENE_MEAN_DELTA_TOLERANCE + 1e-12
        and float(selector["worst_scene_upper_cvar10_delta"])
        <= PER_SCENE_CVAR_DELTA_TOLERANCE + 1e-12
    )
    value.update(
        {
            "surface_control_deltas": surface_deltas,
            "surface_control_feasible": surface_feasible,
            "continuous_selector": dict(selector),
            "continuous_selector_feasible": continuous_feasible,
            "selection_feasible": surface_feasible and continuous_feasible,
        }
    )
    return value


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    if not history or [row.get("epoch") for row in history] != list(
        range(len(history))
    ):
        raise ValueError("pilot history must be contiguous from epoch zero")
    eligible = [row for row in history if row.get("selection_feasible") is True]
    if not eligible:
        raise RuntimeError("pilot selector has no feasible control")

    def rank(row: Mapping[str, Any]) -> tuple[float, ...]:
        selector = row["continuous_selector"]
        return (
            -float(selector["candidate_mean_unit_loss"]),
            -float(selector["normalized_upper_cvar10_delta"]),
            -float(selector["worst_scene_upper_cvar10_delta"]),
            -float(row["text_response_smooth_l1"]),
            -float(row["text_response_mae"]),
            float(row["surface_selection_score"]),
        )

    return int(max(eligible, key=rank)["epoch"])


def _selector_contract() -> dict[str, Any]:
    return {
        "name": "paired_continuous_weighted_gap_fractional_cvar_v1",
        "tail_fraction": CVAR_TAIL_FRACTION,
        "delta_normalization": "control_global_mean_scene_query_unit_loss",
        "surface_noninferiority_tolerance": (
            SURFACE_NONINFERIORITY_TOLERANCE
        ),
        "aggregate_mean_delta_tolerance": AGGREGATE_MEAN_DELTA_TOLERANCE,
        "global_cvar_delta_tolerance": GLOBAL_CVAR_DELTA_TOLERANCE,
        "per_scene_mean_delta_tolerance": PER_SCENE_MEAN_DELTA_TOLERANCE,
        "per_scene_cvar_delta_tolerance": PER_SCENE_CVAR_DELTA_TOLERANCE,
        "discrete_rank_metrics_are_feasibility_gates": False,
    }


def _objective_contract() -> dict[str, Any]:
    return {
        "total": (
            "uncertainty_pairwise+0.25*independent_response+"
            "0.25*all_view_surface_descriptor"
        ),
        "uncertainty_pairwise_weight": PAIRWISE_WEIGHT,
        "independent_response_weight": INDEPENDENT_RESPONSE_WEIGHT,
        "surface_descriptor_weight": SURFACE_DESCRIPTOR_WEIGHT,
        "standard_error_multiplier": STANDARD_ERROR_MULTIPLIER,
        "tie_tolerance": TIE_TOLERANCE,
        "teacher_side_autograd": "detached_teacher_variance_weights_text_bank",
        "vocabulary": "target_blind_fit_only",
    }


def _training_contract() -> dict[str, Any]:
    return {
        "seed": PILOT_SEED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "target_batch_rows": TARGET_BATCH_ROWS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "persistent_adamw_adapter_parameters_only",
        "base_surface_readout": "frozen_exact_seed0_control",
        "official_siglip_summary_head": "frozen",
        "adapter": {
            "rank": ADAPTER_RANK,
            "max_angle_degrees": ADAPTER_MAX_ANGLE_DEGREES,
            "location": "before_frozen_official_siglip_summary_head",
        },
        "objective": _objective_contract(),
        "selector": _selector_contract(),
    }


@torch.no_grad()
def _precompute_base_tokens(
    model: torch.nn.Module,
    data: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(data["radio_features"]), TARGET_BATCH_ROWS):
        stop = min(start + TARGET_BATCH_ROWS, len(data["radio_features"]))
        output = model(
            data["radio_features"][start:stop].to(device),
            data["geometry"][start:stop].to(device),
            anchor_index=data["anchor_index"][start:stop].to(device),
            token_mask=data["token_mask"][start:stop].to(device),
            reliability=data["reliability"][start:stop].to(device),
        )
        outputs.append(output.detach())
    return torch.cat(outputs)


@torch.no_grad()
def _evaluate(
    adapter: LowRankTangentSummaryAdapter,
    head: torch.nn.Module,
    base_tokens: torch.Tensor,
    data: Mapping[str, Any],
    text_bank: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    adapted_parts: list[torch.Tensor] = []
    descriptor_parts: list[torch.Tensor] = []
    target_token_parts: list[torch.Tensor] = []
    target_descriptor_parts: list[torch.Tensor] = []
    all_descriptor_parts: list[torch.Tensor] = []
    teacher_mask_parts: list[torch.Tensor] = []
    for start in range(0, len(base_tokens), TARGET_BATCH_ROWS):
        stop = min(start + TARGET_BATCH_ROWS, len(base_tokens))
        rows = torch.arange(start, stop)
        target_token, target_descriptor, all_descriptors, teacher_mask = _targets(
            data, rows
        )
        adapted = adapter(base_tokens[start:stop])
        descriptor = F.normalize(
            head(adapted[:, None])[:, 0].float(), dim=-1, eps=1e-8
        )
        adapted_parts.append(adapted)
        descriptor_parts.append(descriptor)
        target_token_parts.append(target_token.to(adapted.device))
        target_descriptor_parts.append(target_descriptor.to(adapted.device))
        all_descriptor_parts.append(all_descriptors.to(adapted.device))
        teacher_mask_parts.append(teacher_mask.to(adapted.device))
    adapted = torch.cat(adapted_parts)
    student = torch.cat(descriptor_parts)
    target_token = torch.cat(target_token_parts)
    teacher = torch.cat(target_descriptor_parts)
    all_descriptors = torch.cat(all_descriptor_parts)
    teacher_mask = torch.cat(teacher_mask_parts)
    pair = torch.einsum("bd,bvd->bv", student, all_descriptors)
    pair_loss, pair_stats = (
        compute_scene_wise_uncertainty_weighted_text_response_pairwise_gap_smooth_l1_loss(
            student,
            teacher,
            all_descriptors,
            teacher_mask,
            text_bank,
            data["scene_ids"],
            standard_error_multiplier=STANDARD_ERROR_MULTIPLIER,
            tie_tolerance=TIE_TOLERANCE,
            eps=EPS,
        )
    )
    student_response = student @ F.normalize(text_bank.float(), dim=-1).T
    teacher_response = teacher @ F.normalize(text_bank.float(), dim=-1).T
    surface = {
        "summary_token_cosine": float(
            F.cosine_similarity(adapted, target_token, dim=-1).mean().cpu()
        ),
        "mean_descriptor_cosine": float(
            F.cosine_similarity(student, teacher, dim=-1).mean().cpu()
        ),
        "all_view_descriptor_cosine": float(pair[teacher_mask].mean().cpu()),
    }
    angle_statistics = adapter_angle_statistics(
        base_tokens, adapted, max_angle_degrees=ADAPTER_MAX_ANGLE_DEGREES
    )
    if (
        angle_statistics["max_degrees"]
        > ADAPTER_MAX_ANGLE_DEGREES + ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES
    ):
        raise RuntimeError("response adapter exceeded its hard angular contract")
    return (
        {
            **surface,
            "surface_selection_score": 0.5
            * (
                surface["mean_descriptor_cosine"]
                + surface["all_view_descriptor_cosine"]
            ),
            "uncertainty_pairwise_gap_smooth_l1": float(pair_loss.cpu()),
            "text_response_smooth_l1": float(
                F.smooth_l1_loss(student_response, teacher_response).cpu()
            ),
            "text_response_mae": float(
                (student_response - teacher_response).abs().mean().cpu()
            ),
            "adapter_angle": angle_statistics,
            "uncertainty_weight_mean": float(
                pair_stats["uncertainty_weight_mean"].cpu()
            ),
        },
        pair_stats["scene_query_loss"].cpu(),
        pair_stats["scene_query_valid"].cpu(),
    )


def _all_teacher_targets(
    data: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(len(data["radio_features"]))
    _token, descriptor, all_descriptors, teacher_mask = _targets(data, rows)
    return descriptor, all_descriptors, teacher_mask


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("pilot checkpoint/report output must be new")
    repo_root = Path(__file__).resolve().parents[2]
    protocol_binding = build_binding(
        Path(args.evaluation_protocol_freeze),
        scope=UNOPENED_SCOPE,
        repo_root=repo_root,
    )
    if (
        protocol_binding["scope"] != UNOPENED_SCOPE
        or protocol_binding["task"] is not None
        or protocol_binding["freeze"]["freeze_id"] != EXPECTED_FREEZE_ID
        or protocol_binding["freeze"]["sha256"] != EXPECTED_FREEZE_SHA256
    ):
        raise ValueError("pilot evaluation-protocol freeze binding differs")

    train_paths = _paths(args.train_caches)
    validation_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    validation_data, validation_meta = _load(validation_paths, "validation")
    _validate_train_validation_contracts(train_meta, validation_meta)
    radio_path = Path(args.radio_checkpoint).resolve()
    radio_sha = _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    base_model, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=PILOT_SEED,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=256,
        reliability_attention_mode="log_prior",
        context_pooling_mode="joint_attention_v1",
    )
    if set(train_meta["scenes"]) & set(validation_meta["scenes"]):
        raise ValueError("pilot train/validation scenes overlap")
    if "scene_ids" not in train_data or "scene_ids" not in validation_data:
        raise ValueError("pilot caches require exact row-to-scene bindings")

    device = torch.device(args.device)
    generator = _seed_training(PILOT_SEED, device=device)
    base_model = base_model.to(device).eval().requires_grad_(False)
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).to(device).eval()
    head.requires_grad_(False)
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=1280,
        rank=ADAPTER_RANK,
        max_angle_degrees=ADAPTER_MAX_ANGLE_DEGREES,
    ).to(device)
    text_bank = fit_bank["embeddings"].to(device)
    base_train = _precompute_base_tokens(base_model, train_data, device)
    base_validation = _precompute_base_tokens(base_model, validation_data, device)
    base_state = _clone_state(base_model)
    base_state_sha = state_dict_sha256(base_state)
    if any(parameter.requires_grad for parameter in base_model.parameters()) or any(
        parameter.requires_grad for parameter in head.parameters()
    ):
        raise RuntimeError("pilot base readout/head freeze failed")

    train_teacher, train_views, train_teacher_mask = _all_teacher_targets(train_data)
    validation_teacher, validation_views, validation_teacher_mask = (
        _all_teacher_targets(validation_data)
    )
    uncertainty_statistics = {
        "train": uncertainty_weight_statistics(
            train_teacher,
            train_views,
            train_teacher_mask,
            fit_bank["embeddings"],
            train_data["scene_ids"],
        ),
        "validation": uncertainty_weight_statistics(
            validation_teacher,
            validation_views,
            validation_teacher_mask,
            fit_bank["embeddings"],
            validation_data["scene_ids"],
        ),
    }
    validation_scene_names = _scene_order(validation_data["scene_ids"])
    if len(validation_scene_names) != len(validation_meta["scenes"]):
        raise ValueError("validation scene order/count differs from cache metadata")

    adapter.eval()
    control_metrics, control_units, control_valid = _evaluate(
        adapter, head, base_validation, validation_data, text_bank
    )
    control_selector, _ = continuous_selector_metrics(
        control_units,
        control_valid,
        control_units,
        control_valid,
        validation_scene_names,
    )
    control_record = annotate_selection_record(
        {
            "epoch": 0,
            "initialization": "zero_up_identity_adapter",
            **control_metrics,
        },
        control_record={**control_metrics},
        selector=control_selector,
    )
    control_record["selection_updated_best"] = True
    initial_adapter_state = _clone_state(adapter)
    initial_adapter_sha = state_dict_sha256(initial_adapter_state)
    architecture = adapter.architecture()
    control_record.update(
        {
            "base_surface_state_dict_sha256": base_state_sha,
            "response_adapter_state_dict_sha256": initial_adapter_sha,
            "combined_state_sha256": combined_state_sha256(
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
        adapter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, EPOCHS + 1):
        adapter.train()
        term_values = {
            "total": [],
            "uncertainty_pairwise": [],
            "independent_response": [],
            "surface_descriptor": [],
        }
        batches = complete_scene_batches(
            train_data["scene_ids"],
            row_count=len(train_data["radio_features"]),
            target_batch_rows=TARGET_BATCH_ROWS,
            generator=generator,
        )
        for rows in batches:
            _target_token, teacher, all_descriptors, teacher_mask = _targets(
                train_data, rows
            )
            adapted = adapter(base_train[rows.to(base_train.device)])
            student = F.normalize(
                head(adapted[:, None])[:, 0].float(), dim=-1, eps=1e-8
            )
            teacher = teacher.to(device)
            all_descriptors = all_descriptors.to(device)
            teacher_mask = teacher_mask.to(device)
            pair_loss, _pair_stats = (
                compute_scene_wise_uncertainty_weighted_text_response_pairwise_gap_smooth_l1_loss(
                    student,
                    teacher,
                    all_descriptors,
                    teacher_mask,
                    text_bank,
                    [train_data["scene_ids"][row] for row in rows.tolist()],
                    standard_error_multiplier=STANDARD_ERROR_MULTIPLIER,
                    tie_tolerance=TIE_TOLERANCE,
                    eps=EPS,
                )
            )
            independent_loss = (
                compute_independent_normalized_cosine_response_smooth_l1_loss(
                    student, teacher, text_bank
                )
            )
            all_view_cosine = torch.einsum(
                "bd,bvd->bv", student, all_descriptors
            )
            surface_descriptor_loss = (1.0 - all_view_cosine)[
                teacher_mask
            ].mean()
            total = (
                PAIRWISE_WEIGHT * pair_loss
                + INDEPENDENT_RESPONSE_WEIGHT * independent_loss
                + SURFACE_DESCRIPTOR_WEIGHT * surface_descriptor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for name, value in (
                ("total", total),
                ("uncertainty_pairwise", pair_loss),
                ("independent_response", independent_loss),
                ("surface_descriptor", surface_descriptor_loss),
            ):
                term_values[name].append(float(value.detach().cpu()))

        adapter.eval()
        metrics, units, valid = _evaluate(
            adapter, head, base_validation, validation_data, text_bank
        )
        selector, _delta = continuous_selector_metrics(
            units,
            valid,
            control_units,
            control_valid,
            validation_scene_names,
        )
        state = _clone_state(adapter)
        adapter_sha = state_dict_sha256(state)
        record = annotate_selection_record(
            {
                "epoch": epoch,
                "training_losses": {
                    name: sum(values) / len(values)
                    for name, values in term_values.items()
                },
                **metrics,
            },
            control_record=history[0],
            selector=selector,
        )
        record.update(
            {
                "base_surface_state_dict_sha256": base_state_sha,
                "response_adapter_state_dict_sha256": adapter_sha,
                "combined_state_sha256": combined_state_sha256(
                    base_state_sha, adapter_sha, architecture["digest"]
                ),
                "scene_query_unit_loss_sha256": tensor_sha256(units.float()),
                "scene_query_unit_valid_sha256": tensor_sha256(valid),
            }
        )
        candidate_history = [*history, record]
        selected_epoch = select_best_epoch(candidate_history)
        best_updated = selected_epoch == epoch
        record["selection_updated_best"] = best_updated
        if best_updated:
            best_epoch = epoch
            best_state = state
            stale = 0
        else:
            if selected_epoch != best_epoch:
                raise RuntimeError("pilot online best epoch changed retroactively")
            stale += 1
        record["patience_stale"] = stale
        record["patience_stop"] = stale >= PATIENCE
        history.append(record)
        selector_units.append(
            {"epoch": epoch, "loss": units.float(), "valid": valid}
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= PATIENCE:
            break

    finalized_best = select_best_epoch(history)
    if finalized_best != best_epoch:
        raise RuntimeError("pilot online/final selection differs")
    adapter.load_state_dict(best_state, strict=True)
    final_metrics, final_units, final_valid = _evaluate(
        adapter, head, base_validation, validation_data, text_bank
    )
    if (
        tensor_sha256(final_units.float())
        != history[best_epoch]["scene_query_unit_loss_sha256"]
        or tensor_sha256(final_valid)
        != history[best_epoch]["scene_query_unit_valid_sha256"]
    ):
        raise RuntimeError("selected adapter replay differs")
    best_adapter_sha = state_dict_sha256(best_state)
    best_combined_sha = combined_state_sha256(
        base_state_sha, best_adapter_sha, architecture["digest"]
    )
    selected_selector = history[best_epoch]["continuous_selector"]
    pilot_advance = (
        best_epoch > 0
        and history[best_epoch]["surface_control_feasible"] is True
        and float(selected_selector["normalized_mean_delta"])
        <= -PILOT_REQUIRED_MEAN_IMPROVEMENT
        and float(selected_selector["normalized_upper_cvar10_delta"])
        <= PILOT_GLOBAL_CVAR_TOLERANCE
        and float(selected_selector["worst_scene_upper_cvar10_delta"])
        <= PILOT_PER_SCENE_CVAR_TOLERANCE
        and float(history[best_epoch]["adapter_angle"]["max_degrees"])
        <= ADAPTER_MAX_ANGLE_DEGREES + ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES
    )
    implementation_paths = (
        Path(__file__),
        repo_root / "radio_gs/models/surface_text_response_adapter.py",
        repo_root / "radio_gs/losses/direct_point_query_logit_distill_loss.py",
        repo_root / "radio_gs/scripts/train_surface_region_summary_readout.py",
        repo_root / "radio_gs/scripts/train_surface_region_text_response_distill.py",
        repo_root / "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
    )
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
        "pilot_advance_gate_passed": pilot_advance,
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
                _file_record(path) for path in implementation_paths
            ],
        },
        "training_contract": _training_contract(),
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
        "pilot_advance_gate": {
            "required_mean_improvement": PILOT_REQUIRED_MEAN_IMPROVEMENT,
            "global_cvar_tolerance": PILOT_GLOBAL_CVAR_TOLERANCE,
            "per_scene_cvar_tolerance": PILOT_PER_SCENE_CVAR_TOLERANCE,
            "adapter_max_angle_degrees": ADAPTER_MAX_ANGLE_DEGREES,
            "adapter_angle_audit_absolute_tolerance_degrees": (
                ANGLE_AUDIT_ABSOLUTE_TOLERANCE_DEGREES
            ),
        },
        "uncertainty_weight_statistics": uncertainty_statistics,
        "history_length": len(history),
        "training_contract": _training_contract(),
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
