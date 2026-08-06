#!/usr/bin/env python3
"""Train a global source-only AcceptedV2 typed-context residual (Stage C).

The program pairs each immutable full-scalar sparse-teacher shard with one
adaptive typed-context authority.  Scenes are loaded and released one at a
time: a batch may densify at most four teacher views, while a scene, split,
or cohort teacher tensor is never converted to ``rows x views x 1536``.
Normalization, the OOD envelope, and the boundary-balance threshold are fit
only from the frozen source-train split.  Source validation can select only a
globally non-regressing checkpoint; epoch zero is always the exact AcceptedV2
fallback candidate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    VIEW_CAP_PER_REGION,
)
from radio_gs.interfaces.surface_region_summary import (
    surface_region_state_dict_sha256,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    validate_adaptive_typed_context_authority,
)
from radio_gs.interfaces.surface_region_typed_context_training import (
    COMBINED_SCALAR_DIM,
    TYPED_CONTEXT_NORMALIZATION_CONTRACT_SHA256,
    accepted_v2_authority,
    apply_typed_context_normalization,
    build_typed_context_normalization_authority,
    build_typed_context_training_certificate,
    typed_context_training_source_access,
    typed_context_normalization_authority_sha256,
    validate_typed_context_normalization_authority,
    write_typed_context_checkpoint,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


ARTIFACT_TYPE = (
    "surface_region_accepted_v2_adaptive_typed_context_residual_"
    "source_only_training_v1"
)
SEED = 0
EPOCHS = 30
PATIENCE = 8
BATCH_ROWS = 64
EVAL_BATCH_ROWS = 128
RELATION_EVAL_ROWS = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRADIENT_NORM = 1.0
MAX_ANGLE_RADIANS = 0.15
MAX_ALPHA = 0.25
RELATION_WEIGHT = 0.1
HARD_NEGATIVE_RANKING_WEIGHT = 0.25
HARD_NEGATIVE_MARGIN = 0.10
HARD_NEGATIVE_TEACHER_COSINE_CEILING = 0.80
NON_REGRESSION_TOLERANCE = 1e-7
RESUME_STATE_SCHEMA = "radio_gs.surface_region_typed_context_resume_state.v1"


@dataclass(frozen=True)
class SceneBinding:
    split: str
    scene_id: str
    training_shard: dict[str, str]
    adaptive_context: dict[str, str]


def training_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "seed": SEED,
        "input": {
            "full_scalar_sparse_teacher_shard_contract_sha256": (
                base_trainer.TRAINING_SHARD_CONTRACT_SHA256
            ),
            "adaptive_typed_context_authority": "v2_typed_budget_complete",
            "pairing": (
                "exact_scene_region_row_id_and_canonical_region_index_hash"
            ),
            "accepted_v2": accepted_v2_authority(),
        },
        "cohort": {
            "source_train_scenes": base_trainer.TRAIN_SCENE_COUNT,
            "source_validation_scenes": base_trainer.VALIDATION_SCENE_COUNT,
            "scene_and_physical_space_disjoint": True,
            "frozen_external_authority": True,
            "per_scene_hyperparameters": False,
        },
        "streaming": {
            "unit": "one_scene_shard_plus_one_adaptive_context_authority",
            "retained_cross_scene_teacher_descriptors": False,
            "batch_local_teacher_shape": ["batch_rows", "at_most_4", 1536],
            "global_or_split_teacher_densification": False,
        },
        "normalization": {
            "contract_sha256": TYPED_CONTEXT_NORMALIZATION_CONTRACT_SHA256,
            "fit_split": "source_train_only",
            "fit_mask": (
                "accepted_exact_overlap_and_adaptive_typed_context_valid"
            ),
            "validation_contribution": False,
            "ood_action": "bitwise_immutable_accepted_v2_e0",
        },
        "active": (
            "accepted_exact_overlap_and_adaptive_typed_context_valid_and_not_ood"
        ),
        "model": {
            "class": "SurfaceRegionAcceptedV2TypedContextResidualV1",
            "zero_final_projection": True,
            "max_angle_radians": MAX_ANGLE_RADIANS,
            "max_alpha": MAX_ALPHA,
            "immutable_base": True,
            "immutable_teacher": True,
        },
        "objective": {
            "unit_direction_all_view_cosine_weight": 1.0,
            "teacher_relation_gram_smooth_l1_weight": RELATION_WEIGHT,
            "boundary_balanced_hard_negative_ranking_weight": (
                HARD_NEGATIVE_RANKING_WEIGHT
            ),
            "hard_negative_margin": HARD_NEGATIVE_MARGIN,
            "hard_negative_teacher_cosine_ceiling": (
                HARD_NEGATIVE_TEACHER_COSINE_CEILING
            ),
            "boundary_threshold": "source_train_fit_only_lower_median",
            "scene_weighting": "equal_scene_gradient_accumulation",
            "per_scene_weights": False,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_rows": BATCH_ROWS,
            "max_gradient_norm": MAX_GRADIENT_NORM,
        },
        "selection": {
            "split": "source_validation",
            "gradients": False,
            "aggregation": "scene_macro",
            "epoch_zero_exact_identity_candidate": True,
            "automatic_nonimprovement_fallback": "epoch_zero",
            "non_regression_tolerance": NON_REGRESSION_TOLERANCE,
            "required": [
                "macro_mean_all_view_cosine_not_below_base",
                "macro_p05_row_cosine_not_below_base",
                "macro_relation_fidelity_not_below_base",
                "paired_scene_worst_delta_not_below_base",
                "every_scene_has_two_active_rows",
                "inactive_and_ood_bitwise_base",
            ],
        },
        "source_access": typed_context_training_source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


def _state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _state_sha(model: torch.nn.Module) -> str:
    return surface_region_state_dict_sha256(_state_copy(model))


def _load_context(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    value, observed, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="adaptive typed-context authority",
    )
    return validate_adaptive_typed_context_authority(value), {
        "path": str(source),
        "sha256": observed,
    }


def _validate_scene_pair(
    shard: Mapping[str, Any], context: Mapping[str, Any]
) -> str:
    scenes = sorted(set(str(item) for item in shard["scene_ids"]))
    if len(scenes) != 1 or context["scene_id"] != scenes[0]:
        raise ValueError("typed-context shard/context scene differs")
    if context["region_row_ids"] != shard["region_row_ids"]:
        raise ValueError("typed-context shard/context stable region rows differ")
    canonical_sha = base_trainer._tensor_channel_sha256(
        context["canonical_region_indices"]
    )
    if canonical_sha != shard["sampling_audit"]["canonical_region_indices_sha256"]:
        raise ValueError("typed-context shard/context canonical rows differ")
    if int(context["typed_context_valid"].numel()) != len(shard["region_row_ids"]):
        raise ValueError("typed-context shard/context row count differs")
    if not bool(context["selection_complete"].all()):
        raise ValueError("adaptive typed-context selection is incomplete")
    return scenes[0]


def load_scene(binding: SceneBinding) -> dict[str, Any]:
    shard, shard_record = base_trainer.load_training_shard(
        binding.training_shard["path"],
        expected_sha256=binding.training_shard["sha256"],
        expected_split=binding.split,
    )
    context, context_record = _load_context(
        binding.adaptive_context["path"],
        expected_sha256=binding.adaptive_context["sha256"],
    )
    scene = _validate_scene_pair(shard, context)
    if (
        scene != binding.scene_id
        or shard_record != binding.training_shard
        or context_record != binding.adaptive_context
    ):
        raise ValueError("typed-context scene binding changed")
    return {
        **shard,
        "pooled_context_radio_direction": context[
            "pooled_context_radio_direction"
        ].float(),
        "typed_context_statistics": context["typed_context_statistics"].float(),
        "typed_context_valid": context["typed_context_valid"].bool(),
        "context_token_count": context["context_token_count"].long(),
        "context_input_authority": dict(context["input_authority"]),
    }


def _teacher_view_ids_by_row(shard: Mapping[str, Any]) -> list[list[str]]:
    pair_rows = shard["official_multiview_siglip2_teacher_pair_region_indices"]
    result = [[] for _ in shard["region_row_ids"]]
    for row, view_id in zip(pair_rows.tolist(), shard["teacher_pair_view_ids"]):
        result[int(row)].append(str(view_id))
    return result


def _lightweight_scene_metadata(shard: Mapping[str, Any]) -> dict[str, Any]:
    """Retain cohort IDs/counts only; never retain a teacher descriptor."""

    row_views = _teacher_view_ids_by_row(shard)
    counts = torch.tensor([len(values) for values in row_views], dtype=torch.long)
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    return {
        "scene_ids": list(shard["scene_ids"]),
        "region_row_ids": list(shard["region_row_ids"]),
        "teacher_pair_view_ids": [view for values in row_views for view in values],
        "official_multiview_siglip2_teacher_pair_row_offsets": offsets,
        "eligible": shard["eligible"].detach().cpu().contiguous().clone(),
        "lineage": dict(shard["lineage"]),
    }


def _metadata_for_cohort(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scene_ids: list[str] = []
    region_ids: list[str] = []
    view_ids: list[str] = []
    counts: list[int] = []
    eligible: list[torch.Tensor] = []
    lineages: list[dict[str, Any]] = []
    for shard in shards:
        scene_ids.extend(shard["scene_ids"])
        region_ids.extend(shard["region_row_ids"])
        offsets = shard["official_multiview_siglip2_teacher_pair_row_offsets"]
        view_ids.extend(shard["teacher_pair_view_ids"])
        counts.extend((offsets[1:] - offsets[:-1]).tolist())
        eligible.append(shard["eligible"])
        lineages.append(dict(shard["lineage"]))
    offsets = torch.zeros(len(counts) + 1, dtype=torch.long)
    if counts:
        offsets[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    return {
        "scene_ids": scene_ids,
        "region_row_ids": region_ids,
        "teacher_pair_view_ids": view_ids,
        "official_multiview_siglip2_teacher_pair_row_offsets": offsets,
        "eligible": torch.cat(eligible),
        "lineages": lineages,
    }


def preflight_bindings(
    *,
    train_shards: Sequence[str],
    train_shas: Sequence[str],
    train_contexts: Sequence[str],
    train_context_shas: Sequence[str],
    validation_shards: Sequence[str],
    validation_shas: Sequence[str],
    validation_contexts: Sequence[str],
    validation_context_shas: Sequence[str],
    cohort_and_manifests: tuple[Any, ...],
) -> tuple[list[SceneBinding], list[SceneBinding], dict[str, Any]]:
    def prepare(
        split: str,
        paths: Sequence[str],
        shas: Sequence[str],
        contexts: Sequence[str],
        context_shas: Sequence[str],
        expected_count: int,
    ) -> tuple[list[SceneBinding], list[dict[str, Any]]]:
        if not (
            len(paths) == len(shas) == len(contexts) == len(context_shas)
            == expected_count
        ):
            raise ValueError(f"{split} requires exactly {expected_count} paired scenes")
        bindings: list[SceneBinding] = []
        metadata: list[dict[str, Any]] = []
        for path, sha, context_path, context_sha in zip(
            paths, shas, contexts, context_shas
        ):
            shard, shard_record = base_trainer.load_training_shard(
                path, expected_sha256=sha, expected_split=split
            )
            context, context_record = _load_context(
                context_path, expected_sha256=context_sha
            )
            scene = _validate_scene_pair(shard, context)
            bindings.append(
                SceneBinding(split, scene, shard_record, context_record)
            )
            metadata.append(_lightweight_scene_metadata(shard))
            del shard, context
        bindings.sort(key=lambda item: item.scene_id)
        metadata.sort(key=lambda item: str(item["scene_ids"][0]))
        scenes = [item.scene_id for item in bindings]
        if len(set(scenes)) != expected_count:
            raise ValueError(f"{split} repeats a scene")
        return bindings, metadata

    train_bindings, train_meta = prepare(
        "source_train",
        train_shards,
        train_shas,
        train_contexts,
        train_context_shas,
        base_trainer.TRAIN_SCENE_COUNT,
    )
    validation_bindings, validation_meta = prepare(
        "source_validation",
        validation_shards,
        validation_shas,
        validation_contexts,
        validation_context_shas,
        base_trainer.VALIDATION_SCENE_COUNT,
    )
    (
        cohort_authority,
        cohort_file,
        source_manifest,
        source_file,
        teacher_manifest,
        teacher_file,
        exclusion_manifest,
        exclusion_file,
    ) = cohort_and_manifests
    cohort = base_trainer.validate_training_cohort(
        _metadata_for_cohort(train_meta),
        _metadata_for_cohort(validation_meta),
        cohort_authority,
        cohort_file,
        source_manifest,
        source_file,
        teacher_manifest,
        teacher_file,
        exclusion_manifest,
        exclusion_file,
    )
    return train_bindings, validation_bindings, cohort


def input_records(bindings: Sequence[SceneBinding]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": item.scene_id,
            "training_shard": dict(item.training_shard),
            "adaptive_context": dict(item.adaptive_context),
        }
        for item in bindings
    ]


def fit_normalization(
    bindings: Sequence[SceneBinding],
    *,
    source_state_cohort_authority_sha256: str,
) -> dict[str, Any]:
    summaries: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for binding in bindings:
        scene = load_scene(binding)
        summaries.append(
            torch.cat(
                (
                    scene["raw_full_scalar_summary"],
                    scene["typed_context_statistics"],
                ),
                dim=1,
            )
        )
        masks.append(scene["eligible"] & scene["typed_context_valid"])
        del scene
    authority = build_typed_context_normalization_authority(
        torch.cat(summaries),
        torch.cat(masks),
        source_state_cohort_authority_sha256=(
            source_state_cohort_authority_sha256
        ),
        train_input_records=input_records(bindings),
    )
    return validate_typed_context_normalization_authority(authority)


def _combined_scalars(scene: Mapping[str, Any]) -> torch.Tensor:
    values = torch.cat(
        (scene["raw_full_scalar_summary"], scene["typed_context_statistics"]),
        dim=-1,
    )
    if values.shape[1] != COMBINED_SCALAR_DIM:
        raise ValueError("typed-context combined scalar dimension differs")
    return values


def _routing(
    scene: Mapping[str, Any], normalization: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = apply_typed_context_normalization(
        _combined_scalars(scene), normalization
    )
    eligible = scene["eligible"].bool()
    context_valid = scene["typed_context_valid"].bool()
    # The model's declared mask is exactly the authority that owns a nonzero
    # context carrier.  Ineligible rows are made OOD, so its effective active
    # mask is the required overlap & valid & ~source-train-envelope predicate.
    declared = context_valid
    effective_ood = normalized.ood_mask | ~eligible
    active = eligible & context_valid & ~normalized.ood_mask
    return declared, effective_ood, active


def gather_sparse_teacher_batch(
    scene: Mapping[str, Any], rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Densify only requested rows and at most the frozen four views."""

    requested = torch.as_tensor(rows).detach().long().cpu().reshape(-1)
    if requested.numel() == 0 or bool((requested < 0).any()) or bool(
        (requested >= len(scene["region_row_ids"])).any()
    ):
        raise ValueError("typed-context sparse teacher batch rows differ")
    pair_rows = scene["official_multiview_siglip2_teacher_pair_region_indices"]
    pair_descriptors = scene[
        "official_multiview_siglip2_teacher_pair_descriptors"
    ]
    counts = torch.bincount(pair_rows, minlength=len(scene["region_row_ids"]))
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    maximum = int(counts[requested].max())
    if not 1 <= maximum <= VIEW_CAP_PER_REGION:
        raise ValueError("typed-context sparse teacher view count differs")
    result = torch.zeros(requested.numel(), maximum, pair_descriptors.shape[1])
    mask = torch.zeros(requested.numel(), maximum, dtype=torch.bool)
    for output_row, source_row in enumerate(requested.tolist()):
        start, stop = int(offsets[source_row]), int(offsets[source_row + 1])
        count = stop - start
        result[output_row, :count] = pair_descriptors[start:stop]
        mask[output_row, :count] = True
    return result, mask


def teacher_prototype(teachers: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=1, keepdim=True)
    if bool((count <= 0).any()):
        raise ValueError("typed-context teacher prototype lacks a view")
    mean = (teachers * mask[..., None]).sum(dim=1) / count
    return F.normalize(mean.float(), dim=-1)


def boundary_score(statistics: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(statistics).float()
    resultant = values[..., 7].clamp(0.0, 1.0)
    anchor_cosine = values[..., 10].clamp(-1.0, 1.0)
    return 1.0 - 0.5 * (resultant + 0.5 * (anchor_cosine + 1.0))


def typed_context_objective(
    semantic: torch.Tensor,
    teachers: torch.Tensor,
    teacher_mask: torch.Tensor,
    statistics: torch.Tensor,
    *,
    boundary_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Fixed global cosine + relation + boundary-balanced hard-negative loss."""

    student = F.normalize(semantic.float(), dim=-1)
    teacher_views = F.normalize(teachers.float(), dim=-1)
    pair_cosine = torch.einsum("bd,bvd->bv", student, teacher_views)
    row_cosine_loss = ((1.0 - pair_cosine) * teacher_mask).sum(dim=1) / (
        teacher_mask.sum(dim=1)
    )
    cosine_loss = row_cosine_loss.mean()
    prototype = teacher_prototype(teachers, teacher_mask)
    if semantic.shape[0] >= 2:
        upper = torch.triu(
            torch.ones(
                semantic.shape[0], semantic.shape[0], dtype=torch.bool,
                device=semantic.device,
            ),
            diagonal=1,
        )
        student_gram = student @ student.T
        teacher_gram = prototype @ prototype.T
        relation_loss = F.smooth_l1_loss(
            student_gram[upper], teacher_gram[upper]
        )

        response = student @ prototype.T
        candidate = (teacher_gram <= HARD_NEGATIVE_TEACHER_COSINE_CEILING) & ~torch.eye(
            semantic.shape[0], dtype=torch.bool, device=semantic.device
        )
        valid_negative = candidate.any(dim=1)
        masked_response = response.masked_fill(~candidate, -torch.inf)
        hardest = masked_response.amax(dim=1)
        positive = response.diagonal()
        ranking_rows = F.relu(HARD_NEGATIVE_MARGIN - positive + hardest)
        score = boundary_score(statistics).to(semantic.device)
        high_boundary = score > float(boundary_threshold)
        strata: list[torch.Tensor] = []
        for stratum in (high_boundary, ~high_boundary):
            selected = stratum & valid_negative
            if bool(selected.any()):
                strata.append(ranking_rows[selected].mean())
        ranking_loss = (
            torch.stack(strata).mean()
            if strata
            else semantic.sum() * 0.0
        )
        negative_rows = int(valid_negative.sum())
    else:
        relation_loss = semantic.sum() * 0.0
        ranking_loss = semantic.sum() * 0.0
        negative_rows = 0
    objective = (
        cosine_loss
        + RELATION_WEIGHT * relation_loss
        + HARD_NEGATIVE_RANKING_WEIGHT * ranking_loss
    )
    return objective, {
        "all_view_cosine_loss": cosine_loss,
        "relation_gram_smooth_l1_loss": relation_loss,
        "boundary_balanced_hard_negative_ranking_loss": ranking_loss,
        "hard_negative_rows": negative_rows,
    }


def _training_rows(active: torch.Tensor, *, epoch: int, scene_index: int) -> torch.Tensor:
    rows = torch.where(active)[0]
    if rows.numel() < 2:
        raise ValueError("every source-train scene requires two active rows")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 1_000_003 * int(epoch) + 97 * int(scene_index))
    return rows[torch.randperm(rows.numel(), generator=generator)][
        : min(BATCH_ROWS, rows.numel())
    ]


def train_one_epoch(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    optimizer: torch.optim.Optimizer,
    bindings: Sequence[SceneBinding],
    normalization: Mapping[str, Any],
    device: torch.device,
    *,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "objective": 0.0,
        "all_view_cosine_loss": 0.0,
        "relation_gram_smooth_l1_loss": 0.0,
        "boundary_balanced_hard_negative_ranking_loss": 0.0,
    }
    sampled_rows = 0
    available_rows = 0
    ood_rows = 0
    for scene_index, binding in enumerate(bindings):
        scene = load_scene(binding)
        declared, effective_ood, active = _routing(scene, normalization)
        rows = _training_rows(active, epoch=epoch, scene_index=scene_index)
        teachers_cpu, teacher_mask_cpu = gather_sparse_teacher_batch(scene, rows)
        semantic = model(
            scene["accepted_v2_e0"][rows].to(device),
            scene["pooled_context_radio_direction"][rows].to(device),
            scene["raw_full_scalar_summary"][rows].to(device),
            scene["typed_context_statistics"][rows].to(device),
            active_mask=declared[rows].to(device),
            ood_mask=effective_ood[rows].to(device),
        )
        objective, components = typed_context_objective(
            semantic,
            teachers_cpu.to(device),
            teacher_mask_cpu.to(device),
            scene["typed_context_statistics"][rows].to(device),
            boundary_threshold=float(normalization["source_boundary_score_median"]),
        )
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("typed-context objective is non-finite")
        (objective / len(bindings)).backward()
        totals["objective"] += float(objective.detach())
        for name in totals.keys() - {"objective"}:
            totals[name] += float(components[name].detach())
        sampled_rows += int(rows.numel())
        available_rows += int(active.sum())
        ood_rows += int((scene["eligible"] & scene["typed_context_valid"] & effective_ood).sum())
        del scene, teachers_cpu, teacher_mask_cpu, semantic, objective
    torch.nn.utils.clip_grad_norm_(
        tuple(model.parameters()), MAX_GRADIENT_NORM, error_if_nonfinite=True
    )
    optimizer.step()
    return {
        **{name: value / len(bindings) for name, value in totals.items()},
        "scene_count": len(bindings),
        "equal_scene_weight": 1.0 / len(bindings),
        "sampled_trainable_rows": sampled_rows,
        "available_trainable_rows": available_rows,
        "ood_rows_excluded": ood_rows,
        "global_or_split_teacher_densification": False,
    }


def _lower_quantile(values: torch.Tensor, fraction: float) -> float:
    ordered = torch.sort(values.float().cpu()).values
    index = int(math.floor(float(fraction) * max(0, ordered.numel() - 1)))
    return float(ordered[index])


def _relation_fidelity(
    descriptors: torch.Tensor, prototypes: torch.Tensor
) -> float:
    if descriptors.shape[0] < 2:
        return 1.0
    upper = torch.triu(
        torch.ones(descriptors.shape[0], descriptors.shape[0], dtype=torch.bool),
        diagonal=1,
    )
    student = F.normalize(descriptors.float(), dim=-1)
    teacher = F.normalize(prototypes.float(), dim=-1)
    mae = ((student @ student.T)[upper] - (teacher @ teacher.T)[upper]).abs().mean()
    return float(1.0 - mae)


@torch.no_grad()
def evaluate_scene(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    scene: Mapping[str, Any],
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    declared, effective_ood, active = _routing(scene, normalization)
    if int(active.sum()) < 2:
        raise ValueError("every validation scene requires two active rows")
    all_base: list[torch.Tensor] = []
    all_candidate: list[torch.Tensor] = []
    all_prototype: list[torch.Tensor] = []
    all_base_cosine: list[torch.Tensor] = []
    all_candidate_cosine: list[torch.Tensor] = []
    fallback_equal = True
    for start in range(0, len(scene["region_row_ids"]), EVAL_BATCH_ROWS):
        rows = torch.arange(start, min(start + EVAL_BATCH_ROWS, len(scene["region_row_ids"])))
        base = scene["accepted_v2_e0"][rows].to(device)
        result = model.forward_with_diagnostics(
            base,
            scene["pooled_context_radio_direction"][rows].to(device),
            scene["raw_full_scalar_summary"][rows].to(device),
            scene["typed_context_statistics"][rows].to(device),
            active_mask=declared[rows].to(device),
            ood_mask=effective_ood[rows].to(device),
        )
        fallback = ~active[rows].to(device)
        fallback_equal = fallback_equal and torch.equal(
            result.semantic_descriptor[fallback], base[fallback]
        )
        selected = active[rows]
        if bool(selected.any()):
            active_rows = rows[selected]
            teachers, teacher_mask = gather_sparse_teacher_batch(scene, active_rows)
            prototype = teacher_prototype(teachers, teacher_mask)
            base_active = base[selected.to(device)].cpu()
            candidate_active = result.semantic_descriptor[selected.to(device)].cpu()
            view_cos_base = torch.einsum(
                "bd,bvd->bv", F.normalize(base_active, dim=-1), teachers
            )
            view_cos_candidate = torch.einsum(
                "bd,bvd->bv", F.normalize(candidate_active, dim=-1), teachers
            )
            all_base_cosine.append(
                (view_cos_base * teacher_mask).sum(1) / teacher_mask.sum(1)
            )
            all_candidate_cosine.append(
                (view_cos_candidate * teacher_mask).sum(1) / teacher_mask.sum(1)
            )
            all_base.append(base_active)
            all_candidate.append(candidate_active)
            all_prototype.append(prototype)
    base_rows = torch.cat(all_base)
    candidate_rows = torch.cat(all_candidate)
    prototypes = torch.cat(all_prototype)
    base_cosine = torch.cat(all_base_cosine)
    candidate_cosine = torch.cat(all_candidate_cosine)
    if base_rows.shape[0] > RELATION_EVAL_ROWS:
        selected = torch.linspace(
            0, base_rows.shape[0] - 1, RELATION_EVAL_ROWS
        ).round().long().unique()
        base_relation = base_rows[selected]
        candidate_relation = candidate_rows[selected]
        prototype_relation = prototypes[selected]
    else:
        base_relation = base_rows
        candidate_relation = candidate_rows
        prototype_relation = prototypes
    base_metrics = {
        "mean_all_view_cosine": float(base_cosine.mean()),
        "p05_row_mean_all_view_cosine": _lower_quantile(base_cosine, 0.05),
        "relation_fidelity": _relation_fidelity(base_relation, prototype_relation),
    }
    candidate_metrics = {
        "mean_all_view_cosine": float(candidate_cosine.mean()),
        "p05_row_mean_all_view_cosine": _lower_quantile(candidate_cosine, 0.05),
        "relation_fidelity": _relation_fidelity(
            candidate_relation, prototype_relation
        ),
    }
    delta = {
        name: candidate_metrics[name] - base_metrics[name]
        for name in base_metrics
    }
    return {
        "base": base_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_base": delta,
        "active_rows": int(active.sum()),
        "ood_fallback_rows": int(
            (scene["eligible"] & scene["typed_context_valid"] & effective_ood).sum()
        ),
        "inactive_fallback_rows": int((~active).sum()),
        "fallback_bitwise_accepted_v2_e0": bool(fallback_equal),
        "relation_evaluation_rows": int(base_relation.shape[0]),
        "validation_no_grad": not torch.is_grad_enabled(),
    }


@torch.no_grad()
def evaluate(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    bindings: Sequence[SceneBinding],
    normalization: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    per_scene: dict[str, Any] = {}
    for binding in bindings:
        scene = load_scene(binding)
        per_scene[binding.scene_id] = evaluate_scene(
            model, scene, normalization, device
        )
        del scene
    names = (
        "mean_all_view_cosine",
        "p05_row_mean_all_view_cosine",
        "relation_fidelity",
    )
    base = {
        name: sum(row["base"][name] for row in per_scene.values()) / len(per_scene)
        for name in names
    }
    candidate = {
        name: sum(row["candidate"][name] for row in per_scene.values()) / len(per_scene)
        for name in names
    }
    delta = {name: candidate[name] - base[name] for name in names}
    scene_mean_deltas = [
        row["candidate_minus_base"]["mean_all_view_cosine"]
        for row in per_scene.values()
    ]
    checks = {
        "macro_mean_all_view_cosine": (
            delta["mean_all_view_cosine"] >= -NON_REGRESSION_TOLERANCE
        ),
        "macro_p05_row_mean_all_view_cosine": (
            delta["p05_row_mean_all_view_cosine"] >= -NON_REGRESSION_TOLERANCE
        ),
        "macro_relation_fidelity": (
            delta["relation_fidelity"] >= -NON_REGRESSION_TOLERANCE
        ),
        "paired_scene_worst_mean_delta": (
            min(scene_mean_deltas) >= -NON_REGRESSION_TOLERANCE
        ),
        "every_scene_two_active_rows": all(
            row["active_rows"] >= 2 for row in per_scene.values()
        ),
        "fallback_bitwise_accepted_v2_e0": all(
            row["fallback_bitwise_accepted_v2_e0"] for row in per_scene.values()
        ),
    }
    return {
        "aggregation": "scene_macro",
        "base": base,
        "candidate": candidate,
        "candidate_minus_base": delta,
        "paired_scene_mean_delta": {
            "minimum": min(scene_mean_deltas),
            "p05": sorted(scene_mean_deltas)[
                int(math.floor(0.05 * max(0, len(scene_mean_deltas) - 1)))
            ],
            "maximum": max(scene_mean_deltas),
        },
        "per_scene": per_scene,
        "non_regression_checks": checks,
        "non_regression_passed": all(checks.values()),
        "validation_no_grad": not torch.is_grad_enabled(),
        "global_or_split_teacher_densification": False,
    }


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    if not history or [int(item.get("epoch", -1)) for item in history] != list(
        range(len(history))
    ):
        raise ValueError("typed-context history must be contiguous from epoch zero")
    eligible = [
        item
        for item in history
        if item.get("validation", {}).get("non_regression_passed") is True
    ]
    if not eligible:
        raise RuntimeError("typed-context has no non-regressing checkpoint")

    def rank(item: Mapping[str, Any]) -> tuple[float, float, float, int]:
        metrics = item["validation"]["candidate"]
        return (
            float(metrics["mean_all_view_cosine"]),
            float(metrics["p05_row_mean_all_view_cosine"]),
            float(metrics["relation_fidelity"]),
            -int(item["epoch"]),
        )

    return int(max(eligible, key=rank)["epoch"])


def _input_records_sha256(
    train_bindings: Sequence[SceneBinding],
    validation_bindings: Sequence[SceneBinding],
) -> str:
    return canonical_json_sha256(
        {
            "source_train": input_records(train_bindings),
            "source_validation": input_records(validation_bindings),
        }
    )


def build_resume_state(
    *,
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    best_epoch: int,
    best_state: Mapping[str, torch.Tensor],
    history: Sequence[Mapping[str, Any]],
    epochs_without_improvement: int,
    normalization_authority_sha256: str,
    input_records_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": RESUME_STATE_SCHEMA,
        "schema_version": 1,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "normalization_authority_sha256": str(normalization_authority_sha256),
        "input_records_sha256": str(input_records_sha256),
        "next_epoch": int(next_epoch),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": [dict(item) for item in history],
        "model_state_dict": _state_copy(model),
        "best_model_state_dict": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in best_state.items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "source_access": typed_context_training_source_access(),
    }
    payload["model_state_dict_sha256"] = surface_region_state_dict_sha256(
        payload["model_state_dict"]
    )
    payload["best_model_state_dict_sha256"] = surface_region_state_dict_sha256(
        payload["best_model_state_dict"]
    )
    return validate_resume_state(
        payload,
        expected_normalization_authority_sha256=normalization_authority_sha256,
        expected_input_records_sha256=input_records_sha256,
    )


def validate_resume_state(
    value: object,
    *,
    expected_normalization_authority_sha256: str,
    expected_input_records_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("typed-context resume state must be a mapping")
    state = dict(value)
    required = {
        "schema", "schema_version", "training_contract_sha256",
        "normalization_authority_sha256", "input_records_sha256", "next_epoch",
        "best_epoch", "epochs_without_improvement", "history",
        "model_state_dict", "model_state_dict_sha256", "best_model_state_dict",
        "best_model_state_dict_sha256", "optimizer_state_dict", "source_access",
    }
    if set(state) != required:
        raise ValueError("typed-context resume state fields differ")
    if (
        state.get("schema") != RESUME_STATE_SCHEMA
        or state.get("schema_version") != 1
        or state.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256
        or state.get("normalization_authority_sha256")
        != str(expected_normalization_authority_sha256)
        or state.get("input_records_sha256") != str(expected_input_records_sha256)
        or state.get("source_access") != typed_context_training_source_access()
    ):
        raise ValueError("typed-context resume authority differs")
    history = state.get("history")
    next_epoch = int(state.get("next_epoch", -1))
    best_epoch = int(state.get("best_epoch", -1))
    if (
        not isinstance(history, list)
        or not history
        or [int(item.get("epoch", -1)) for item in history] != list(range(len(history)))
        or next_epoch != len(history)
        or not 0 <= best_epoch < next_epoch
        or int(state.get("epochs_without_improvement", -1)) < 0
        or not isinstance(state.get("optimizer_state_dict"), Mapping)
    ):
        raise ValueError("typed-context resume progress differs")
    for prefix in ("model", "best_model"):
        values = state.get(f"{prefix}_state_dict")
        if not isinstance(values, Mapping) or surface_region_state_dict_sha256(
            values
        ) != state.get(f"{prefix}_state_dict_sha256"):
            raise ValueError("typed-context resume model hash differs")
    return state


def _load_manifests(args: argparse.Namespace) -> tuple[Any, ...]:
    cohort, cohort_file = base_trainer.load_cohort_authority(
        args.cohort_authority,
        expected_sha256=args.expected_cohort_authority_sha256,
    )
    source, source_file = base_trainer._load_json_manifest(
        args.source_state_manifest,
        expected_sha256=args.expected_source_state_manifest_sha256,
        label="source-state manifest",
        validator=base_trainer.validate_source_state_manifest,
    )
    teacher, teacher_file = base_trainer._load_json_manifest(
        args.teacher_manifest,
        expected_sha256=args.expected_teacher_manifest_sha256,
        label="teacher manifest",
        validator=base_trainer.validate_teacher_manifest,
    )
    exclusion, exclusion_file = base_trainer._load_json_manifest(
        args.benchmark_exclusion_manifest,
        expected_sha256=args.expected_benchmark_exclusion_manifest_sha256,
        label="benchmark exclusion manifest",
        validator=base_trainer.validate_benchmark_exclusion_manifest,
    )
    return (
        cohort, cohort_file, source, source_file, teacher, teacher_file,
        exclusion, exclusion_file,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    normalization_output = output.with_suffix(output.suffix + ".normalization.pt")
    certificate_output = output.with_suffix(output.suffix + ".certificate.json")
    report_output = output.with_suffix(output.suffix + ".json")
    existing = [
        path for path in (output, normalization_output, certificate_output, report_output)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "typed-context first-writer outputs already exist: "
            + ", ".join(str(item) for item in existing)
        )
    authorities = _load_manifests(args)
    train_bindings, validation_bindings, cohort = preflight_bindings(
        train_shards=args.train_shard,
        train_shas=args.expected_train_shard_sha256,
        train_contexts=args.train_context,
        train_context_shas=args.expected_train_context_sha256,
        validation_shards=args.validation_shard,
        validation_shas=args.expected_validation_shard_sha256,
        validation_contexts=args.validation_context,
        validation_context_shas=args.expected_validation_context_sha256,
        cohort_and_manifests=authorities,
    )
    normalization = fit_normalization(
        train_bindings,
        source_state_cohort_authority_sha256=(
            cohort["source_state_cohort_authority_sha256"]
        ),
    )
    # Resume binds the validated content authority; promotion later binds the
    # actual first-writer file SHA-256.
    normalization_content_sha = typed_context_normalization_authority_sha256(
        normalization
    )

    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = SurfaceRegionAcceptedV2TypedContextResidualV1(
        scalar_median=normalization["median"],
        scalar_robust_scale=normalization["robust_scale"],
        max_angle_radians=MAX_ANGLE_RADIANS,
        max_alpha=MAX_ALPHA,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    records_sha = _input_records_sha256(train_bindings, validation_bindings)
    history: list[dict[str, Any]]
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    epochs_without_improvement: int
    start_epoch: int
    if args.resume_state:
        raw, _, _ = load_torch_mapping(
            args.resume_state,
            expected_sha256=args.expected_resume_state_sha256,
            map_location="cpu",
            label="typed-context resume state",
        )
        resume = validate_resume_state(
            raw,
            expected_normalization_authority_sha256=normalization_content_sha,
            expected_input_records_sha256=records_sha,
        )
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = list(resume["history"])
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in resume["best_model_state_dict"].items()
        }
        best_epoch = int(resume["best_epoch"])
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        start_epoch = int(resume["next_epoch"])
    else:
        epoch_zero = evaluate(model, validation_bindings, normalization, device)
        history = [{
            "epoch": 0,
            "training": None,
            "validation": epoch_zero,
            "model_state_dict_sha256": _state_sha(model),
        }]
        if not epoch_zero["non_regression_passed"]:
            raise RuntimeError("zero-step typed-context identity did not pass")
        best_state = _state_copy(model)
        best_epoch = 0
        epochs_without_improvement = 0
        start_epoch = 1
    for epoch in range(start_epoch, EPOCHS + 1):
        training_metrics = train_one_epoch(
            model, optimizer, train_bindings, normalization, device, epoch=epoch
        )
        validation_metrics = evaluate(
            model, validation_bindings, normalization, device
        )
        record = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
            "model_state_dict_sha256": _state_sha(model),
        }
        history.append(record)
        selected = select_best_epoch(history)
        if selected == epoch:
            best_epoch = epoch
            best_state = _state_copy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.resume_state_output_prefix:
            resume = build_resume_state(
                model=model,
                optimizer=optimizer,
                next_epoch=epoch + 1,
                best_epoch=best_epoch,
                best_state=best_state,
                history=history,
                epochs_without_improvement=epochs_without_improvement,
                normalization_authority_sha256=normalization_content_sha,
                input_records_sha256=records_sha,
            )
            write_torch_noclobber(
                f"{args.resume_state_output_prefix}.epoch-{epoch:04d}.pt", resume
            )
        print(json.dumps(record, sort_keys=True), flush=True)
        if epochs_without_improvement >= PATIENCE:
            break
    selected_epoch = select_best_epoch(history)
    if selected_epoch != best_epoch:
        raise RuntimeError("typed-context selected/best epoch state differs")
    model.load_state_dict(best_state, strict=True)
    selected_validation = evaluate(
        model, validation_bindings, normalization, device
    )
    if selected_validation != history[selected_epoch]["validation"]:
        raise RuntimeError("typed-context restored checkpoint metrics differ")
    model.cpu()

    normalization_path = write_torch_noclobber(
        normalization_output, normalization
    )
    normalization_sha = sha256_file(normalization_path)
    cohort_authority, cohort_file, _source, source_file, _teacher, teacher_file, _exclusion, exclusion_file = authorities
    certificate = build_typed_context_training_certificate(
        training_contract=training_contract(),
        model=model,
        normalization_authority_file=file_record(normalization_path),
        cohort_authority={
            "file": cohort_file,
            "authority_sha256": cohort_authority["authority_sha256"],
        },
        external_manifests={
            "source_state": source_file,
            "teacher": teacher_file,
            "benchmark_exclusion": exclusion_file,
        },
        input_records_by_split={
            "source_train": input_records(train_bindings),
            "source_validation": input_records(validation_bindings),
        },
        selected_epoch=selected_epoch,
        selected_validation=selected_validation,
    )
    certificate_path = write_frozen_json(certificate_output, certificate)
    certificate_sha = sha256_file(certificate_path)
    checkpoint_path, checkpoint_sha = write_typed_context_checkpoint(
        output,
        model,
        normalization_authority=normalization,
        normalization_file_sha256=normalization_sha,
        certificate=certificate,
        certificate_file_sha256=certificate_sha,
    )
    report = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "normalization_authority": file_record(normalization_path),
        "training_certificate": file_record(certificate_path),
        "selected_epoch": selected_epoch,
        "automatic_fallback_to_epoch_zero": selected_epoch == 0,
        "selected_validation": selected_validation,
        "history": history,
        "input_records_by_split": {
            "source_train": input_records(train_bindings),
            "source_validation": input_records(validation_bindings),
        },
        "cohort": cohort,
        "source_access": typed_context_training_source_access(),
    }
    write_frozen_json(report_output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", action="append", required=True)
    parser.add_argument("--expected-train-shard-sha256", action="append", required=True)
    parser.add_argument("--train-context", action="append", required=True)
    parser.add_argument("--expected-train-context-sha256", action="append", required=True)
    parser.add_argument("--validation-shard", action="append", required=True)
    parser.add_argument("--expected-validation-shard-sha256", action="append", required=True)
    parser.add_argument("--validation-context", action="append", required=True)
    parser.add_argument("--expected-validation-context-sha256", action="append", required=True)
    parser.add_argument("--cohort-authority", required=True)
    parser.add_argument("--expected-cohort-authority-sha256", required=True)
    parser.add_argument("--source-state-manifest", required=True)
    parser.add_argument("--expected-source-state-manifest-sha256", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--expected-teacher-manifest-sha256", required=True)
    parser.add_argument("--benchmark-exclusion-manifest", required=True)
    parser.add_argument("--expected-benchmark-exclusion-manifest-sha256", required=True)
    parser.add_argument("--resume-state")
    parser.add_argument("--expected-resume-state-sha256")
    parser.add_argument("--resume-state-output-prefix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.resume_state) != bool(args.expected_resume_state_sha256):
        raise ValueError("resume state path and expected SHA-256 must be paired")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
