#!/usr/bin/env python3
"""Source-only direct-boundary alignment (DBA-v1) for factorized-native V2.1.

This is an independent warm-start experiment.  It never mutates the promoted
contrast-V2.1 checkpoint: the exact selected direction-only state is loaded as
step zero, the complete V2.1 visual objective is retained, and one fixed
source-only auxiliary directly supervises the frozen text decision boundary.
No target asset, benchmark query, mask, label, or metric is accepted here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces import (
    factorized_native_contrast_v21_target_descriptor as source_formal,
)
from radio_gs.losses import factorized_native_source_boundary_alignment as boundary
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import (
    calibrate_factorized_native_contrast_v21_global_margin as text_assets,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as legacy,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 as contrast_v2,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v21 as contrast_v21,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as exclusion
from radio_gs.scripts import train_surface_region_typed_context_residual as sparse_teacher
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_direct_boundary_alignment_"
    "execution_authority.v1"
)
CHECKPOINT_SCHEMA = (
    "radio_gs.factorized_native_direct_boundary_alignment_checkpoint.v1"
)
RESULT_SCHEMA = "radio_gs.factorized_native_direct_boundary_alignment_source_result.v1"
SCHEMA_VERSION = 1
TRAIN_SCENES = contrast_v21.TRAIN_SCENES
VALIDATION_SCENES = contrast_v21.VALIDATION_SCENES
OPTIMIZER_STEPS = 64
EVALUATION_INTERVAL = 8
BATCH_ROWS = 64
EVAL_BATCH_ROWS = 128
FIT_QUERY_ROWS = 806
REGIONS_PER_SCENE = 4096
LEARNING_RATE = contrast_v2.LEARNING_RATE
WEIGHT_DECAY = contrast_v2.WEIGHT_DECAY
MAX_GRADIENT_NORM = contrast_v2.MAX_GRADIENT_NORM
DBA_AUXILIARY_WEIGHT = 0.25
BOUNDARY_BCE_IMPROVEMENT = 1e-4
BOUNDARY_BRIER_IMPROVEMENT = 1e-4
RECALL_IMPROVEMENT = 0.02
F1_IMPROVEMENT = 0.01
MINIMUM_PRECISION = 0.25
RANK_NON_REGRESSION_TOLERANCE = 0.005
MAXIMUM_PREDICTED_POSITIVE_FLOOR = 0.02
MAXIMUM_TEACHER_POSITIVE_MULTIPLIER = 8.0
RAW_MEAN_DROP_TOLERANCE = 0.002
RAW_P05_DROP_TOLERANCE = 0.005
CENTERED_DROP_TOLERANCE = 0.01
PAIR_MAE_INCREASE_TOLERANCE = 0.01
PROBE_MAE_INCREASE_TOLERANCE = 0.01
RANK_SAMPLE_CAP = 262_144
SEED = 0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_access() -> dict[str, bool]:
    return {
        "query_independent": True,
        "source_train_opened": True,
        "source_validation_opened_for_promotion_only": True,
        "generic_target_blind_text_bank_opened": True,
        "canonical_generic_negative_bank_opened": True,
        "benchmark_exclusion_manifest_opened": True,
        "target_heldout_opened": False,
        "target_descriptor_opened": False,
        "target_relevance_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "runtime_query_strings_consumed": False,
        "target_metrics_computed": False,
        "scene_identifiers_consumed_by_model": False,
        "per_scene_parameters": False,
        "per_query_parameters": False,
    }


def training_contract() -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "candidate": {
            "initialization": "exact_promoted_contrast_v2_1_direction_only_checkpoint",
            "new_inference_parameters": False,
            "scene_parameters": False,
            "query_parameters": False,
        },
        "cohort": {
            "source_train": list(TRAIN_SCENES),
            "source_validation": list(VALIDATION_SCENES),
            "regions_per_scene": REGIONS_PER_SCENE,
            "scene_and_physical_space_disjoint": True,
            "validation_gradient_contribution": False,
        },
        "text_supervision": {
            "positive": "frozen_target_blind_806_fit_embedding_rows",
            "negative": "frozen_four_canonical_official_siglip2_rows",
            "benchmark_vocabulary": False,
            "runtime_query_strings": False,
            "teacher": (
                "valid_view_mean_of_sigmoid_10_times_"
                "positive_cosine_minus_per_view_max_four_negative_cosines"
            ),
            "student_margin": (
                "positive_cosine_minus_max_four_negative_cosines"
            ),
        },
        "objective": {
            "visual": "complete_immutable_contrast_v2_1_visual_objective",
            "boundary_primary": (
                "half_positive_mean_softplus_negative_10margin_plus_"
                "half_negative_mean_softplus_positive_10margin"
            ),
            "boundary_soft_fidelity": {
                "loss": "class_balanced_smooth_l1_probability_to_teacher_probability",
                "beta": boundary.SMOOTH_L1_BETA,
                "weight_inside_boundary_loss": boundary.SOFT_FIDELITY_WEIGHT,
            },
            "boundary_auxiliary_weight_on_complete_boundary_loss": (
                DBA_AUXILIARY_WEIGHT
            ),
            "equal_scene_gradient_accumulation": True,
        },
        "optimizer": {
            "name": "AdamW",
            "steps": OPTIMIZER_STEPS,
            "batch_rows_per_scene_per_step": BATCH_ROWS,
            "complete_source_epoch": True,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "maximum_gradient_norm": MAX_GRADIENT_NORM,
            "evaluation_steps": [
                0,
                *range(EVALUATION_INTERVAL, OPTIMIZER_STEPS + 1, EVALUATION_INTERVAL),
            ],
            "seed": SEED,
        },
        "promotion": {
            "source_validation_only": True,
            "required_scenes": list(VALIDATION_SCENES),
            "boundary_every_scene": {
                "class_balanced_hard_bce_improvement": BOUNDARY_BCE_IMPROVEMENT,
                "teacher_positive_recall_improvement": RECALL_IMPROVEMENT,
                "f1_improvement": F1_IMPROVEMENT,
                "minimum_precision": MINIMUM_PRECISION,
                "class_balanced_soft_teacher_brier_improvement": (
                    BOUNDARY_BRIER_IMPROVEMENT
                ),
                "sampled_margin_rank_correlation_maximum_drop": (
                    RANK_NON_REGRESSION_TOLERANCE
                ),
                "rank_axis": (
                    "fixed_evenly_spaced_flat_region_query_pairs_cap262144"
                ),
                "maximum_predicted_positive_rate": (
                    "max_0p02_or_8_times_teacher_positive_rate"
                ),
            },
            "visual_every_scene": {
                "maximum_mean_all_view_cosine_drop": RAW_MEAN_DROP_TOLERANCE,
                "maximum_p05_all_view_cosine_drop": RAW_P05_DROP_TOLERANCE,
                "maximum_centered_mean_and_p05_drop": CENTERED_DROP_TOLERANCE,
                "minimum_student_to_teacher_spread_ratio": 0.75,
                "minimum_centered_pair_gram_correlation": 0.20,
                "maximum_pair_gram_mae_increase": PAIR_MAE_INCREASE_TOLERANCE,
                "minimum_absolute_visual_probe_correlation": 0.20,
                "maximum_visual_probe_mae_increase": PROBE_MAE_INCREASE_TOLERANCE,
                "visual_probe_std_ratio_interval": [0.75, 1.25],
            },
            "ranking": [
                "maximum_macro_f1_improvement",
                "maximum_macro_recall_improvement",
                "maximum_macro_balanced_hard_bce_improvement",
                "maximum_macro_centered_residual_cosine",
                "earliest_step",
            ],
            "all_checks_required": True,
        },
        "target_query_or_metric_execution_authorized": False,
        "source_access": source_access(),
    }


TRAINING_CONTRACT_SHA256 = canonical_json_sha256(training_contract())


_DEPENDENCY_PATHS = {
    "boundary_alignment_loss": Path(boundary.__file__).resolve(),
    "contrast_v21_source_validator": Path(source_formal.__file__).resolve(),
    "contrast_v21_trainer": Path(contrast_v21.__file__).resolve(),
    "contrast_v2_visual_objective_trainer": Path(contrast_v2.__file__).resolve(),
    "fit_text_bank_loader": Path(text_assets.__file__).resolve(),
    "canonical_negative_loader": Path(relevance.__file__).resolve(),
    "readout_interface": Path(readout.__file__).resolve(),
    "benchmark_exclusion_validator": Path(exclusion.__file__).resolve(),
}


@dataclass(frozen=True)
class PreparedInputs:
    authority: dict[str, Any]
    source_gate: dict[str, Any]
    source: legacy.PreparedInputs


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError("DBA-v1 output must be canonical absolute")
    return resolved


def validate_execution_authority_header(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "implementation_dependencies",
        "training_contract_sha256",
        "source_contrast_v21_result",
        "fit_text_bank",
        "canonical_negative_bank",
        "benchmark_exclusion_manifest",
        "training_output",
        "training_authorized",
        "target_execution_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("DBA-v1 execution authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != EXECUTION_AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_dba_v1_exact4train_2validation"
        or authority.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256
        or authority.get("training_authorized") is not True
        or authority.get("target_execution_authorized") is not False
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("source_access") != source_access()
    ):
        raise ValueError("DBA-v1 execution authority header differs")
    authority["implementation"] = _record(
        authority["implementation"], label="DBA-v1 implementation"
    )
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        _DEPENDENCY_PATHS
    ):
        raise ValueError("DBA-v1 implementation dependencies differ")
    authority["implementation_dependencies"] = {
        name: _record(dependencies[name], label=f"DBA-v1 dependency {name}")
        for name in sorted(_DEPENDENCY_PATHS)
    }
    for name in (
        "source_contrast_v21_result",
        "fit_text_bank",
        "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    ):
        authority[name] = _record(authority[name], label=f"DBA-v1 {name}")
    authority["training_output"] = _canonical_output(authority["training_output"])
    return authority


def prepare_inputs(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> PreparedInputs:
    raw, digest, source_path = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="DBA-v1 execution authority",
    )
    authority = validate_execution_authority_header(raw)

    # The promoted source chain is validated before generic text files are
    # touched.  This preserves the source/target ordering invariant.
    source_gate = source_formal.validate_source_contrast_v21_result(
        authority["source_contrast_v21_result"]
    )
    source_result = source_gate["result"]
    prepared_v21 = contrast_v21.prepare_inputs(
        source_result["execution_authority"]["path"],
        expected_sha256=source_result["execution_authority"]["sha256"],
    )
    source = prepared_v21.base_v2.source
    if (
        tuple(item.scene_id for item in source.train) != TRAIN_SCENES
        or tuple(item.scene_id for item in source.validation) != VALIDATION_SCENES
    ):
        raise ValueError("DBA-v1 exact4x2 source cohort differs")
    if source.authority["benchmark_exclusion_manifest"] != authority[
        "benchmark_exclusion_manifest"
    ]:
        raise ValueError("DBA-v1 benchmark exclusion lineage differs")

    observed = validate_file_record(
        authority["implementation"], label="DBA-v1 implementation"
    )
    if observed != Path(__file__).resolve():
        raise ValueError("DBA-v1 authority binds another trainer")
    for name, expected in _DEPENDENCY_PATHS.items():
        observed = validate_file_record(
            authority["implementation_dependencies"][name],
            label=f"DBA-v1 dependency {name}",
        )
        if observed != expected:
            raise ValueError(f"DBA-v1 dependency differs: {name}")
    for name in (
        "fit_text_bank",
        "canonical_negative_bank",
        "benchmark_exclusion_manifest",
    ):
        observed = validate_file_record(authority[name], label=f"DBA-v1 {name}")
        if str(observed) != authority[name]["path"]:
            raise ValueError(f"DBA-v1 {name} path differs")
    if expected_output is not None and authority["training_output"] != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("DBA-v1 authority binds another output")
    authority["verified_record"] = {
        "path": str(source_path),
        "sha256": digest,
    }
    return PreparedInputs(authority=authority, source_gate=source_gate, source=source)


def _rank_correlation(student: torch.Tensor, teacher: torch.Tensor) -> float:
    left = torch.as_tensor(student).detach().float().flatten().cpu().contiguous()
    right = torch.as_tensor(teacher).detach().float().flatten().cpu().contiguous()
    if left.shape != right.shape or left.numel() < RANK_SAMPLE_CAP:
        raise ValueError("DBA-v1 rank axes differ")
    axis = torch.div(
        torch.arange(RANK_SAMPLE_CAP, dtype=torch.long) * left.numel(),
        RANK_SAMPLE_CAP,
        rounding_mode="floor",
    )

    def ordinal(value: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(value, stable=True)
        rank = torch.empty(value.numel(), dtype=torch.float64)
        rank[order] = torch.arange(value.numel(), dtype=torch.float64)
        return rank

    x = ordinal(left[axis])
    y = ordinal(right[axis])
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 0:
        raise ValueError("DBA-v1 rank correlation has zero variance")
    return float((x * y).sum() / denominator)


def boundary_metrics(
    student_margin: torch.Tensor, teacher_probability: torch.Tensor
) -> dict[str, float | int]:
    margin = torch.as_tensor(student_margin).detach().float().cpu().contiguous()
    target = torch.as_tensor(teacher_probability).detach().float().cpu().contiguous()
    if (
        margin.shape != target.shape
        or margin.ndim != 2
        or margin.shape[1] != FIT_QUERY_ROWS
        or not bool(torch.isfinite(margin).all())
        or not bool(torch.isfinite(target).all())
        or bool((target < 0).any())
        or bool((target > 1).any())
    ):
        raise ValueError("DBA-v1 boundary metric axes differ")
    teacher_positive = target >= 0.5
    positives = int(teacher_positive.sum())
    negatives = int((~teacher_positive).sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("DBA-v1 validation requires both boundary classes")
    logits = boundary.INFERENCE_LOGIT_SCALE * margin
    probability = torch.sigmoid(logits)
    predicted_positive = logits >= 0
    hard_units = torch.where(
        teacher_positive, F.softplus(-logits), F.softplus(logits)
    )
    brier_units = (probability - target).square()
    hard_bce = 0.5 * hard_units[teacher_positive].mean() + 0.5 * hard_units[
        ~teacher_positive
    ].mean()
    balanced_brier = 0.5 * brier_units[teacher_positive].mean() + 0.5 * brier_units[
        ~teacher_positive
    ].mean()
    true_positive = int((predicted_positive & teacher_positive).sum())
    predicted_count = int(predicted_positive.sum())
    precision = true_positive / max(predicted_count, 1)
    recall = true_positive / positives
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "pairs": int(margin.numel()),
        "teacher_positive_pairs": positives,
        "teacher_negative_pairs": negatives,
        "class_balanced_hard_bce": float(hard_bce),
        "class_balanced_soft_teacher_brier": float(balanced_brier),
        "teacher_positive_rate": positives / margin.numel(),
        "predicted_positive_rate": predicted_count / margin.numel(),
        "teacher_positive_precision": precision,
        "teacher_positive_recall": recall,
        "teacher_positive_f1": f1,
        "sampled_teacher_student_margin_rank_correlation": _rank_correlation(
            margin, target
        ),
        "rank_samples": RANK_SAMPLE_CAP,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    head: torch.nn.Module,
    scenes: Sequence[legacy.LoadedScene],
    teacher_center: torch.Tensor,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    visual = contrast_v2.evaluate(model, head, scenes, teacher_center, device)
    positive = positive_text.to(device)
    negative = negative_text.to(device)
    for scene in scenes:
        rows = legacy._scene_rows(scene)
        if rows.numel() != REGIONS_PER_SCENE:
            raise ValueError("DBA-v1 evaluation requires all 4096 canonical rows")
        margins: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for start in range(0, rows.numel(), EVAL_BATCH_ROWS):
            selected = rows[start : start + EVAL_BATCH_ROWS]
            descriptor = legacy._descriptor_batch(model, head, scene, selected, device)
            margins.append(
                boundary.exact_student_margin(descriptor, positive, negative).cpu()
            )
            teacher, mask = sparse_teacher.gather_sparse_teacher_batch(
                scene.shard, selected
            )
            targets.append(
                boundary.exact_multiview_teacher_probability(
                    teacher.to(device), mask.to(device), positive, negative
                ).cpu()
            )
        scene_boundary = boundary_metrics(torch.cat(margins), torch.cat(targets))
        visual["per_scene"][scene.binding.scene_id].update(scene_boundary)
    boundary_keys = (
        "class_balanced_hard_bce",
        "class_balanced_soft_teacher_brier",
        "teacher_positive_rate",
        "predicted_positive_rate",
        "teacher_positive_precision",
        "teacher_positive_recall",
        "teacher_positive_f1",
        "sampled_teacher_student_margin_rank_correlation",
    )
    for key in boundary_keys:
        visual[f"macro_{key}"] = sum(
            float(visual["per_scene"][scene][key]) for scene in VALIDATION_SCENES
        ) / len(VALIDATION_SCENES)
    return visual


def attach_selection(
    validation: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    if set(validation.get("per_scene", {})) != set(VALIDATION_SCENES) or set(
        baseline.get("per_scene", {})
    ) != set(VALIDATION_SCENES):
        raise ValueError("DBA-v1 gate requires both heldout source scenes")
    per_scene: dict[str, Any] = {}
    for scene in VALIDATION_SCENES:
        current = validation["per_scene"][scene]
        reference = baseline["per_scene"][scene]
        maximum_positive_rate = max(
            MAXIMUM_PREDICTED_POSITIVE_FLOOR,
            MAXIMUM_TEACHER_POSITIVE_MULTIPLIER
            * float(current["teacher_positive_rate"]),
        )
        boundary_checks = {
            "balanced_hard_bce_improves_at_least_1e_4": (
                float(current["class_balanced_hard_bce"])
                + BOUNDARY_BCE_IMPROVEMENT
                <= float(reference["class_balanced_hard_bce"])
            ),
            "teacher_positive_recall_improves_at_least_0p02": (
                float(current["teacher_positive_recall"])
                >= float(reference["teacher_positive_recall"])
                + RECALL_IMPROVEMENT
            ),
            "teacher_positive_f1_improves_at_least_0p01": (
                float(current["teacher_positive_f1"])
                >= float(reference["teacher_positive_f1"]) + F1_IMPROVEMENT
            ),
            "teacher_positive_precision_at_least_0p25": (
                float(current["teacher_positive_precision"]) >= MINIMUM_PRECISION
            ),
            "balanced_soft_teacher_brier_improves_at_least_1e_4": (
                float(current["class_balanced_soft_teacher_brier"])
                + BOUNDARY_BRIER_IMPROVEMENT
                <= float(reference["class_balanced_soft_teacher_brier"])
            ),
            "sampled_margin_rank_correlation_drop_at_most_0p005": (
                float(current["sampled_teacher_student_margin_rank_correlation"])
                + RANK_NON_REGRESSION_TOLERANCE
                >= float(reference["sampled_teacher_student_margin_rank_correlation"])
            ),
            "predicted_positive_rate_bounded": (
                float(current["predicted_positive_rate"]) <= maximum_positive_rate
            ),
        }
        visual_checks = {
            "mean_all_view_cosine_drop_at_most_0p002": (
                float(current["mean_all_view_cosine"])
                + RAW_MEAN_DROP_TOLERANCE
                >= float(reference["mean_all_view_cosine"])
            ),
            "p05_all_view_cosine_drop_at_most_0p005": (
                float(current["p05_row_mean_all_view_cosine"])
                + RAW_P05_DROP_TOLERANCE
                >= float(reference["p05_row_mean_all_view_cosine"])
            ),
            "centered_mean_drop_at_most_0p01": (
                float(current["mean_teacher_centered_residual_cosine"])
                + CENTERED_DROP_TOLERANCE
                >= float(reference["mean_teacher_centered_residual_cosine"])
            ),
            "centered_p05_drop_at_most_0p01": (
                float(current["p05_teacher_centered_residual_cosine"])
                + CENTERED_DROP_TOLERANCE
                >= float(reference["p05_teacher_centered_residual_cosine"])
            ),
            "student_to_teacher_spread_ratio_at_least_0p75": (
                float(current["student_to_teacher_spread_ratio"]) >= 0.75
            ),
            "centered_pair_gram_correlation_at_least_0p20": (
                float(current["teacher_centered_pair_gram_correlation"]) >= 0.20
            ),
            "centered_pair_gram_mae_increase_at_most_0p01": (
                float(current["teacher_centered_pair_gram_mae"])
                <= float(reference["teacher_centered_pair_gram_mae"])
                + PAIR_MAE_INCREASE_TOLERANCE
            ),
            "absolute_visual_probe_correlation_at_least_0p20": (
                float(current["absolute_visual_probe_response_correlation"])
                >= 0.20
            ),
            "absolute_visual_probe_mae_increase_at_most_0p01": (
                float(current["absolute_visual_probe_response_mae"])
                <= float(reference["absolute_visual_probe_response_mae"])
                + PROBE_MAE_INCREASE_TOLERANCE
            ),
            "absolute_visual_probe_std_ratio_in_0p75_1p25": (
                0.75
                <= float(current["absolute_visual_probe_response_std_ratio"])
                <= 1.25
            ),
        }
        per_scene[scene] = {
            "boundary_checks": boundary_checks,
            "visual_checks": visual_checks,
            "maximum_predicted_positive_rate": maximum_positive_rate,
            "passed": all(boundary_checks.values()) and all(visual_checks.values()),
        }
    selection = {
        "per_scene": per_scene,
        "macro_f1_improvement": (
            float(validation["macro_teacher_positive_f1"])
            - float(baseline["macro_teacher_positive_f1"])
        ),
        "macro_recall_improvement": (
            float(validation["macro_teacher_positive_recall"])
            - float(baseline["macro_teacher_positive_recall"])
        ),
        "macro_balanced_hard_bce_improvement": (
            float(baseline["macro_class_balanced_hard_bce"])
            - float(validation["macro_class_balanced_hard_bce"])
        ),
        "eligible": all(row["passed"] for row in per_scene.values()),
    }
    return {**dict(validation), "selection": selection}


def select_step(history: Sequence[Mapping[str, Any]]) -> int | None:
    expected = [
        0,
        *range(EVALUATION_INTERVAL, OPTIMIZER_STEPS + 1, EVALUATION_INTERVAL),
    ]
    if [int(row.get("step", -1)) for row in history] != expected:
        raise ValueError("DBA-v1 evaluation history schedule differs")
    eligible = [
        row
        for row in history
        if row.get("validation", {}).get("selection", {}).get("eligible") is True
    ]
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda row: (
            float(row["validation"]["selection"]["macro_f1_improvement"]),
            float(row["validation"]["selection"]["macro_recall_improvement"]),
            float(
                row["validation"]["selection"][
                    "macro_balanced_hard_bce_improvement"
                ]
            ),
            float(
                row["validation"][
                    "macro_mean_teacher_centered_residual_cosine"
                ]
            ),
            -int(row["step"]),
        ),
    )
    return int(selected["step"])


def _load_text_assets(
    authority: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    fit = text_assets._load_fit_text_bank(authority["fit_text_bank"])
    negative = relevance.load_frozen_canonical_negative_bank(
        authority["canonical_negative_bank"]["path"],
        expected_file_sha256=authority["canonical_negative_bank"]["sha256"],
    )
    positive_tensor = F.normalize(fit.embeddings.float(), dim=-1).to(device)
    negative_tensor = F.normalize(negative.embeddings.float(), dim=-1).to(device)
    if positive_tensor.shape != (FIT_QUERY_ROWS, 1536) or negative_tensor.shape != (
        boundary.CANONICAL_NEGATIVE_ROWS,
        1536,
    ):
        raise ValueError("DBA-v1 frozen text bank axes differ")
    return positive_tensor, negative_tensor


def _batch_objective(
    model: torch.nn.Module,
    head: torch.nn.Module,
    scene: legacy.LoadedScene,
    rows: torch.Tensor,
    teacher_center: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Evaluate the complete visual and DBA objectives with one model forward."""

    descriptor = legacy._descriptor_batch(model, head, scene, rows, device)
    teacher, mask = sparse_teacher.gather_sparse_teacher_batch(scene.shard, rows)
    teacher_device = teacher.to(device)
    mask_device = mask.to(device)
    visual_loss, visual_parts = contrast_v2.contrast.contrast_preserving_objective(
        descriptor,
        teacher_device,
        mask_device,
        teacher_center.to(device),
    )
    aligned = boundary.source_balanced_boundary_alignment_loss(
        descriptor,
        teacher_device,
        mask_device,
        positive,
        negative,
    )
    total = visual_loss + DBA_AUXILIARY_WEIGHT * aligned.loss
    parts: dict[str, float | int] = {
        "total": float(total.detach().cpu()),
        "visual_total": float(visual_loss.detach().cpu()),
        "dba_complete": float(aligned.loss.detach().cpu()),
        "dba_weighted": float(
            (DBA_AUXILIARY_WEIGHT * aligned.loss).detach().cpu()
        ),
        "dba_balanced_hard_boundary": float(
            aligned.balanced_hard_boundary_loss.detach().cpu()
        ),
        "dba_balanced_soft_fidelity": float(
            aligned.balanced_soft_fidelity_loss.detach().cpu()
        ),
        "dba_teacher_positive_pairs": aligned.teacher_positive_pairs,
        "dba_teacher_negative_pairs": aligned.teacher_negative_pairs,
    }
    parts.update(
        {
            f"visual_{name}": float(value.detach().cpu())
            for name, value in visual_parts.items()
        }
    )
    return total, parts


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    result_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or output.is_symlink() or result_path.exists() or result_path.is_symlink():
        raise FileExistsError("DBA-v1 first-writer output already exists")
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    exclusion_raw, _, _ = load_json_object(
        prepared.authority["benchmark_exclusion_manifest"]["path"],
        expected_sha256=prepared.authority["benchmark_exclusion_manifest"]["sha256"],
        label="DBA-v1 benchmark exclusion manifest",
    )
    exclusion.validate_benchmark_exclusion_manifest(exclusion_raw)
    train_scenes = [legacy.load_scene(binding) for binding in prepared.source.train]
    validation_scenes = [
        legacy.load_scene(binding) for binding in prepared.source.validation
    ]
    for scene in (*train_scenes, *validation_scenes):
        rows = legacy._scene_rows(scene)
        if rows.numel() != REGIONS_PER_SCENE:
            raise ValueError("DBA-v1 requires all 4096 canonical rows")

    device = torch.device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    gate = prepared.source_gate
    model = readout.build_model(DIRECTION_ONLY, gate["normalization"])
    model.load_state_dict(gate["checkpoint"]["model_state_dict"], strict=True)
    model = model.to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        gate["official_radio_checkpoint"]["path"],
        expected_sha256=gate["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    positive, negative = _load_text_assets(prepared.authority, device)
    teacher_center = gate["contrast_reference"]["teacher_center"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    baseline = evaluate(
        model,
        head,
        validation_scenes,
        teacher_center,
        positive,
        negative,
        device,
    )
    zero_state = contrast_v2._state_copy(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "training_scene_objective": None,
            "validation": attach_selection(baseline, baseline),
            "model_state_dict_sha256": contrast_v2._state_sha(zero_state),
        }
    ]
    saved_states: dict[int, dict[str, torch.Tensor]] = {}
    last_training: dict[str, Any] | None = None
    for step in range(1, OPTIMIZER_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scene_objective: dict[str, Any] = {}
        for scene in train_scenes:
            rows = legacy._cyclic_batch(legacy._scene_rows(scene), step=step)
            total, parts = _batch_objective(
                model,
                head,
                scene,
                rows,
                teacher_center,
                positive,
                negative,
                device,
            )
            (total / len(train_scenes)).backward()
            scene_objective[scene.binding.scene_id] = parts
        torch.nn.utils.clip_grad_norm_(
            tuple(model.parameters()),
            MAX_GRADIENT_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()
        last_training = {"step": step, "per_scene": scene_objective}
        if step % EVALUATION_INTERVAL != 0:
            continue
        validation = attach_selection(
            evaluate(
                model,
                head,
                validation_scenes,
                teacher_center,
                positive,
                negative,
                device,
            ),
            baseline,
        )
        state = contrast_v2._state_copy(model)
        entry = {
            "step": step,
            "training_scene_objective": scene_objective,
            "validation": validation,
            "model_state_dict_sha256": contrast_v2._state_sha(state),
        }
        history.append(entry)
        if validation["selection"]["eligible"] is True:
            saved_states[step] = state
        print(json.dumps(entry, sort_keys=True), flush=True)

    selected = select_step(history)
    checkpoint_record: dict[str, str] | None = None
    if selected is not None:
        selected_state = saved_states[selected]
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "training_contract": training_contract(),
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "interface_contract_sha256": readout.INTERFACE_CONTRACT_SHA256,
            "model_architecture": model.architecture(
                readout.INTERFACE_CONTRACT_SHA256
            ),
            "model_state_dict": selected_state,
            "model_state_dict_sha256": contrast_v2._state_sha(selected_state),
            "warm_start_source_contrast_v21_result": dict(
                prepared.authority["source_contrast_v21_result"]
            ),
            "warm_start_source_contrast_v21_checkpoint": dict(
                gate["result"]["checkpoint"]
            ),
            "normalization": dict(gate["result"]["normalization"]),
            "contrast_reference": dict(gate["result"]["contrast_reference"]),
            "fit_text_bank": dict(prepared.authority["fit_text_bank"]),
            "canonical_negative_bank": dict(
                prepared.authority["canonical_negative_bank"]
            ),
            "execution_authority": dict(prepared.authority["verified_record"]),
            "selected_step": selected,
            "source_access": source_access(),
        }
        checkpoint_file = write_torch_noclobber(output, checkpoint)
        checkpoint_record = file_record(checkpoint_file)
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "source_only_dba_v1_promotion_candidate_complete"
            if selected is not None
            else "source_only_dba_v1_complete_no_eligible_candidate"
        ),
        "arm": DIRECTION_ONLY,
        "training_contract": training_contract(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "execution_authority": dict(prepared.authority["verified_record"]),
        "input_authority": {
            "source_contrast_v21_result": dict(
                prepared.authority["source_contrast_v21_result"]
            ),
            "source_contrast_v21_checkpoint": dict(gate["result"]["checkpoint"]),
            "fit_text_bank": dict(prepared.authority["fit_text_bank"]),
            "canonical_negative_bank": dict(
                prepared.authority["canonical_negative_bank"]
            ),
            "benchmark_exclusion_manifest": dict(
                prepared.authority["benchmark_exclusion_manifest"]
            ),
        },
        "checkpoint": checkpoint_record,
        "selected_step": selected,
        "history": history,
        "last_training_step": last_training,
        "target_query_or_metric_authorized": False,
        "benchmark_opened": False,
        "source_access": source_access(),
    }
    write_frozen_json(result_path, report)
    return report


def synthetic_dry_run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    student = F.normalize(torch.randn(8, 16), dim=-1).requires_grad_(True)
    teacher = F.normalize(torch.randn(8, 2, 16), dim=-1)
    mask = torch.ones(8, 2, dtype=torch.bool)
    positive = F.normalize(torch.randn(FIT_QUERY_ROWS, 16), dim=-1)
    negative = F.normalize(torch.randn(4, 16), dim=-1)
    output = boundary.source_balanced_boundary_alignment_loss(
        student, teacher, mask, positive, negative
    )
    output.loss.backward()
    return {
        "schema": "radio_gs.factorized_native_dba_v1_synthetic_dry_run.v1",
        "loss_finite": bool(torch.isfinite(output.loss)),
        "gradient_finite_and_nonzero": (
            student.grad is not None
            and bool(torch.isfinite(student.grad).all())
            and float(student.grad.norm()) > 0
        ),
        "training_steps": OPTIMIZER_STEPS,
        "rows_per_scene_per_step": BATCH_ROWS,
        "complete_rows_per_scene": OPTIMIZER_STEPS * BATCH_ROWS,
        "evaluation_steps": [
            0,
            *range(EVALUATION_INTERVAL, OPTIMIZER_STEPS + 1, EVALUATION_INTERVAL),
        ],
        "dba_auxiliary_weight": DBA_AUXILIARY_WEIGHT,
        "target_query_or_benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-dry-run")
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    run = commands.add_parser("train")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "synthetic-dry-run":
        result = synthetic_dry_run()
    elif args.command == "validate-authority":
        prepared = prepare_inputs(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
        )
        result = {
            "status": "source_only_dba_v1_authority_validated",
            "source_train": [item.scene_id for item in prepared.source.train],
            "source_validation": [
                item.scene_id for item in prepared.source.validation
            ],
            "warm_start_selected_step": prepared.source_gate["selected_step"],
            "target_query_or_benchmark_opened": False,
        }
    else:
        result = train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DBA_AUXILIARY_WEIGHT",
    "EVALUATION_INTERVAL",
    "EXECUTION_AUTHORITY_SCHEMA",
    "OPTIMIZER_STEPS",
    "RESULT_SCHEMA",
    "TRAINING_CONTRACT_SHA256",
    "attach_selection",
    "boundary_metrics",
    "build_parser",
    "prepare_inputs",
    "select_step",
    "source_access",
    "synthetic_dry_run",
    "train",
    "training_contract",
    "validate_execution_authority_header",
]
