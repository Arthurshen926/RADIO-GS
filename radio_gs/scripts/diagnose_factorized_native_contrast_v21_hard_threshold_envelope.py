#!/usr/bin/env python3
"""Run an exact source-only global hard-threshold PR/F1 envelope diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces import factorized_native_source_global_hard_threshold_envelope as formal
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance_loss
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import calibrate_factorized_native_contrast_v21_global_margin as v1_runner
from radio_gs.scripts import train_factorized_native_gauge_state_readout_exact4x2_contrast_v21 as contrast_v21
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


DESCRIPTOR_BATCH_ROWS = v1_runner.DESCRIPTOR_BATCH_ROWS


def classification_metrics(
    margin: torch.Tensor,
    hard_teacher: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float | int]:
    x = torch.as_tensor(margin).detach().float().cpu().contiguous()
    hard = torch.as_tensor(hard_teacher).detach().bool().cpu().contiguous()
    if (
        x.ndim != 1
        or hard.shape != x.shape
        or x.numel() <= 0
        or not bool(torch.isfinite(x).all())
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("hard-threshold classification inputs differ")
    predicted = x >= float(threshold)
    true_positive = int((predicted & hard).sum())
    predicted_count = int(predicted.sum())
    teacher_count = int(hard.sum())
    precision = true_positive / predicted_count if predicted_count > 0 else 0.0
    recall = true_positive / teacher_count if teacher_count > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "pairs": int(x.numel()),
        "teacher_positive_count": teacher_count,
        "predicted_positive_count": predicted_count,
        "true_positive_count": true_positive,
        "teacher_positive_rate": teacher_count / int(x.numel()),
        "predicted_positive_rate": predicted_count / int(x.numel()),
        "teacher_positive_precision": precision,
        "teacher_positive_recall": recall,
        "teacher_positive_f1": f1,
    }


def exact_grouped_pr_curve(
    margin: torch.Tensor,
    hard_teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(margin).detach().float().cpu().contiguous()
    hard = torch.as_tensor(hard_teacher).detach().bool().cpu().contiguous()
    if (
        x.ndim != 1
        or hard.shape != x.shape
        or x.numel() <= 0
        or not bool(torch.isfinite(x).all())
        or not bool(hard.any())
    ):
        raise ValueError("hard-threshold PR curve inputs differ")
    order = torch.argsort(x, descending=True, stable=True)
    score = x[order]
    positive = hard[order].to(torch.int64)
    cumulative_positive = positive.cumsum(dim=0)
    prefix = torch.arange(1, x.numel() + 1, dtype=torch.int64)
    group_end = torch.ones(x.numel(), dtype=torch.bool)
    group_end[:-1] = score[:-1] != score[1:]
    return (
        score[group_end].contiguous(),
        cumulative_positive[group_end].contiguous(),
        prefix[group_end].contiguous(),
    )


def select_train_threshold(
    margin: torch.Tensor,
    hard_teacher: torch.Tensor,
) -> dict[str, Any]:
    x = torch.as_tensor(margin).detach().float().cpu().contiguous()
    hard = torch.as_tensor(hard_teacher).detach().bool().cpu().contiguous()
    identity = classification_metrics(x, hard, threshold=formal.IDENTITY_THRESHOLD)
    precision_floor = max(
        formal.MINIMUM_PRECISION,
        formal.MINIMUM_IDENTITY_PRECISION_RETENTION
        * float(identity["teacher_positive_precision"]),
    )
    thresholds, true_positive, predicted = exact_grouped_pr_curve(x, hard)
    teacher_count = int(hard.sum())
    precision = true_positive.double() / predicted.double()
    recall = true_positive.double() / float(teacher_count)
    f1 = torch.where(
        precision + recall > 0.0,
        2.0 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    eligible = (
        (precision >= precision_floor)
        & (
            recall
            >= float(identity["teacher_positive_recall"])
            + formal.METRIC_STRICT_IMPROVEMENT
        )
        & (
            f1
            >= float(identity["teacher_positive_f1"])
            + formal.METRIC_STRICT_IMPROVEMENT
        )
    )
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_indices.numel() <= 0:
        raise RuntimeError("no train threshold satisfies the frozen eligibility rule")
    # Thresholds are descending, and torch.argmax returns the first exact maximum.
    selected_local = int(torch.argmax(f1[eligible_indices]))
    selected_index = int(eligible_indices[selected_local])
    threshold = float(thresholds[selected_index])
    candidate = classification_metrics(x, hard, threshold=threshold)
    checks = formal.expected_checks(identity, candidate)
    if not all(checks.values()):
        raise RuntimeError("selected train threshold does not satisfy its frozen checks")
    return {
        "threshold": threshold,
        "unique_thresholds_swept": int(thresholds.numel()),
        "precision_floor": precision_floor,
        "identity": identity,
        "candidate": candidate,
        "candidate_checks": checks,
        "selection_objective": "maximum_train_hard_f1_among_eligible_thresholds",
        "validation_contribution": False,
    }


def exact_unified_validation_oracle(
    margins: Mapping[str, torch.Tensor],
    hard_teachers: Mapping[str, torch.Tensor],
    identities: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    if tuple(margins) != formal.VALIDATION_SCENES or tuple(hard_teachers) != formal.VALIDATION_SCENES:
        raise ValueError("hard-threshold oracle validation scene order differs")
    left_scene, right_scene = formal.VALIDATION_SCENES
    left_margin = torch.as_tensor(margins[left_scene]).detach().float().cpu().contiguous()
    right_margin = torch.as_tensor(margins[right_scene]).detach().float().cpu().contiguous()
    left_hard = torch.as_tensor(hard_teachers[left_scene]).detach().bool().cpu().contiguous()
    right_hard = torch.as_tensor(hard_teachers[right_scene]).detach().bool().cpu().contiguous()
    if (
        left_margin.ndim != 1
        or right_margin.ndim != 1
        or left_hard.shape != left_margin.shape
        or right_hard.shape != right_margin.shape
        or not bool(left_hard.any())
        or not bool(right_hard.any())
    ):
        raise ValueError("hard-threshold oracle inputs differ")
    margin = torch.cat((left_margin, right_margin))
    scene = torch.cat(
        (
            torch.zeros(left_margin.numel(), dtype=torch.bool),
            torch.ones(right_margin.numel(), dtype=torch.bool),
        )
    )
    hard = torch.cat((left_hard, right_hard))
    order = torch.argsort(margin, descending=True, stable=True)
    score = margin[order]
    sorted_scene = scene[order]
    sorted_hard = hard[order]
    group_end = torch.ones(score.numel(), dtype=torch.bool)
    group_end[:-1] = score[:-1] != score[1:]
    thresholds = score[group_end].contiguous()
    metrics_by_scene: dict[str, dict[str, torch.Tensor]] = {}
    feasible = torch.ones(thresholds.numel(), dtype=torch.bool)
    for scene_id, membership, teacher in (
        (left_scene, ~sorted_scene, left_hard),
        (right_scene, sorted_scene, right_hard),
    ):
        predicted = membership.to(torch.int64).cumsum(dim=0)[group_end]
        true_positive = (membership & sorted_hard).to(torch.int64).cumsum(dim=0)[group_end]
        teacher_count = int(teacher.sum())
        precision = torch.where(
            predicted > 0,
            true_positive.double() / predicted.double(),
            torch.zeros_like(predicted, dtype=torch.float64),
        )
        recall = true_positive.double() / float(teacher_count)
        f1 = torch.where(
            precision + recall > 0.0,
            2.0 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )
        identity = identities[scene_id]
        precision_floor = max(
            formal.MINIMUM_PRECISION,
            formal.MINIMUM_IDENTITY_PRECISION_RETENTION
            * float(identity["teacher_positive_precision"]),
        )
        scene_feasible = (
            (precision >= precision_floor)
            & (
                recall
                >= float(identity["teacher_positive_recall"])
                + formal.METRIC_STRICT_IMPROVEMENT
            )
            & (
                f1
                >= float(identity["teacher_positive_f1"])
                + formal.METRIC_STRICT_IMPROVEMENT
            )
        )
        feasible &= scene_feasible
        metrics_by_scene[scene_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    feasible_indices = torch.nonzero(feasible, as_tuple=False).flatten()
    exists = feasible_indices.numel() > 0
    representative_threshold: float | None = None
    representative_metrics: dict[str, Any] | None = None
    if exists:
        mean_f1 = 0.5 * (
            metrics_by_scene[left_scene]["f1"]
            + metrics_by_scene[right_scene]["f1"]
        )
        local = int(torch.argmax(mean_f1[feasible_indices]))
        index = int(feasible_indices[local])
        representative_threshold = float(thresholds[index])
        representative_metrics = {}
        for scene_id in formal.VALIDATION_SCENES:
            observed = classification_metrics(
                margins[scene_id], hard_teachers[scene_id],
                threshold=representative_threshold,
            )
            checks = formal.expected_checks(identities[scene_id], observed)
            if not all(checks.values()):
                raise RuntimeError("oracle representative is not exactly feasible")
            representative_metrics[scene_id] = {
                "metrics": observed,
                "checks": checks,
            }
    return {
        "role": "validation_label_oracle_ranking_diagnostic_only",
        "joint_unique_thresholds_swept": int(thresholds.numel()),
        "unified_feasible_threshold_exists": bool(exists),
        "unified_feasible_threshold_count": int(feasible_indices.numel()),
        "representative_threshold": representative_threshold,
        "representative_metrics": representative_metrics,
        "promotion_authorized": False,
        "parameter_export_authorized": False,
    }


def _prepared_source(execution: Mapping[str, Any]):
    result = execution["verified_source_gate"]["result"]
    prepared = contrast_v21.prepare_inputs(
        result["execution_authority"]["path"],
        expected_sha256=result["execution_authority"]["sha256"],
    )
    source = prepared.base_v2.source
    if (
        tuple(item.scene_id for item in source.train) != formal.TRAIN_SCENES
        or tuple(item.scene_id for item in source.validation) != formal.VALIDATION_SCENES
        or source.authority["benchmark_exclusion_manifest"]
        != execution["benchmark_exclusion_manifest"]
    ):
        raise ValueError("hard-threshold exact4x2 source cohort differs")
    return source


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber hard-threshold envelope: {output}")
    execution = formal.validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    source_inputs = _prepared_source(execution)
    fit_bank = v1_runner._load_fit_text_bank(execution["fit_text_bank"])
    negative_bank = relevance_loss.load_frozen_canonical_negative_bank(
        execution["canonical_negative_bank"]["path"],
        expected_file_sha256=execution["canonical_negative_bank"]["sha256"],
    )
    exclusion_raw, _, _ = load_json_object(
        execution["benchmark_exclusion_manifest"]["path"],
        expected_sha256=execution["benchmark_exclusion_manifest"]["sha256"],
        label="hard-threshold benchmark exclusion manifest",
    )
    base_trainer.validate_benchmark_exclusion_manifest(exclusion_raw)

    source_gate = execution["verified_source_gate"]
    device = torch.device(str(args.device))
    model = readout.build_model(DIRECTION_ONLY, source_gate["normalization"])
    model.load_state_dict(source_gate["checkpoint"]["model_state_dict"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        source_gate["official_radio_checkpoint"]["path"],
        expected_sha256=source_gate["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    positive = F.normalize(fit_bank.embeddings.float(), dim=-1)
    negative = F.normalize(negative_bank.embeddings.float(), dim=-1)

    train_margins: list[torch.Tensor] = []
    train_hard: list[torch.Tensor] = []
    for binding in source_inputs.train:
        margin, target = v1_runner.scene_margin_pairs(
            v1_runner.legacy.load_scene(binding), model=model, head=head,
            positive_text=positive, canonical_negative_text=negative,
            device=device, batch_rows=int(args.batch_rows),
        )
        train_margins.append(margin)
        train_hard.append(target >= formal.HARD_BOUNDARY)
    fit_margin = torch.cat(train_margins)
    fit_hard = torch.cat(train_hard)
    train_selection = select_train_threshold(fit_margin, fit_hard)
    candidate_threshold = float(train_selection.pop("threshold"))
    del train_margins, train_hard, fit_margin, fit_hard

    validation_margins: dict[str, torch.Tensor] = {}
    validation_hard: dict[str, torch.Tensor] = {}
    validation: dict[str, Any] = {}
    identities: dict[str, Mapping[str, float | int]] = {}
    for binding in source_inputs.validation:
        margin, target = v1_runner.scene_margin_pairs(
            v1_runner.legacy.load_scene(binding), model=model, head=head,
            positive_text=positive, canonical_negative_text=negative,
            device=device, batch_rows=int(args.batch_rows),
        )
        hard = target >= formal.HARD_BOUNDARY
        identity = classification_metrics(
            margin, hard, threshold=formal.IDENTITY_THRESHOLD
        )
        candidate = classification_metrics(
            margin, hard, threshold=candidate_threshold
        )
        validation[binding.scene_id] = {
            "identity": identity,
            "candidate": candidate,
            "checks": formal.expected_checks(identity, candidate),
        }
        validation_margins[binding.scene_id] = margin
        validation_hard[binding.scene_id] = hard
        identities[binding.scene_id] = identity
    oracle = exact_unified_validation_oracle(
        validation_margins, validation_hard, identities
    )
    promotion_checks = {
        "train_candidate_passed": all(train_selection["candidate_checks"].values()),
        "every_validation_scene_passed": all(
            all(validation[scene]["checks"].values())
            for scene in formal.VALIDATION_SCENES
        ),
        "candidate_is_one_global_train_selected_threshold": True,
    }
    promoted = all(promotion_checks.values())
    result = {
        "schema": formal.RESULT_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "source_only_promoted" if promoted else "source_only_complete_no_promotion",
        "contract": formal.envelope_contract(),
        "contract_sha256": formal.ENVELOPE_CONTRACT_SHA256,
        "execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_contrast_v21_result": dict(execution["source_contrast_v21_result"]),
            "source_contrast_v21_checkpoint": dict(source_gate["result"]["checkpoint"]),
            "class_balanced_v2_result": dict(execution["class_balanced_v2_result"]),
            "fit_text_bank": dict(execution["fit_text_bank"]),
            "canonical_negative_bank": dict(execution["canonical_negative_bank"]),
            "benchmark_exclusion_manifest": dict(execution["benchmark_exclusion_manifest"]),
        },
        "thresholds": {
            "identity": formal.IDENTITY_THRESHOLD,
            "train_selected_candidate": candidate_threshold,
        },
        "sampling_audit": {
            "train_scenes": list(formal.TRAIN_SCENES),
            "validation_scenes": list(formal.VALIDATION_SCENES),
            "regions_per_scene": formal.REGIONS_PER_SCENE,
            "query_rows": formal.FIT_QUERY_ROWS,
            "canonical_negative_rows": formal.CANONICAL_NEGATIVE_ROWS,
            "train_pairs": len(formal.TRAIN_SCENES) * formal.REGIONS_PER_SCENE * formal.FIT_QUERY_ROWS,
            "validation_pairs_per_scene": formal.REGIONS_PER_SCENE * formal.FIT_QUERY_ROWS,
            "all_rows_and_queries_used": True,
            "validation_contribution_to_train_selection": False,
        },
        "train_selection": train_selection,
        "validation_audit": validation,
        "promotion_checks": promotion_checks,
        "no_promotion_oracle": oracle,
        "global_threshold_authorized": promoted,
        "source_access": formal.source_access(),
    }
    result = formal.validate_result(result)
    write_frozen_json(output, result)
    return {
        "status": result["status"],
        "thresholds": result["thresholds"],
        "train_selection": result["train_selection"],
        "validation_audit": result["validation_audit"],
        "promotion_checks": result["promotion_checks"],
        "no_promotion_oracle": result["no_promotion_oracle"],
        "global_threshold_authorized": result["global_threshold_authorized"],
        "output": file_record(output),
        "source_access": formal.source_access(),
    }


def _bound_record(path: str, digest: str, *, label: str) -> dict[str, str]:
    value = {"path": str(Path(path).expanduser().resolve()), "sha256": str(digest)}
    validate_file_record(value, label=label)
    return value


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.authority_output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber hard-threshold authority: {output}")
    source_record = _bound_record(
        args.source_contrast_v21_result,
        args.source_contrast_v21_result_sha256,
        label="contrast V2.1 source result",
    )
    source_gate = formal.source_formal.validate_source_contrast_v21_result(source_record)
    prepared = contrast_v21.prepare_inputs(
        source_gate["result"]["execution_authority"]["path"],
        expected_sha256=source_gate["result"]["execution_authority"]["sha256"],
    )
    v2_result = _bound_record(
        args.class_balanced_v2_result,
        args.class_balanced_v2_result_sha256,
        label="class-balanced V2 diagnostic result",
    )
    checked_v2 = formal._validate_v2_diagnostic(v2_result)
    fit = _bound_record(args.fit_text_bank, args.fit_text_bank_sha256, label="fit text bank")
    negative = _bound_record(
        args.canonical_negative_bank,
        args.canonical_negative_bank_sha256,
        label="canonical negative bank",
    )
    exclusion = _bound_record(
        args.benchmark_exclusion_manifest,
        args.benchmark_exclusion_manifest_sha256,
        label="benchmark exclusion manifest",
    )
    if checked_v2["input_authority"]["source_contrast_v21_result"] != source_record:
        raise ValueError("hard-threshold and V2 source result differ")
    for name, value in (
        ("fit_text_bank", fit),
        ("canonical_negative_bank", negative),
        ("benchmark_exclusion_manifest", exclusion),
    ):
        if checked_v2["input_authority"][name] != value:
            raise ValueError(f"hard-threshold and V2 {name} differ")
    if prepared.base_v2.source.authority["benchmark_exclusion_manifest"] != exclusion:
        raise ValueError("hard-threshold exclusion differs from exact4x2 source")
    v1_runner._load_fit_text_bank(fit)
    relevance_loss.load_frozen_canonical_negative_bank(
        negative["path"], expected_file_sha256=negative["sha256"]
    )
    exclusion_raw, _, _ = load_json_object(
        exclusion["path"], expected_sha256=exclusion["sha256"],
        label="hard-threshold benchmark exclusion manifest",
    )
    base_trainer.validate_benchmark_exclusion_manifest(exclusion_raw)
    envelope_output = str(Path(args.envelope_output).expanduser().resolve())
    authority = {
        "schema": formal.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "authorized_source_only_hard_threshold_envelope",
        "implementation": file_record(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in formal.IMPLEMENTATION_DEPENDENCIES.items()
        },
        "envelope_contract_sha256": formal.ENVELOPE_CONTRACT_SHA256,
        "source_contrast_v21_result": source_record,
        "class_balanced_v2_result": v2_result,
        "fit_text_bank": fit,
        "canonical_negative_bank": negative,
        "benchmark_exclusion_manifest": exclusion,
        "envelope_output": envelope_output,
        "diagnostic_authorized": True,
        "target_execution_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "source_access": formal.source_access(),
    }
    write_frozen_json(output, authority)
    return {
        "status": "source_global_hard_threshold_envelope_authority_built",
        "authority": file_record(output),
        "envelope_output": envelope_output,
        "source_selected_step": source_gate["selected_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--source-contrast-v21-result", required=True)
    build.add_argument("--source-contrast-v21-result-sha256", required=True)
    build.add_argument("--class-balanced-v2-result", required=True)
    build.add_argument("--class-balanced-v2-result-sha256", required=True)
    build.add_argument("--fit-text-bank", required=True)
    build.add_argument("--fit-text-bank-sha256", required=True)
    build.add_argument("--canonical-negative-bank", required=True)
    build.add_argument("--canonical-negative-bank-sha256", required=True)
    build.add_argument("--benchmark-exclusion-manifest", required=True)
    build.add_argument("--benchmark-exclusion-manifest-sha256", required=True)
    build.add_argument("--envelope-output", required=True)
    build.add_argument("--authority-output", required=True)
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    run = commands.add_parser("diagnose")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-rows", type=int, default=DESCRIPTOR_BATCH_ROWS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-authority":
        result = build_authority(args)
    elif args.command == "validate-authority":
        authority = formal.validate_execution_authority(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
        )
        result = {
            "status": "source_global_hard_threshold_envelope_authority_validated",
            "source_selected_step": authority["verified_source_gate"]["selected_step"],
            "target_or_query_opened": False,
        }
    else:
        result = diagnose(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "build_authority", "build_parser", "classification_metrics", "diagnose",
    "exact_grouped_pr_curve", "exact_unified_validation_oracle",
    "select_train_threshold",
]
