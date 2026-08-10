#!/usr/bin/env python3
"""Fit the source-only class-balanced global margin calibrator V2."""

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
from radio_gs.interfaces import factorized_native_source_global_margin_calibration_v2 as formal
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


def train_mass_weights(target: torch.Tensor) -> dict[str, float | bool]:
    y = torch.as_tensor(target).detach().double().cpu().contiguous()
    if (
        y.ndim != 1
        or y.numel() <= 0
        or not bool(torch.isfinite(y).all())
        or bool((y < 0).any())
        or bool((y > 1).any())
    ):
        raise ValueError("class-balanced train teacher target differs")
    positive_rate = float(y.mean())
    negative_rate = 1.0 - positive_rate
    if not 0.0 < positive_rate < 1.0 or not 0.0 < negative_rate < 1.0:
        raise ValueError("class-balanced train teacher mass is degenerate")
    positive_weight = 0.5 / positive_rate
    negative_weight = 0.5 / negative_rate
    return {
        "teacher_soft_positive_rate": positive_rate,
        "teacher_soft_negative_rate": negative_rate,
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
        "weighted_positive_mass_fraction": positive_weight * positive_rate,
        "weighted_negative_mass_fraction": negative_weight * negative_rate,
        "derived_from_train_only": True,
        "validation_reuses_fixed_train_weights": True,
    }


def _balanced_soft_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    return (
        float(positive_weight) * target * F.softplus(-logits)
        + float(negative_weight) * (1.0 - target) * F.softplus(logits)
    ).mean()


def fit_global_calibrator(
    margin: torch.Tensor,
    target: torch.Tensor,
    weights: Mapping[str, float | bool],
) -> tuple[dict[str, float], dict[str, Any]]:
    x = torch.as_tensor(margin).detach().double().cpu().contiguous()
    y = torch.as_tensor(target).detach().double().cpu().contiguous()
    expected_pairs = (
        len(formal.TRAIN_SCENES)
        * formal.REGIONS_PER_SCENE
        * formal.FIT_QUERY_ROWS
    )
    positive_weight = float(weights["positive_weight"])
    negative_weight = float(weights["negative_weight"])
    if (
        x.ndim != 1
        or y.shape != x.shape
        or x.numel() != expected_pairs
        or not bool(torch.isfinite(x).all())
        or not bool(torch.isfinite(y).all())
        or bool((y < 0).any())
        or bool((y > 1).any())
        or float(x.std()) <= 1e-8
        or not math.isfinite(positive_weight)
        or not math.isfinite(negative_weight)
        or positive_weight <= 0.0
        or negative_weight <= 0.0
    ):
        raise ValueError("class-balanced global margin fit samples differ")
    observed = train_mass_weights(y)
    if (
        abs(positive_weight - float(observed["positive_weight"])) > 1e-10
        or abs(negative_weight - float(observed["negative_weight"])) > 1e-10
    ):
        raise ValueError("class-balanced global margin weights were not train-derived")

    theta = torch.tensor(
        [formal.IDENTITY_A, formal.IDENTITY_B], dtype=torch.float64
    )
    initial = float(
        _balanced_soft_bce_logits(
            theta[0] * x + theta[1], y,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
        )
    )
    converged = False
    iterations = 0
    reductions = 0
    final_gradient = math.inf
    minimum_hessian_eigenvalue = math.inf
    positive_mass = positive_weight * y
    negative_mass = negative_weight * (1.0 - y)
    total_mass = positive_mass + negative_mass
    for iteration in range(1, formal.MAXIMUM_NEWTON_ITERATIONS + 1):
        logits = theta[0] * x + theta[1]
        probability = torch.sigmoid(logits)
        derivative = total_mass * probability - positive_mass
        curvature = total_mass * probability * (1.0 - probability)
        gradient = torch.stack(((derivative * x).mean(), derivative.mean()))
        hessian = torch.stack(
            (
                torch.stack(
                    ((curvature * x.square()).mean(), (curvature * x).mean())
                ),
                torch.stack(((curvature * x).mean(), curvature.mean())),
            )
        )
        eigenvalue = float(torch.linalg.eigvalsh(hessian).min())
        minimum_hessian_eigenvalue = min(minimum_hessian_eigenvalue, eigenvalue)
        if not math.isfinite(eigenvalue) or eigenvalue <= 1e-12:
            raise RuntimeError("class-balanced calibration Hessian is not positive definite")
        final_gradient = float(gradient.abs().max())
        iterations = iteration
        if final_gradient <= formal.OPTIMIZATION_GRADIENT_TOLERANCE:
            converged = True
            break
        step = torch.linalg.solve(hessian, gradient)
        current = float(
            _balanced_soft_bce_logits(
                logits, y,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
            )
        )
        accepted = False
        scale = 1.0
        for _ in range(formal.MAXIMUM_BACKTRACK_STEPS):
            candidate = theta - scale * step
            if float(candidate[0]) > formal.MINIMUM_POSITIVE_SLOPE:
                objective = float(
                    _balanced_soft_bce_logits(
                        candidate[0] * x + candidate[1], y,
                        positive_weight=positive_weight,
                        negative_weight=negative_weight,
                    )
                )
                if math.isfinite(objective) and objective < current:
                    theta = candidate
                    accepted = True
                    break
            scale *= 0.5
            reductions += 1
        if not accepted:
            raise RuntimeError("class-balanced Newton backtracking failed")
    final = float(
        _balanced_soft_bce_logits(
            theta[0] * x + theta[1], y,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
        )
    )
    if (
        not converged
        or float(theta[0]) <= formal.MINIMUM_POSITIVE_SLOPE
        or not math.isfinite(final)
        or final >= initial
    ):
        raise RuntimeError("class-balanced calibration did not reach its unique optimum")
    return (
        {"a": float(theta[0]), "b": float(theta[1])},
        {
            "solver": "deterministic_strictly_convex_two_parameter_newton_irls",
            "iterations": iterations,
            "converged": converged,
            "initial_balanced_soft_binary_cross_entropy": initial,
            "final_balanced_soft_binary_cross_entropy": final,
            "final_max_absolute_gradient": final_gradient,
            "minimum_observed_hessian_eigenvalue": minimum_hessian_eigenvalue,
            "backtrack_reductions": reductions,
        },
    )


def calibration_metrics(
    margin: torch.Tensor,
    target: torch.Tensor,
    *,
    a: float,
    b: float,
    positive_weight: float,
    negative_weight: float,
    fixed_rank_correlation: float | None = None,
) -> dict[str, float | int]:
    x = torch.as_tensor(margin).detach().float().cpu().contiguous()
    y = torch.as_tensor(target).detach().float().cpu().contiguous()
    if (
        x.ndim != 1
        or y.shape != x.shape
        or not math.isfinite(float(a))
        or not math.isfinite(float(b))
        or float(a) <= 0.0
        or not math.isfinite(float(positive_weight))
        or not math.isfinite(float(negative_weight))
        or float(positive_weight) <= 0.0
        or float(negative_weight) <= 0.0
    ):
        raise ValueError("class-balanced global margin metric inputs differ")
    logits = float(a) * x + float(b)
    probability = torch.sigmoid(logits)
    teacher_positive = y >= 0.5
    predicted_positive = probability >= 0.5
    true_positive = int((teacher_positive & predicted_positive).sum())
    predicted_count = int(predicted_positive.sum())
    teacher_count = int(teacher_positive.sum())
    precision = true_positive / predicted_count if predicted_count > 0 else 0.0
    recall = true_positive / teacher_count if teacher_count > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    correlation = (
        v1_runner.rank_correlation(x, y)
        if fixed_rank_correlation is None
        else float(fixed_rank_correlation)
    )
    balanced_bce = _balanced_soft_bce_logits(
        logits, y,
        positive_weight=float(positive_weight),
        negative_weight=float(negative_weight),
    )
    balanced_brier = (
        float(positive_weight) * y * (1.0 - probability).square()
        + float(negative_weight) * (1.0 - y) * probability.square()
    ).mean()
    natural_bce = (F.softplus(logits) - y * logits).mean()
    return {
        "pairs": int(x.numel()),
        "rank_samples": formal.RANK_SAMPLE_CAP,
        "brier": float((probability - y).square().mean()),
        "soft_binary_cross_entropy": float(natural_bce),
        "mean_absolute_error": float((probability - y).abs().mean()),
        "balanced_brier": float(balanced_brier),
        "balanced_soft_binary_cross_entropy": float(balanced_bce),
        "rank_correlation": correlation,
        "teacher_positive_rate": float(teacher_positive.float().mean()),
        "predicted_positive_rate": float(predicted_positive.float().mean()),
        "teacher_positive_precision": precision,
        "teacher_positive_recall": recall,
        "teacher_positive_f1": f1,
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
        raise ValueError("class-balanced global margin exact4x2 cohort differs")
    return source


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber class-balanced calibration: {output}")
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
        label="class-balanced benchmark exclusion manifest",
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

    train_margin: list[torch.Tensor] = []
    train_target: list[torch.Tensor] = []
    for binding in source_inputs.train:
        margin, target = v1_runner.scene_margin_pairs(
            v1_runner.legacy.load_scene(binding), model=model, head=head,
            positive_text=positive, canonical_negative_text=negative,
            device=device, batch_rows=int(args.batch_rows),
        )
        train_margin.append(margin)
        train_target.append(target)
    fit_margin = torch.cat(train_margin)
    fit_target = torch.cat(train_target)
    weights = train_mass_weights(fit_target)
    parameters, optimization = fit_global_calibrator(fit_margin, fit_target, weights)
    fit_rank = v1_runner.rank_correlation(fit_margin, fit_target)
    metric_kwargs = {
        "positive_weight": float(weights["positive_weight"]),
        "negative_weight": float(weights["negative_weight"]),
        "fixed_rank_correlation": fit_rank,
    }
    fit_metrics = {
        "identity": calibration_metrics(
            fit_margin, fit_target, a=formal.IDENTITY_A, b=formal.IDENTITY_B,
            **metric_kwargs,
        ),
        "candidate": calibration_metrics(
            fit_margin, fit_target, a=parameters["a"], b=parameters["b"],
            **metric_kwargs,
        ),
    }
    del train_margin, train_target, fit_margin, fit_target

    validation: dict[str, Any] = {}
    for binding in source_inputs.validation:
        margin, target = v1_runner.scene_margin_pairs(
            v1_runner.legacy.load_scene(binding), model=model, head=head,
            positive_text=positive, canonical_negative_text=negative,
            device=device, batch_rows=int(args.batch_rows),
        )
        correlation = v1_runner.rank_correlation(margin, target)
        validation_kwargs = {
            "positive_weight": float(weights["positive_weight"]),
            "negative_weight": float(weights["negative_weight"]),
            "fixed_rank_correlation": correlation,
        }
        identity = calibration_metrics(
            margin, target, a=formal.IDENTITY_A, b=formal.IDENTITY_B,
            **validation_kwargs,
        )
        candidate = calibration_metrics(
            margin, target, a=parameters["a"], b=parameters["b"],
            **validation_kwargs,
        )
        validation[binding.scene_id] = {
            "identity": identity,
            "candidate": candidate,
            "checks": formal.expected_validation_checks(identity, candidate),
        }
    promotion_checks = {
        "optimization_converged": optimization["converged"] is True,
        "candidate_slope_strictly_positive": (
            parameters["a"] > formal.MINIMUM_POSITIVE_SLOPE
        ),
        "train_soft_mass_exactly_class_balanced": (
            abs(float(weights["weighted_positive_mass_fraction"]) - 0.5) <= 1e-10
            and abs(float(weights["weighted_negative_mass_fraction"]) - 0.5) <= 1e-10
        ),
        "every_validation_scene_passed": all(
            all(validation[scene]["checks"].values())
            for scene in formal.VALIDATION_SCENES
        ),
    }
    promoted = all(promotion_checks.values())
    result = {
        "schema": formal.RESULT_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "source_only_promoted" if promoted else "source_only_complete_no_promotion",
        "contract": formal.calibration_contract(),
        "contract_sha256": formal.CALIBRATION_CONTRACT_SHA256,
        "execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_contrast_v21_result": dict(execution["source_contrast_v21_result"]),
            "source_contrast_v21_checkpoint": dict(source_gate["result"]["checkpoint"]),
            "unweighted_v1_result": dict(execution["unweighted_v1_result"]),
            "fit_text_bank": dict(execution["fit_text_bank"]),
            "canonical_negative_bank": dict(execution["canonical_negative_bank"]),
            "benchmark_exclusion_manifest": dict(execution["benchmark_exclusion_manifest"]),
        },
        "parameters": {
            "identity": {"a": formal.IDENTITY_A, "b": formal.IDENTITY_B},
            "candidate": parameters,
        },
        "train_mass_weights": weights,
        "optimization_audit": optimization,
        "sampling_audit": {
            "train_scenes": list(formal.TRAIN_SCENES),
            "validation_scenes": list(formal.VALIDATION_SCENES),
            "regions_per_scene": formal.REGIONS_PER_SCENE,
            "query_rows": formal.FIT_QUERY_ROWS,
            "canonical_negative_rows": formal.CANONICAL_NEGATIVE_ROWS,
            "train_pairs": len(formal.TRAIN_SCENES) * formal.REGIONS_PER_SCENE * formal.FIT_QUERY_ROWS,
            "validation_pairs_per_scene": formal.REGIONS_PER_SCENE * formal.FIT_QUERY_ROWS,
            "validation_contribution_to_fit_or_weights": False,
            "all_rows_and_queries_used": True,
        },
        "fit_metrics": fit_metrics,
        "validation_metrics": validation,
        "promotion_checks": promotion_checks,
        "query_calibration_authorized": promoted,
        "source_access": formal.source_access(),
    }
    result = formal.validate_calibration_result(result)
    write_frozen_json(output, result)
    return {
        "status": result["status"],
        "parameters": result["parameters"],
        "train_mass_weights": result["train_mass_weights"],
        "optimization_audit": result["optimization_audit"],
        "fit_metrics": result["fit_metrics"],
        "validation_metrics": result["validation_metrics"],
        "promotion_checks": result["promotion_checks"],
        "query_calibration_authorized": result["query_calibration_authorized"],
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
        raise FileExistsError(f"refuses to clobber class-balanced authority: {output}")
    source_record = _bound_record(
        args.source_contrast_v21_result,
        args.source_contrast_v21_result_sha256,
        label="contrast V2.1 source result",
    )
    # Complete promoted source validation precedes every later artifact open.
    source_gate = formal.source_formal.validate_source_contrast_v21_result(source_record)
    prepared = contrast_v21.prepare_inputs(
        source_gate["result"]["execution_authority"]["path"],
        expected_sha256=source_gate["result"]["execution_authority"]["sha256"],
    )
    v1_result = _bound_record(
        args.unweighted_v1_result,
        args.unweighted_v1_result_sha256,
        label="unweighted V1 diagnostic result",
    )
    checked_v1 = formal._validate_unweighted_v1(v1_result)
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
    if checked_v1["input_authority"]["source_contrast_v21_result"] != source_record:
        raise ValueError("class-balanced V2 and unweighted V1 source differ")
    for name, value in (
        ("fit_text_bank", fit),
        ("canonical_negative_bank", negative),
        ("benchmark_exclusion_manifest", exclusion),
    ):
        if checked_v1["input_authority"][name] != value:
            raise ValueError(f"class-balanced V2 and unweighted V1 {name} differ")
    if prepared.base_v2.source.authority["benchmark_exclusion_manifest"] != exclusion:
        raise ValueError("class-balanced exclusion differs from exact4x2 source")
    v1_runner._load_fit_text_bank(fit)
    relevance_loss.load_frozen_canonical_negative_bank(
        negative["path"], expected_file_sha256=negative["sha256"]
    )
    exclusion_raw, _, _ = load_json_object(
        exclusion["path"], expected_sha256=exclusion["sha256"],
        label="class-balanced benchmark exclusion manifest",
    )
    base_trainer.validate_benchmark_exclusion_manifest(exclusion_raw)
    calibration_output = str(Path(args.calibration_output).expanduser().resolve())
    authority = {
        "schema": formal.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "authorized_source_only_class_balanced_global_margin_fit",
        "implementation": file_record(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in formal.IMPLEMENTATION_DEPENDENCIES.items()
        },
        "calibration_contract_sha256": formal.CALIBRATION_CONTRACT_SHA256,
        "source_contrast_v21_result": source_record,
        "unweighted_v1_result": v1_result,
        "fit_text_bank": fit,
        "canonical_negative_bank": negative,
        "benchmark_exclusion_manifest": exclusion,
        "calibration_output": calibration_output,
        "calibration_authorized": True,
        "target_execution_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "source_access": formal.source_access(),
    }
    write_frozen_json(output, authority)
    return {
        "status": "source_class_balanced_global_margin_authority_built",
        "authority": file_record(output),
        "calibration_output": calibration_output,
        "source_selected_step": source_gate["selected_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--source-contrast-v21-result", required=True)
    build.add_argument("--source-contrast-v21-result-sha256", required=True)
    build.add_argument("--unweighted-v1-result", required=True)
    build.add_argument("--unweighted-v1-result-sha256", required=True)
    build.add_argument("--fit-text-bank", required=True)
    build.add_argument("--fit-text-bank-sha256", required=True)
    build.add_argument("--canonical-negative-bank", required=True)
    build.add_argument("--canonical-negative-bank-sha256", required=True)
    build.add_argument("--benchmark-exclusion-manifest", required=True)
    build.add_argument("--benchmark-exclusion-manifest-sha256", required=True)
    build.add_argument("--calibration-output", required=True)
    build.add_argument("--authority-output", required=True)
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    run = commands.add_parser("calibrate")
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
            "status": "source_class_balanced_global_margin_authority_validated",
            "source_selected_step": authority["verified_source_gate"]["selected_step"],
            "target_or_query_opened": False,
        }
    else:
        result = calibrate(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "build_authority", "build_parser", "calibrate", "calibration_metrics",
    "fit_global_calibrator", "train_mass_weights",
]
