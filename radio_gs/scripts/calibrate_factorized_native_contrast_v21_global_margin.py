#!/usr/bin/env python3
"""Fit one source-only global absolute-margin calibrator for contrast V2.1."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces import factorized_native_source_global_margin_calibration as formal
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance_loss
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import train_factorized_native_gauge_state_readout_exact4x2 as legacy
from radio_gs.scripts import train_factorized_native_gauge_state_readout_exact4x2_contrast_v21 as contrast_v21
from radio_gs.scripts import train_surface_region_full_scalar_residual as base_trainer
from radio_gs.scripts import train_surface_region_typed_context_residual as sparse_teacher
from radio_gs.scripts import train_surface_region_typed_context_response_listwise_v2 as text_loader
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


DESCRIPTOR_BATCH_ROWS = 128


def _load_fit_text_bank(record: Mapping[str, str]) -> text_loader.FitTextBank:
    bank = text_loader.load_fit_text_bank(
        record["path"], expected_sha256=record["sha256"]
    )
    raw, digest, source = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="global margin target-blind fit text bank",
    )
    queries = raw.get("queries")
    records = raw.get("records")
    encoder = raw.get("text_encoder")
    embeddings = bank.embeddings
    if (
        bank.record != {"path": str(source), "sha256": digest}
        or raw.get("schema_version") != 1
        or raw.get("artifact_type") != "target_blind_text_embedding_cache"
        or raw.get("algorithm_version") != "siglip2-target-blind-split-v1"
        or raw.get("split") != "fit"
        or raw.get("benchmark_vocabulary_opened") is not False
        or raw.get("uses_benchmark_vocabulary_for_construction") is not False
        or raw.get("prompt_templates") != ["{query}"]
        or raw.get("text_canonicalization") != "official_c_radio_siglip2_g"
        or not isinstance(queries, list)
        or len(queries) != formal.FIT_QUERY_ROWS
        or len(set(queries)) != formal.FIT_QUERY_ROWS
        or not isinstance(records, list)
        or len(records) != formal.FIT_QUERY_ROWS
        or any(
            row
            != {"synset": raw["synsets"][index], "query": queries[index], "split": "fit"}
            for index, row in enumerate(records)
        )
        or not isinstance(encoder, Mapping)
        or encoder.get("model_id") != relevance_loss.CANONICAL_NEGATIVE_MODEL
        or encoder.get("output_dimension") != 1536
        or encoder.get("dtype") != "float32"
        or encoder.get("normalization") != "l2"
        or embeddings.shape != (formal.FIT_QUERY_ROWS, 1536)
        or not torch.allclose(
            torch.linalg.vector_norm(embeddings, dim=-1),
            torch.ones(formal.FIT_QUERY_ROWS),
            rtol=0.0,
            atol=2e-4,
        )
    ):
        raise ValueError("global margin fit text bank contract differs")
    return bank


def deterministic_rank_axis(pairs: int) -> torch.Tensor:
    count = int(pairs)
    if count < formal.RANK_SAMPLE_CAP:
        raise ValueError("global margin rank axis requires the frozen sample cap")
    axis = torch.div(
        torch.arange(formal.RANK_SAMPLE_CAP, dtype=torch.long) * count,
        formal.RANK_SAMPLE_CAP,
        rounding_mode="floor",
    )
    if axis.unique().numel() != formal.RANK_SAMPLE_CAP:
        raise RuntimeError("global margin rank axis repeats a pair")
    return axis


def soft_teacher_probability(
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
) -> torch.Tensor:
    if (
        teacher_views.ndim != 3
        or teacher_mask.shape != teacher_views.shape[:2]
        or teacher_mask.dtype != torch.bool
        or positive_text.ndim != 2
        or canonical_negative_text.shape != (formal.CANONICAL_NEGATIVE_ROWS, 1536)
        or teacher_views.shape[-1] != 1536
        or positive_text.shape[-1] != 1536
        or not bool(teacher_mask.any(dim=1).all())
    ):
        raise ValueError("global margin teacher probability layout differs")
    positive = torch.einsum("bvd,qd->bvq", teacher_views, positive_text)
    negative = torch.einsum(
        "bvd,kd->bvk", teacher_views, canonical_negative_text
    ).amax(dim=-1)
    probability = torch.sigmoid(
        formal.TEACHER_LOGIT_SCALE * (positive - negative[..., None])
    )
    mask = teacher_mask[..., None]
    return (probability * mask).sum(dim=1) / mask.sum(dim=1)


@torch.inference_mode()
def scene_margin_pairs(
    scene: legacy.LoadedScene,
    *,
    model: torch.nn.Module,
    head: torch.nn.Module,
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
    device: torch.device,
    batch_rows: int = DESCRIPTOR_BATCH_ROWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = legacy._scene_rows(scene)
    if (
        rows.numel() != formal.REGIONS_PER_SCENE
        or positive_text.shape != (formal.FIT_QUERY_ROWS, 1536)
        or canonical_negative_text.shape != (formal.CANONICAL_NEGATIVE_ROWS, 1536)
        or int(batch_rows) <= 0
    ):
        raise ValueError("global margin scene axes differ")
    positive = positive_text.to(device)
    negative = canonical_negative_text.to(device)
    margins: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    head.eval()
    for start in range(0, rows.numel(), int(batch_rows)):
        selected = rows[start : start + int(batch_rows)]
        student = legacy._descriptor_batch(model, head, scene, selected, device)
        student_positive = student @ positive.T
        student_negative = (student @ negative.T).amax(dim=-1)
        margin = student_positive - student_negative[:, None]
        teacher, mask = sparse_teacher.gather_sparse_teacher_batch(
            scene.shard, selected
        )
        target = soft_teacher_probability(
            F.normalize(teacher.to(device).float(), dim=-1),
            mask.to(device),
            positive,
            negative,
        )
        if (
            margin.shape != target.shape
            or not bool(torch.isfinite(margin).all())
            or not bool(torch.isfinite(target).all())
            or bool((target < 0).any())
            or bool((target > 1).any())
        ):
            raise ValueError("global margin source pair values differ")
        margins.append(margin.float().cpu())
        targets.append(target.float().cpu())
    return torch.cat(margins).flatten().contiguous(), torch.cat(targets).flatten().contiguous()


def _soft_bce_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (F.softplus(logits) - target * logits).mean()


def fit_global_calibrator(
    margin: torch.Tensor, target: torch.Tensor
) -> tuple[dict[str, float], dict[str, Any]]:
    x = torch.as_tensor(margin).detach().double().cpu().contiguous()
    y = torch.as_tensor(target).detach().double().cpu().contiguous()
    if (
        x.ndim != 1
        or y.shape != x.shape
        or x.numel() != len(formal.TRAIN_SCENES) * formal.REGIONS_PER_SCENE * formal.FIT_QUERY_ROWS
        or not bool(torch.isfinite(x).all())
        or not bool(torch.isfinite(y).all())
        or bool((y < 0).any())
        or bool((y > 1).any())
        or float(x.std()) <= 1e-8
    ):
        raise ValueError("global margin fit samples differ")
    theta = torch.tensor([formal.IDENTITY_A, formal.IDENTITY_B], dtype=torch.float64)
    initial = float(_soft_bce_logits(theta[0] * x + theta[1], y))
    converged = False
    iterations = 0
    reductions = 0
    final_gradient = math.inf
    minimum_hessian_eigenvalue = math.inf
    for iteration in range(1, formal.MAXIMUM_NEWTON_ITERATIONS + 1):
        logits = theta[0] * x + theta[1]
        probability = torch.sigmoid(logits)
        residual = probability - y
        weight = probability * (1.0 - probability)
        gradient = torch.stack(((residual * x).mean(), residual.mean()))
        hessian = torch.stack(
            (
                torch.stack(((weight * x.square()).mean(), (weight * x).mean())),
                torch.stack(((weight * x).mean(), weight.mean())),
            )
        )
        eigenvalue = float(torch.linalg.eigvalsh(hessian).min())
        minimum_hessian_eigenvalue = min(minimum_hessian_eigenvalue, eigenvalue)
        if not math.isfinite(eigenvalue) or eigenvalue <= 1e-12:
            raise RuntimeError("global margin calibration Hessian is not positive definite")
        final_gradient = float(gradient.abs().max())
        iterations = iteration
        if final_gradient <= formal.OPTIMIZATION_GRADIENT_TOLERANCE:
            converged = True
            break
        step = torch.linalg.solve(hessian, gradient)
        current = float(_soft_bce_logits(logits, y))
        accepted = False
        scale = 1.0
        for _ in range(formal.MAXIMUM_BACKTRACK_STEPS):
            candidate = theta - scale * step
            if float(candidate[0]) > formal.MINIMUM_POSITIVE_SLOPE:
                objective = float(
                    _soft_bce_logits(candidate[0] * x + candidate[1], y)
                )
                if math.isfinite(objective) and objective < current:
                    theta = candidate
                    accepted = True
                    break
            scale *= 0.5
            reductions += 1
        if not accepted:
            raise RuntimeError("global margin Newton backtracking failed")
    final = float(_soft_bce_logits(theta[0] * x + theta[1], y))
    if (
        not converged
        or float(theta[0]) <= formal.MINIMUM_POSITIVE_SLOPE
        or not math.isfinite(final)
        or final >= initial
    ):
        raise RuntimeError("global margin calibration did not reach the unique optimum")
    return (
        {"a": float(theta[0]), "b": float(theta[1])},
        {
            "solver": "deterministic_strictly_convex_two_parameter_newton_irls",
            "iterations": iterations,
            "converged": converged,
            "initial_soft_binary_cross_entropy": initial,
            "final_soft_binary_cross_entropy": final,
            "final_max_absolute_gradient": final_gradient,
            "minimum_observed_hessian_eigenvalue": minimum_hessian_eigenvalue,
            "backtrack_reductions": reductions,
        },
    )


def _ordinal_rank(value: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(value).detach().float().cpu().contiguous()
    order = torch.argsort(vector, stable=True)
    rank = torch.empty(vector.numel(), dtype=torch.float64)
    rank[order] = torch.arange(vector.numel(), dtype=torch.float64)
    return rank


def rank_correlation(margin: torch.Tensor, target: torch.Tensor) -> float:
    axis = deterministic_rank_axis(int(margin.numel()))
    left = _ordinal_rank(margin[axis])
    right = _ordinal_rank(target[axis])
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0:
        raise ValueError("global margin rank correlation has zero variance")
    return float((left * right).sum() / denominator)


def calibration_metrics(
    margin: torch.Tensor,
    target: torch.Tensor,
    *,
    a: float,
    b: float,
    fixed_rank_correlation: float | None = None,
) -> dict[str, float | int]:
    x = torch.as_tensor(margin).detach().float().cpu().contiguous()
    y = torch.as_tensor(target).detach().float().cpu().contiguous()
    if x.ndim != 1 or y.shape != x.shape or not math.isfinite(float(a)) or not math.isfinite(float(b)) or float(a) <= 0:
        raise ValueError("global margin metric inputs differ")
    logits = float(a) * x + float(b)
    probability = torch.sigmoid(logits)
    teacher_positive = y >= 0.5
    predicted_positive = probability >= 0.5
    true_positive = int((teacher_positive & predicted_positive).sum())
    predicted_count = int(predicted_positive.sum())
    teacher_count = int(teacher_positive.sum())
    if predicted_count <= 0 or teacher_count <= 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = true_positive / predicted_count
        recall = true_positive / teacher_count
    correlation = (
        rank_correlation(x, y)
        if fixed_rank_correlation is None
        else float(fixed_rank_correlation)
    )
    return {
        "pairs": int(x.numel()),
        "rank_samples": formal.RANK_SAMPLE_CAP,
        "brier": float((probability - y).square().mean()),
        "soft_binary_cross_entropy": float(_soft_bce_logits(logits, y)),
        "mean_absolute_error": float((probability - y).abs().mean()),
        "rank_correlation": correlation,
        "teacher_positive_rate": float(teacher_positive.float().mean()),
        "predicted_positive_rate": float(predicted_positive.float().mean()),
        "teacher_positive_precision": precision,
        "teacher_positive_recall": recall,
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
        or tuple(item.scene_id for item in source.validation)
        != formal.VALIDATION_SCENES
        or source.authority["benchmark_exclusion_manifest"]
        != execution["benchmark_exclusion_manifest"]
    ):
        raise ValueError("global margin exact4x2 source cohort differs")
    return source


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber global margin calibration: {output}")
    execution = formal.validate_execution_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    source_inputs = _prepared_source(execution)
    fit_bank = _load_fit_text_bank(execution["fit_text_bank"])
    negative_bank = relevance_loss.load_frozen_canonical_negative_bank(
        execution["canonical_negative_bank"]["path"],
        expected_file_sha256=execution["canonical_negative_bank"]["sha256"],
    )
    exclusion_raw, _, _ = load_json_object(
        execution["benchmark_exclusion_manifest"]["path"],
        expected_sha256=execution["benchmark_exclusion_manifest"]["sha256"],
        label="global margin benchmark exclusion manifest",
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
        margin, target = scene_margin_pairs(
            legacy.load_scene(binding), model=model, head=head,
            positive_text=positive, canonical_negative_text=negative,
            device=device, batch_rows=int(args.batch_rows),
        )
        train_margin.append(margin)
        train_target.append(target)
    fit_margin = torch.cat(train_margin)
    fit_target = torch.cat(train_target)
    parameters, optimization = fit_global_calibrator(fit_margin, fit_target)
    fit_rank = rank_correlation(fit_margin, fit_target)
    fit_metrics = {
        "identity": calibration_metrics(
            fit_margin, fit_target, a=formal.IDENTITY_A, b=formal.IDENTITY_B,
            fixed_rank_correlation=fit_rank,
        ),
        "candidate": calibration_metrics(
            fit_margin, fit_target, a=parameters["a"], b=parameters["b"],
            fixed_rank_correlation=fit_rank,
        ),
    }
    del train_margin, train_target, fit_margin, fit_target

    validation: dict[str, Any] = {}
    for binding in source_inputs.validation:
        margin, target = scene_margin_pairs(
            legacy.load_scene(binding), model=model, head=head,
            positive_text=positive, canonical_negative_text=negative,
            device=device, batch_rows=int(args.batch_rows),
        )
        correlation = rank_correlation(margin, target)
        identity = calibration_metrics(
            margin, target, a=formal.IDENTITY_A, b=formal.IDENTITY_B,
            fixed_rank_correlation=correlation,
        )
        candidate = calibration_metrics(
            margin, target, a=parameters["a"], b=parameters["b"],
            fixed_rank_correlation=correlation,
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
            "fit_text_bank": dict(execution["fit_text_bank"]),
            "canonical_negative_bank": dict(execution["canonical_negative_bank"]),
            "benchmark_exclusion_manifest": dict(execution["benchmark_exclusion_manifest"]),
        },
        "parameters": {
            "identity": {"a": formal.IDENTITY_A, "b": formal.IDENTITY_B},
            "candidate": parameters,
        },
        "optimization_audit": optimization,
        "sampling_audit": {
            "train_scenes": list(formal.TRAIN_SCENES),
            "validation_scenes": list(formal.VALIDATION_SCENES),
            "regions_per_scene": formal.REGIONS_PER_SCENE,
            "query_rows": formal.FIT_QUERY_ROWS,
            "canonical_negative_rows": formal.CANONICAL_NEGATIVE_ROWS,
            "train_pairs": len(formal.TRAIN_SCENES) * formal.REGIONS_PER_SCENE * formal.FIT_QUERY_ROWS,
            "validation_pairs_per_scene": formal.REGIONS_PER_SCENE * formal.FIT_QUERY_ROWS,
            "validation_contribution_to_fit": False,
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
        raise FileExistsError(f"refuses to clobber global margin authority: {output}")
    source_record = _bound_record(
        args.source_contrast_v21_result,
        args.source_contrast_v21_result_sha256,
        label="contrast V2.1 source result",
    )
    # Fail closed on the complete source promotion before opening text banks.
    source_gate = formal.source_formal.validate_source_contrast_v21_result(source_record)
    prepared = contrast_v21.prepare_inputs(
        source_gate["result"]["execution_authority"]["path"],
        expected_sha256=source_gate["result"]["execution_authority"]["sha256"],
    )
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
    if prepared.base_v2.source.authority["benchmark_exclusion_manifest"] != exclusion:
        raise ValueError("global margin exclusion manifest differs from exact4x2 source")
    _load_fit_text_bank(fit)
    relevance_loss.load_frozen_canonical_negative_bank(
        negative["path"], expected_file_sha256=negative["sha256"]
    )
    exclusion_raw, _, _ = load_json_object(
        exclusion["path"], expected_sha256=exclusion["sha256"],
        label="global margin benchmark exclusion manifest",
    )
    base_trainer.validate_benchmark_exclusion_manifest(exclusion_raw)
    calibration_output = str(Path(args.calibration_output).expanduser().resolve())
    authority = {
        "schema": formal.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "authorized_source_only_global_margin_fit_4train_2validation",
        "implementation": file_record(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path) for name, path in formal.IMPLEMENTATION_DEPENDENCIES.items()
        },
        "calibration_contract_sha256": formal.CALIBRATION_CONTRACT_SHA256,
        "source_contrast_v21_result": source_record,
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
        "status": "source_global_margin_calibration_authority_built",
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
            "status": "source_global_margin_calibration_authority_validated",
            "source_selected_step": authority["verified_source_gate"]["selected_step"],
            "target_or_query_opened": False,
        }
    else:
        result = calibrate(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "build_authority",
    "build_parser",
    "calibrate",
    "calibration_metrics",
    "deterministic_rank_axis",
    "fit_global_calibrator",
    "rank_correlation",
    "scene_margin_pairs",
    "soft_teacher_probability",
]
