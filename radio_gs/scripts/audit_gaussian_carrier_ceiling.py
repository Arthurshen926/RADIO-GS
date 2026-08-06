#!/usr/bin/env python3
"""Measure the frozen RGB-Gaussian semantic carrier ceiling on LERF-OVS.

This is a label-oracle diagnostic, not a valid benchmark method.  It assigns
one continuous membership probability to every frozen Gaussian and optimizes
those probabilities against official masks through the exact, fixed
front-to-back contribution operator.  The same pass also measures how often a
single primitive contributes to both foreground and background evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.evaluation.gaussian_carrier_ceiling import (
    binary_membership_entropy,
    weighted_carrier_mixing_summary,
)
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    OPEN_GAUSSIAN_LERF_FRAMES,
    build_lerf_dataset_for_scene,
    build_mask_renderer,
)
from radio_gs.scripts.eval_lerf_grounding import (
    build_gt_masks,
    load_lerf_ovs_labels,
    load_render_pipeline,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json


AUDIT_CONTRACT = "frozen_rgb_gaussian_scalar_membership_ceiling_v1"


def _tensor_rows_sha256(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _grouped_csr(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    values: torch.Tensor,
    *,
    num_pixels: int,
    num_gaussians: int,
) -> torch.Tensor:
    gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
    pids = torch.as_tensor(pixel_ids).long().reshape(-1)
    weights = torch.as_tensor(values).float().reshape(-1)
    if gids.shape != pids.shape or gids.shape != weights.shape:
        raise ValueError("CSR hit fields must be aligned")
    if pids.numel() and not bool((pids[1:] >= pids[:-1]).all()):
        raise ValueError("exact contribution hits must be grouped by pixel")
    counts = torch.bincount(pids, minlength=int(num_pixels))
    crow = torch.cat([torch.zeros(1, dtype=torch.long, device=pids.device), counts.cumsum(0)])
    return torch.sparse_csr_tensor(
        crow,
        gids,
        weights,
        size=(int(num_pixels), int(num_gaussians)),
        device=weights.device,
        dtype=torch.float32,
    )


def _boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = np.asarray(mask).astype(np.uint8)
    if int(radius) <= 0:
        return np.zeros_like(binary, dtype=bool)
    size = 2 * int(radius) + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1) > 0
    eroded = cv2.erode(binary, kernel, iterations=1) > 0
    return dilated != eroded


def _capped_indices(mask: torch.Tensor, cap: int, generator: torch.Generator) -> torch.Tensor:
    indices = torch.nonzero(torch.as_tensor(mask).bool(), as_tuple=False).flatten()
    if int(cap) == 0:
        return indices[:0]
    if int(cap) < 0 or indices.numel() <= int(cap):
        return indices
    order = torch.randperm(indices.numel(), generator=generator)[: int(cap)]
    return indices[order]


def _sample_pixels(
    foreground: torch.Tensor,
    boundary: torch.Tensor,
    *,
    foreground_cap: int,
    boundary_cap: int,
    random_cap: int,
    seed: int,
) -> torch.Tensor:
    """Select one deterministic shared pixel set for all queries in a view."""

    fg = torch.as_tensor(foreground).bool()
    edge = torch.as_tensor(boundary).bool()
    if fg.shape != edge.shape or fg.ndim != 2:
        raise ValueError("foreground and boundary must be matching [P,Q]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    pieces = [
        _capped_indices(edge.any(dim=1), int(boundary_cap), generator),
        _capped_indices(fg.any(dim=1), int(foreground_cap), generator),
    ]
    if int(random_cap) > 0:
        pieces.append(torch.randperm(fg.shape[0], generator=generator)[: int(random_cap)])
    selected = torch.unique(torch.cat(pieces), sorted=True)
    if selected.numel() == 0:
        raise RuntimeError("oracle optimization pixel sample is empty")
    return selected


def _frame_targets(
    frame_objects: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    *,
    height: int,
    width: int,
    boundary_radius: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    active = sorted({str(obj["category"]) for obj in frame_objects})
    masks = build_gt_masks(list(frame_objects), list(categories), int(height), int(width))
    target = torch.from_numpy(
        np.stack([masks[name].reshape(-1) for name in active], axis=1)
    ).bool()
    boundary = torch.from_numpy(
        np.stack(
            [
                _boundary_band(masks[name], int(boundary_radius)).reshape(-1)
                for name in active
            ],
            axis=1,
        )
    ).bool()
    return active, target, boundary


def _oracle_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    *,
    dice_weight: float,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    score = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    truth = target.float()
    valid = available.bool()
    valid_f = valid.float()
    positives = (truth * valid_f).sum(dim=0)
    negatives = ((1.0 - truth) * valid_f).sum(dim=0)
    active = (positives > 0) & (negatives > 0)
    if not bool(active.any()):
        raise RuntimeError("no query has both foreground and background optimization pixels")
    if mode == "balanced_bce_dice":
        positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0)
        weights = torch.where(truth > 0.5, positive_weight[None], 1.0) * valid_f
    elif mode == "uniform_bce_dice":
        weights = valid_f
    else:
        raise ValueError(f"unsupported oracle loss mode: {mode}")
    pointwise = F.binary_cross_entropy(score, truth, reduction="none")
    bce = (pointwise * weights).sum() / weights.sum().clamp_min(1.0)
    intersection = (score * truth * valid_f).sum(dim=0)
    denominator = ((score + truth) * valid_f).sum(dim=0)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice = dice[active].mean()
    loss = bce + float(dice_weight) * dice
    return loss, {"bce": float(bce.detach()), "dice_loss": float(dice.detach())}


def _optimize_memberships(
    operator: torch.Tensor,
    initial_probability: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    dice_weight: float,
    loss_mode: str,
    log_every: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    initial = initial_probability.float().clamp(1e-4, 1.0 - 1e-4)
    logits = torch.nn.Parameter(torch.logit(initial))
    optimizer = torch.optim.Adam([logits], lr=float(learning_rate), weight_decay=0.0)
    history: list[dict[str, float | int]] = []
    for step in range(int(steps) + 1):
        probability = logits.sigmoid()
        prediction = torch.sparse.mm(operator, probability).clamp(0.0, 1.0)
        loss, pieces = _oracle_loss(
            prediction,
            target,
            available,
            dice_weight=float(dice_weight),
            mode=str(loss_mode),
        )
        if step % max(int(log_every), 1) == 0 or step == int(steps):
            row = {"step": int(step), "loss": float(loss.detach()), **pieces}
            history.append(row)
            print(
                f"[oracle-opt] step={step:03d} loss={row['loss']:.6f} "
                f"bce={row['bce']:.6f} dice={row['dice_loss']:.6f}",
                flush=True,
            )
        if step == int(steps):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return logits.detach().sigmoid(), history


def _empty_method_accumulator(num_queries: int, thresholds: torch.Tensor) -> dict[str, Any]:
    return {
        "curve_sum": torch.zeros(thresholds.numel(), dtype=torch.float64),
        "query_curve_sum": torch.zeros(num_queries, thresholds.numel(), dtype=torch.float64),
        "query_count": torch.zeros(num_queries, dtype=torch.long),
        "sample_count": 0,
        "fixed": [],
        "soft": [],
        "per_sample_oracle": [],
        "details": [],
    }


def _accumulate_metrics(
    accumulator: dict[str, Any],
    scores: torch.Tensor,
    target: torch.Tensor,
    *,
    thresholds: torch.Tensor,
    frame_id: int,
    active_names: Sequence[str],
    active_indices: Sequence[int],
) -> None:
    score = scores.float()
    truth = target.bool()
    curves = []
    for threshold in thresholds:
        prediction = score >= float(threshold)
        intersection = (prediction & truth).sum(dim=0).float()
        union = (prediction | truth).sum(dim=0).float()
        curves.append(torch.where(union > 0, intersection / union, torch.ones_like(union)))
    curve = torch.stack(curves).detach().cpu()
    fixed_prediction = score >= 0.5
    fixed_intersection = (fixed_prediction & truth).sum(dim=0).float()
    fixed_union = (fixed_prediction | truth).sum(dim=0).float()
    fixed = torch.where(
        fixed_union > 0,
        fixed_intersection / fixed_union,
        torch.ones_like(fixed_union),
    )
    truth_f = truth.float()
    soft_intersection = (score * truth_f).sum(dim=0)
    soft_union = score.sum(dim=0) + truth_f.sum(dim=0) - soft_intersection
    soft = torch.where(
        soft_union > 0,
        soft_intersection / soft_union.clamp_min(1e-8),
        torch.ones_like(soft_union),
    )
    accumulator["curve_sum"] += curve.double().sum(dim=1)
    accumulator["sample_count"] += len(active_indices)
    accumulator["fixed"].extend(float(value) for value in fixed.detach().cpu())
    accumulator["soft"].extend(float(value) for value in soft.detach().cpu())
    accumulator["per_sample_oracle"].extend(
        float(value) for value in curve.max(dim=0).values
    )
    for local, (name, query_index) in enumerate(zip(active_names, active_indices)):
        accumulator["query_curve_sum"][int(query_index)] += curve[:, local].double()
        accumulator["query_count"][int(query_index)] += 1
        best_index = int(curve[:, local].argmax())
        accumulator["details"].append(
            {
                "frame_id": int(frame_id),
                "category": str(name),
                "fixed_0p5_iou": float(fixed[local]),
                "soft_iou": float(soft[local]),
                "per_sample_oracle_iou": float(curve[best_index, local]),
                "per_sample_oracle_threshold": float(thresholds[best_index]),
                "gt_pixels": int(truth[:, local].sum()),
            }
        )


def _finalize_method(accumulator: dict[str, Any], thresholds: torch.Tensor) -> dict[str, Any]:
    count = int(accumulator["sample_count"])
    if count <= 0:
        raise RuntimeError("carrier evaluation produced zero benchmark samples")
    mean_curve = accumulator["curve_sum"] / count
    best_global = int(mean_curve.argmax())
    query_best: list[float] = []
    query_thresholds: list[float] = []
    for query, query_count in zip(
        accumulator["query_curve_sum"], accumulator["query_count"]
    ):
        if int(query_count) == 0:
            continue
        curve = query / int(query_count)
        best = int(curve.argmax())
        query_best.append(float(curve[best]))
        query_thresholds.append(float(thresholds[best]))
    return {
        "sample_count": count,
        "fixed_0p5_miou": float(np.mean(accumulator["fixed"])),
        "soft_miou": float(np.mean(accumulator["soft"])),
        "global_oracle_threshold_miou": float(mean_curve[best_global]),
        "global_oracle_threshold": float(thresholds[best_global]),
        "per_query_oracle_threshold_miou": float(np.mean(query_best)),
        "per_query_oracle_thresholds": query_thresholds,
        "per_sample_oracle_threshold_miou": float(
            np.mean(accumulator["per_sample_oracle"])
        ),
        "threshold_curve": [
            {"threshold": float(value), "miou": float(mean_curve[index])}
            for index, value in enumerate(thresholds)
        ],
        "details": accumulator["details"],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(args.allow_benchmark_mask_oracle):
        raise ValueError(
            "this diagnostic opens benchmark masks; pass "
            "--allow-benchmark-mask-oracle explicitly"
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("full-resolution exact carrier audit requires CUDA")
    torch.manual_seed(int(args.seed))
    annotations, categories, height, width = load_lerf_ovs_labels(
        args.label_dir, args.scene
    )
    if not bool(args.all_annotated_frames):
        official = set(OPEN_GAUSSIAN_LERF_FRAMES.get(args.scene, []))
        annotations = {
            frame: objects for frame, objects in annotations.items() if frame in official
        }
    if not annotations:
        raise RuntimeError("no annotated carrier-audit frames selected")

    model, codec, _renderer, sharpener, refiner, config, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    del codec, _renderer, sharpener, refiner
    gc.collect()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = build_lerf_dataset_for_scene(
        args.scene,
        config,
        args.label_dir,
        feature_height=height,
        feature_width=width,
    )
    missing = sorted(set(annotations) - set(dataset.pose_by_frame_idx))
    if missing:
        raise RuntimeError(f"missing frozen poses for official frames: {missing}")
    renderer = build_mask_renderer(
        config, height=height, width=width, device=device
    )
    num_gaussians = int(model.get_xyz().shape[0])
    num_queries = len(categories)
    num_pixels = int(height) * int(width)
    category_to_index = {name: index for index, name in enumerate(categories)}

    foreground_mass = torch.zeros(
        num_gaussians, num_queries, dtype=torch.float32, device=device
    )
    total_mass = torch.zeros_like(foreground_mass)
    boundary_mass = torch.zeros_like(foreground_mass)
    sampled_columns: list[torch.Tensor] = []
    sampled_values: list[torch.Tensor] = []
    sampled_counts: list[torch.Tensor] = []
    sampled_targets: list[torch.Tensor] = []
    sampled_available: list[torch.Tensor] = []
    sampled_frames: list[dict[str, Any]] = []

    print(
        f"[carrier-audit] stage1 exact adjoint + optimization sample: "
        f"scene={args.scene} rows={num_gaussians} queries={num_queries} "
        f"views={len(annotations)} resolution={width}x{height}",
        flush=True,
    )
    with torch.inference_mode():
        for frame_id, frame_objects in sorted(annotations.items()):
            active_names, target_cpu, boundary_cpu = _frame_targets(
                frame_objects,
                categories,
                height=height,
                width=width,
                boundary_radius=int(args.boundary_radius),
            )
            active_indices = [category_to_index[name] for name in active_names]
            pose = torch.from_numpy(dataset.pose_by_frame_idx[frame_id].copy()).float().to(device)
            hits = rasterize_single_view_contributions(
                model, renderer, pose, height=height, width=width
            )
            operator = _grouped_csr(
                hits["gaussian_ids"],
                hits["pixel_ids"],
                hits["weights"],
                num_pixels=num_pixels,
                num_gaussians=num_gaussians,
            )
            target = target_cpu.to(device=device, dtype=torch.float32)
            boundary = boundary_cpu.to(device=device, dtype=torch.float32)
            row_foreground = torch.sparse.mm(operator.transpose(0, 1), target)
            row_total = torch.sparse.mm(
                operator.transpose(0, 1),
                torch.ones(num_pixels, 1, dtype=torch.float32, device=device),
            )
            row_boundary = torch.sparse.mm(operator.transpose(0, 1), boundary)
            foreground_mass[:, active_indices] += row_foreground
            total_mass[:, active_indices] += row_total
            boundary_mass[:, active_indices] += row_boundary

            selected_cpu = _sample_pixels(
                target_cpu,
                boundary_cpu,
                foreground_cap=int(args.foreground_union_cap),
                boundary_cap=int(args.boundary_union_cap),
                random_cap=int(args.random_pixel_cap),
                seed=int(args.seed) + int(frame_id),
            )
            selected = selected_cpu.to(device)
            remap = torch.full(
                (num_pixels,), -1, dtype=torch.long, device=device
            )
            remap[selected] = torch.arange(selected.numel(), device=device)
            keep = remap[hits["pixel_ids"]] >= 0
            local_pixels = remap[hits["pixel_ids"][keep]]
            alpha = hits["accumulated_alpha"].reshape(-1)
            normalized_weight = hits["weights"][keep] / alpha[
                hits["pixel_ids"][keep]
            ].clamp_min(float(args.alpha_eps))
            sampled_columns.append(hits["gaussian_ids"][keep])
            sampled_values.append(normalized_weight)
            sampled_counts.append(
                torch.bincount(local_pixels, minlength=selected.numel())
            )
            frame_target = torch.zeros(
                selected.numel(), num_queries, dtype=torch.bool, device=device
            )
            frame_available = torch.zeros_like(frame_target)
            frame_target[:, active_indices] = target_cpu[selected_cpu].to(device)
            frame_available[:, active_indices] = True
            sampled_targets.append(frame_target)
            sampled_available.append(frame_available)
            sampled_frames.append(
                {
                    "frame_id": int(frame_id),
                    "active_queries": active_names,
                    "exact_hits": int(hits["weights"].numel()),
                    "sampled_pixels": int(selected.numel()),
                    "sampled_hits": int(keep.sum()),
                    "mean_alpha": float(alpha.mean()),
                }
            )
            print(
                f"[carrier-audit] frame={frame_id} hits={hits['weights'].numel()} "
                f"sample_pixels={selected.numel()} sample_hits={int(keep.sum())}",
                flush=True,
            )
            del (
                active_indices,
                target,
                boundary,
                operator,
                row_foreground,
                row_total,
                row_boundary,
                selected,
                remap,
                keep,
                local_pixels,
                normalized_weight,
                hits,
            )
            torch.cuda.empty_cache()

    counts = torch.cat(sampled_counts)
    crow = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)]
    )
    columns = torch.cat(sampled_columns)
    values = torch.cat(sampled_values).float()
    sample_target = torch.cat(sampled_targets)
    sample_available = torch.cat(sampled_available)
    sample_operator = torch.sparse_csr_tensor(
        crow,
        columns,
        values,
        size=(int(counts.numel()), num_gaussians),
        device=device,
        dtype=torch.float32,
    )
    del (
        sampled_counts,
        sampled_columns,
        sampled_values,
        sampled_targets,
        sampled_available,
        counts,
        crow,
        columns,
        values,
    )
    torch.cuda.empty_cache()

    initial_probability = torch.where(
        total_mass > float(args.mass_eps),
        foreground_mass / total_mass.clamp_min(float(args.mass_eps)),
        torch.zeros_like(total_mass),
    ).clamp(0.0, 1.0)
    optimized_probability, optimization_history = _optimize_memberships(
        sample_operator,
        initial_probability,
        sample_target,
        sample_available,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        dice_weight=float(args.dice_weight),
        loss_mode=str(args.loss_mode),
        log_every=int(args.log_every),
    )
    del sample_operator, sample_target, sample_available
    torch.cuda.empty_cache()

    thresholds = torch.arange(
        float(args.threshold_min),
        float(args.threshold_max) + 0.5 * float(args.threshold_step),
        float(args.threshold_step),
        dtype=torch.float32,
    )
    probabilities = {
        "exact_adjoint_ratio": initial_probability,
        "optimized_scalar_membership": optimized_probability,
    }
    accumulators = {
        name: _empty_method_accumulator(num_queries, thresholds)
        for name in probabilities
    }
    evaluation_frames: list[dict[str, Any]] = []
    print("[carrier-audit] stage2 full-resolution exact evaluation", flush=True)
    with torch.inference_mode():
        for frame_id, frame_objects in sorted(annotations.items()):
            active_names, target_cpu, _boundary_cpu = _frame_targets(
                frame_objects,
                categories,
                height=height,
                width=width,
                boundary_radius=int(args.boundary_radius),
            )
            active_indices = [category_to_index[name] for name in active_names]
            pose = torch.from_numpy(dataset.pose_by_frame_idx[frame_id].copy()).float().to(device)
            hits = rasterize_single_view_contributions(
                model, renderer, pose, height=height, width=width
            )
            alpha = hits["accumulated_alpha"].reshape(-1)
            normalized = hits["weights"] / alpha[hits["pixel_ids"]].clamp_min(
                float(args.alpha_eps)
            )
            operator = _grouped_csr(
                hits["gaussian_ids"],
                hits["pixel_ids"],
                normalized,
                num_pixels=num_pixels,
                num_gaussians=num_gaussians,
            )
            target = target_cpu.to(device)
            frame_methods: dict[str, Any] = {}
            for name, probability in probabilities.items():
                scores = torch.sparse.mm(operator, probability[:, active_indices]).clamp(0.0, 1.0)
                _accumulate_metrics(
                    accumulators[name],
                    scores,
                    target,
                    thresholds=thresholds.to(device),
                    frame_id=int(frame_id),
                    active_names=active_names,
                    active_indices=active_indices,
                )
                prediction = scores >= 0.5
                intersection = (prediction & target).sum(dim=0).float()
                union = (prediction | target).sum(dim=0).float()
                frame_methods[name] = {
                    "fixed_0p5_miou": float(
                        torch.where(union > 0, intersection / union, torch.ones_like(union)).mean()
                    )
                }
            evaluation_frames.append(
                {
                    "frame_id": int(frame_id),
                    "active_queries": active_names,
                    "exact_hits": int(hits["weights"].numel()),
                    "methods": frame_methods,
                }
            )
            print(
                f"[carrier-audit] evaluated frame={frame_id} "
                + " ".join(
                    f"{name}={values['fixed_0p5_miou']:.4f}"
                    for name, values in frame_methods.items()
                ),
                flush=True,
            )
            del hits, alpha, normalized, operator, target, scores, prediction
            torch.cuda.empty_cache()

    methods = {
        name: _finalize_method(accumulator, thresholds)
        for name, accumulator in accumulators.items()
    }
    scales = model.get_scaling().detach().float().mean(dim=1).cpu()
    mixing: dict[str, Any] = {}
    boundary_ambiguous_fractions: list[float] = []
    for query_index, category in enumerate(categories):
        fg = foreground_mass[:, query_index].detach().cpu()
        total = total_mass[:, query_index].detach().cpu()
        membership, entropy = binary_membership_entropy(
            fg, total, eps=float(args.mass_eps)
        )
        summary = weighted_carrier_mixing_summary(
            fg,
            total,
            ambiguity_low=float(args.ambiguity_low),
            ambiguity_high=float(args.ambiguity_high),
            eps=float(args.mass_eps),
        )
        observed = total > float(args.mass_eps)
        ambiguous = observed & (membership > float(args.ambiguity_low)) & (
            membership < float(args.ambiguity_high)
        )
        pure_foreground = observed & (membership >= float(args.ambiguity_high))
        edge = boundary_mass[:, query_index].detach().cpu()
        edge_fraction = float(
            edge[ambiguous].sum() / edge.sum().clamp_min(float(args.mass_eps))
        )
        boundary_ambiguous_fractions.append(edge_fraction)
        summary.update(
            {
                "boundary_mass_carried_by_ambiguous_rows": edge_fraction,
                "mean_scale_ambiguous": (
                    float(scales[ambiguous].mean()) if bool(ambiguous.any()) else None
                ),
                "mean_scale_pure_foreground": (
                    float(scales[pure_foreground].mean())
                    if bool(pure_foreground.any())
                    else None
                ),
                "observed_entropy_p90": (
                    float(torch.quantile(entropy[observed], 0.9))
                    if bool(observed.any())
                    else None
                ),
            }
        )
        mixing[category] = summary

    feasible_scores = {
        name: float(values["global_oracle_threshold_miou"])
        for name, values in methods.items()
    }
    best_feasible_method = max(feasible_scores, key=feasible_scores.get)
    feasible_lower_bound = feasible_scores[best_feasible_method]
    optimized_score = feasible_scores["optimized_scalar_membership"]
    initial_score = feasible_scores["exact_adjoint_ratio"]
    reference = float(args.reference_miou)
    gate: dict[str, Any] = {
        "metric_semantics": (
            "best constructed feasible scalar membership is a rigorous lower bound on "
            "the unknown carrier optimum"
        ),
        "best_feasible_method": best_feasible_method,
        "feasible_lower_bound_miou": feasible_lower_bound,
        "optimization_delta_from_adjoint": optimized_score - initial_score,
        "optimization_regressed": optimized_score < initial_score,
        "reference_miou": reference if reference > 0 else None,
        "hard_limit_gap_points": float(args.hard_limit_gap_points),
        "decision": "reference_not_supplied",
    }
    if reference > 0:
        margin_points = 100.0 * (feasible_lower_bound - reference)
        gate["feasible_margin_over_reference_points"] = margin_points
        gate["decision"] = (
            "carrier_not_primary_on_this_task"
            if margin_points >= -float(args.hard_limit_gap_points)
            else "inconclusive_requires_tighter_oracle_optimization"
        )

    label_records = [
        file_record(Path(args.label_dir) / args.scene / f"frame_{frame:05d}.json")
        for frame in sorted(annotations)
    ]
    report = {
        "schema_version": 1,
        "audit": AUDIT_CONTRACT,
        "protocol": {
            "diagnostic_only": True,
            "valid_benchmark_method": False,
            "benchmark_masks_opened": True,
            "text_queries_opened": False,
            "geometry_frozen": True,
            "one_scalar_membership_per_gaussian_and_query": True,
            "observation_operator": (
                "exact front-to-back Gaussian contribution, divided by frozen total alpha"
            ),
            "optimization_pixel_sampling_is_label_oracle": True,
            "full_resolution_evaluation": True,
            "thresholds_are_diagnostic_only": True,
        },
        "scene": args.scene,
        "categories": categories,
        "frame_ids": sorted(annotations),
        "resolution": {"height": int(height), "width": int(width)},
        "num_gaussians": num_gaussians,
        "geometry_xyz_sha256": _tensor_rows_sha256(model.get_xyz()),
        "artifacts": {
            "config": file_record(args.config),
            "geometry_checkpoint": file_record(args.geometry_checkpoint),
            "labels": label_records,
            "preregistration": file_record(args.preregistration),
        },
        "optimization": {
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "dice_weight": float(args.dice_weight),
            "loss_mode": str(args.loss_mode),
            "foreground_union_cap_per_view": int(args.foreground_union_cap),
            "boundary_union_cap_per_view": int(args.boundary_union_cap),
            "random_pixel_cap_per_view": int(args.random_pixel_cap),
            "sampled_rows": int(sum(row["sampled_pixels"] for row in sampled_frames)),
            "sampled_hits": int(sum(row["sampled_hits"] for row in sampled_frames)),
            "history": optimization_history,
        },
        "methods": methods,
        "mixing": {
            "ambiguity_interval": [float(args.ambiguity_low), float(args.ambiguity_high)],
            "category_mean_boundary_mass_carried_by_ambiguous_rows": float(
                np.mean(boundary_ambiguous_fractions)
            ),
            "per_category": mixing,
        },
        "sampled_frames": sampled_frames,
        "evaluation_frames": evaluation_frames,
        "decision_gate": gate,
        "runtime": {
            "device": str(device),
            "maximum_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    write_frozen_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-benchmark-mask-oracle", action="store_true")
    parser.add_argument("--all-annotated-frames", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument(
        "--loss-mode",
        choices=["balanced_bce_dice", "uniform_bce_dice"],
        default="balanced_bce_dice",
    )
    parser.add_argument("--foreground-union-cap", type=int, default=49152)
    parser.add_argument("--boundary-union-cap", type=int, default=49152)
    parser.add_argument("--random-pixel-cap", type=int, default=65536)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--ambiguity-low", type=float, default=0.1)
    parser.add_argument("--ambiguity-high", type=float, default=0.9)
    parser.add_argument("--alpha-eps", type=float, default=1e-8)
    parser.add_argument("--mass-eps", type=float, default=1e-10)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--reference-miou", type=float, default=0.0)
    parser.add_argument("--hard-limit-gap-points", type=float, default=3.0)
    args = parser.parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "methods": {
                    name: {
                        key: values[key]
                        for key in (
                            "fixed_0p5_miou",
                            "global_oracle_threshold_miou",
                            "global_oracle_threshold",
                            "per_sample_oracle_threshold_miou",
                        )
                    }
                    for name, values in report["methods"].items()
                },
                "mixing": {
                    "category_mean_boundary_mass_carried_by_ambiguous_rows": report[
                        "mixing"
                    ]["category_mean_boundary_mass_carried_by_ambiguous_rows"]
                },
                "decision_gate": report["decision_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
