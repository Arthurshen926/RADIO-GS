#!/usr/bin/env python3
"""Development-only D0--D5 support/readout oracle on LERF Figurines."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts.audit_gaussian_carrier_ceiling import (
    _frame_targets,
    _grouped_csr,
)
from radio_gs.scripts.eval_lerf_grounding import (
    build_gt_masks,
    load_lerf_ovs_labels,
    load_render_pipeline,
)
from radio_gs.scripts.eval_lerf_teacher_view_oracle_diagnostic import (
    _binary_midrank_correlation,
    _grouped_average_precision,
    _oracle_iou,
)
from radio_gs.interfaces.surface_region_selection import (
    surface_region_contract_from_metadata,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.rendering.contribution_compositor import (
    iter_single_view_contribution_chunks,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_support_readout_oracle_d0_d5.v2"
TEACHER_SCHEMA = "radio_gs.lerf_source_teacher_view_siglip_authority.v1"
FIXED_THRESHOLD = 0.6
POOL_SIZE = 1024
MAX_UNION_REGIONS = 8
TARGET_ADJOINT_HIT_CHUNK = 262_144


def _require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or sha256_file(source) != str(expected_sha256):
        raise ValueError(f"{label} is missing or has a different SHA-256")
    return source


def _binary_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.as_tensor(prediction).bool()
    truth = torch.as_tensor(target).bool()
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    return float(intersection / union) if union else 1.0


def _primitive_query_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    available: torch.Tensor,
    *,
    include_ranking_metrics: bool = True,
) -> dict[str, Any]:
    value = torch.as_tensor(scores).float().cpu()
    truth = torch.as_tensor(target).bool().cpu()
    seen = torch.as_tensor(observed).bool().cpu()
    permitted = torch.as_tensor(available).bool().cpu()
    if value.shape != truth.shape or truth.shape != seen.shape:
        raise ValueError("primitive score, target and observation axes differ")
    if permitted.ndim != 1 or permitted.shape[0] != value.shape[0]:
        raise ValueError("primitive availability axis differs")
    rows = []
    for query in range(value.shape[1]):
        mask = seen[:, query] & permitted
        if not bool(mask.any()) or not bool(truth[mask, query].any()):
            continue
        query_score = value[mask, query].numpy().astype(np.float64)
        query_target = truth[mask, query].numpy()
        if bool(np.logical_or(query_score == 0.0, query_score == 1.0).all()):
            one_iou = _binary_iou(query_score >= 1.0, query_target)
            all_iou = _binary_iou(np.ones_like(query_target), query_target)
            if one_iou >= all_iou:
                oracle_iou, oracle_threshold = one_iou, 1.0
            else:
                oracle_iou, oracle_threshold = all_iou, 0.0
        else:
            oracle_iou, oracle_threshold = _oracle_iou(query_score, query_target)
        fixed_prediction = query_score >= FIXED_THRESHOLD
        row = {
                "query_index": int(query),
                "observed_rows": int(mask.sum()),
                "positive_rows": int(truth[mask, query].sum()),
                "selected_rows": int(fixed_prediction.sum()),
                "fixed_iou": _binary_iou(fixed_prediction, query_target),
                "oracle_threshold_iou": oracle_iou,
                "oracle_threshold": oracle_threshold,
            }
        if include_ranking_metrics:
            row["average_precision"] = _grouped_average_precision(
                query_score, query_target
            )
        rows.append(row)
    if not rows:
        raise RuntimeError("primitive diagnostic produced no evaluable queries")
    keys = ["fixed_iou", "oracle_threshold_iou"]
    if include_ranking_metrics:
        keys.append("average_precision")
    return {
        "query_count": len(rows),
        "aggregate_query_mean": {
            key: float(np.mean([float(row[key]) for row in rows])) for key in keys
        },
        "queries": rows,
    }


def _d4_score_variants(
    query_scores: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return every preregistered fixed D4 scale treatment.

    The maximum is applied independently per primitive/query and never uses a
    target, a label, or a scene-level statistic.  Unavailable teacher rows
    remain exact zero for every branch.
    """

    values = torch.as_tensor(query_scores).detach().float().cpu()
    available = torch.as_tensor(teacher_valid).bool().cpu().reshape(-1)
    if values.ndim != 3 or values.shape[1] != 3:
        raise ValueError("D4 requires exactly three registered scale slots")
    if available.shape != (values.shape[0],):
        raise ValueError("D4 teacher availability axis differs")
    raw = {
        "D4_s0": values[:, 0],
        "D4_s1": values[:, 1],
        "D4_s2": values[:, 2],
        "D4_per_primitive_max": values.amax(dim=1),
    }
    result = {}
    for name, score in raw.items():
        corrected = torch.zeros_like(score)
        corrected[available] = score[available]
        result[name] = corrected
    return result


def _render_membership_diagnostic(
    *,
    name: str,
    membership: torch.Tensor,
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[str],
    height: int,
    width: int,
    model: torch.nn.Module,
    renderer: Any,
    dataset: Any,
    device: torch.device,
    include_ranking_metrics: bool = True,
    include_oracle_threshold: bool = True,
) -> dict[str, Any]:
    values = torch.as_tensor(membership).float()
    if values.shape != (int(model.get_xyz().shape[0]), len(categories)):
        raise ValueError(f"{name} membership shape differs")
    proxy = frozen.GaussianSelectionProxy(model, values.to(device))
    rows: list[dict[str, Any]] = []
    category_to_index = {value: index for index, value in enumerate(categories)}
    for frame_id, objects in sorted(annotations.items()):
        pose = dataset.pose_by_frame_idx.get(int(frame_id))
        if pose is None:
            raise RuntimeError(f"missing pose for frame {frame_id}")
        viewmat = torch.from_numpy(pose.copy()).float().to(device)
        with torch.inference_mode():
            rendered = (
                renderer.render_features(proxy, viewmat)["feature_map"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        gt_masks = build_gt_masks(list(objects), list(categories), height, width)
        resized = np.stack(
            [frozen._resize_float_map(value, (height, width)) for value in rendered],
            axis=0,
        )
        for category in sorted({str(item["category"]) for item in objects}):
            query = category_to_index[category]
            target = np.asarray(gt_masks[category], dtype=bool)
            score = resized[query]
            flattened = score.reshape(-1).astype(np.float64)
            labels = target.reshape(-1)
            prediction = score >= FIXED_THRESHOLD
            intersection = int(np.logical_and(prediction, target).sum())
            union = int(np.logical_or(prediction, target).sum())
            positive = flattened[labels]
            negative = flattened[~labels]
            if include_oracle_threshold:
                oracle_iou, oracle_threshold = _oracle_iou(flattened, labels)
            row = {
                    "frame_id": int(frame_id),
                    "category": category,
                    "fixed_iou": float(intersection / union) if union else 1.0,
                    "selected_purity": float(
                        intersection / max(int(prediction.sum()), 1)
                    ),
                    "positive_coverage": float(
                        intersection / max(int(target.sum()), 1)
                    ),
                    "positive_negative_margin": float(
                        positive.mean() - negative.mean()
                    ),
                    "within_scene_top1": float(
                        (resized.argmax(axis=0) == query)[target].mean()
                    ),
                }
            if include_oracle_threshold:
                row["oracle_threshold_iou"] = oracle_iou
                row["oracle_threshold"] = oracle_threshold
            if include_ranking_metrics:
                row["average_precision"] = _grouped_average_precision(
                    flattened, labels
                )
                row["rank_correlation"] = _binary_midrank_correlation(
                    flattened, labels
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"{name} renderer produced no labeled samples")
    keys = [
        "fixed_iou",
        "selected_purity",
        "positive_coverage",
        "positive_negative_margin",
        "within_scene_top1",
    ]
    if include_oracle_threshold:
        keys.append("oracle_threshold_iou")
    if include_ranking_metrics:
        keys.extend(("average_precision", "rank_correlation"))
    return {
        "name": name,
        "sample_count": len(rows),
        "aggregate_sample_mean": {
            key: float(np.mean([float(row[key]) for row in rows])) for key in keys
        },
        "samples": rows,
    }


def _materialize_target_membership(
    *,
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Sequence[str],
    height: int,
    width: int,
    model: torch.nn.Module,
    renderer: Any,
    dataset: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    count = int(model.get_xyz().shape[0])
    foreground = torch.zeros(count, len(categories), dtype=torch.float32, device=device)
    total = torch.zeros_like(foreground)
    category_to_index = {value: index for index, value in enumerate(categories)}
    frames = []
    pixels = int(height) * int(width)
    for frame_id, objects in sorted(annotations.items()):
        active, target_cpu, _ = _frame_targets(
            objects,
            categories,
            height=height,
            width=width,
            boundary_radius=0,
        )
        indices = [category_to_index[value] for value in active]
        pose = (
            torch.from_numpy(dataset.pose_by_frame_idx[int(frame_id)].copy())
            .float()
            .to(device)
        )
        with torch.inference_mode():
            target = target_cpu.to(device=device, dtype=torch.float32)
            # This is exactly A^T @ target and A^T @ 1 for the sparse
            # contribution operator A.  Both rasterization and the reductions
            # are streamed over consecutive front-to-back depth intervals, so
            # memory is bounded independently of whole-frame overdraw.
            row_foreground = torch.zeros(
                count, len(indices), dtype=torch.float32, device=device
            )
            row_total = torch.zeros(count, dtype=torch.float32, device=device)
            hit_count = 0
            for hits in iter_single_view_contribution_chunks(
                model,
                renderer,
                pose,
                height=height,
                width=width,
                batch_per_iter=1,
            ):
                chunk_count = int(hits["weights"].numel())
                hit_count += chunk_count
                for start in range(0, chunk_count, TARGET_ADJOINT_HIT_CHUNK):
                    stop = min(start + TARGET_ADJOINT_HIT_CHUNK, chunk_count)
                    gids = hits["gaussian_ids"][start:stop]
                    pids = hits["pixel_ids"][start:stop]
                    weights = hits["weights"][start:stop]
                    row_total.index_add_(0, gids, weights)
                    row_foreground.index_add_(
                        0, gids, target[pids] * weights[:, None]
                    )
                del hits
            foreground[:, indices] += row_foreground
            total[:, indices] += row_total[:, None]
        frames.append(
            {
                "frame_id": int(frame_id),
                "active_queries": active,
                "exact_hits": int(hit_count),
            }
        )
        del target, row_foreground, row_total
        torch.cuda.empty_cache()
    observed = total > 1e-12
    probability = torch.where(
        observed, foreground / total.clamp_min(1e-12), torch.zeros_like(total)
    ).clamp(0.0, 1.0)
    return probability.cpu(), observed.cpu(), frames


def _enumerate_region_candidates(
    *,
    contract: Any,
    support: PrimitiveSupportGraph,
    xyz: torch.Tensor,
    global_rows: torch.Tensor,
    target_global: torch.Tensor,
    observed_global: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool((global_rows[1:] > global_rows[:-1]).all()):
        raise ValueError("support graph global rows are not strictly increasing")
    prepared = contract.prepare_graph(support, xyz)
    count = int(global_rows.numel()) * len(contract.radii_m)
    width = int(contract.maximum_tokens)
    rows_by_candidate = torch.full((count, width), -1, dtype=torch.int32)
    scores = torch.zeros(count, target_global.shape[1], dtype=torch.float32)
    anchor_local = torch.empty(count, dtype=torch.int32)
    scale_index = torch.empty(count, dtype=torch.int8)
    target = target_global[global_rows].bool()
    observed = observed_global[global_rows].bool()
    # The oracle must pay for target-positive primitives that the registered
    # SurfaceRegion graph cannot express.  Counting only graph-local positives
    # would silently turn a support-capacity audit into a conditional metric.
    target_count = target_global.bool().sum(dim=0).float()
    cursor = 0
    anchors = torch.arange(global_rows.numel(), dtype=torch.long)
    for scale, radius in enumerate(contract.radii_m):
        for start in range(0, anchors.numel(), int(batch_size)):
            selected = anchors[start : start + int(batch_size)]
            regions = contract.expand_batch(
                support,
                xyz,
                selected.tolist(),
                float(radius),
                prepared_graph=prepared,
            )
            batch_rows = torch.full((len(regions), width), -1, dtype=torch.long)
            for local, (rows, core, _distances) in enumerate(regions):
                semantic = torch.as_tensor(rows).long()[torch.as_tensor(core).bool()]
                batch_rows[local, : semantic.numel()] = semantic
            stop = cursor + len(regions)
            rows_by_candidate[cursor:stop] = batch_rows.to(torch.int32)
            anchor_local[cursor:stop] = selected.to(torch.int32)
            scale_index[cursor:stop] = int(scale)
            valid = batch_rows >= 0
            safe = batch_rows.clamp_min(0)
            gathered_target = target[safe]
            gathered_observed = observed[safe]
            active = valid[:, :, None] & gathered_observed
            intersection = (active & gathered_target).sum(dim=1).float()
            predicted = active.sum(dim=1).float()
            union = target_count[None] + predicted - intersection
            scores[cursor:stop] = torch.where(
                union > 0, intersection / union, torch.ones_like(union)
            )
            cursor = stop
    if cursor != count:
        raise RuntimeError("region candidate enumeration count differs")
    return rows_by_candidate, scores, anchor_local, scale_index


def _candidate_memberships(
    *,
    rows_by_candidate: torch.Tensor,
    candidate_scores: torch.Tensor,
    anchor_local: torch.Tensor,
    scale_index: torch.Tensor,
    global_rows: torch.Tensor,
    target_global: torch.Tensor,
    observed_global: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    count = int(target_global.shape[0])
    queries = int(target_global.shape[1])
    d2 = torch.zeros(count, queries, dtype=torch.bool)
    d3 = torch.zeros_like(d2)
    local_target = target_global[global_rows].bool()
    local_observed = observed_global[global_rows].bool()
    rows: list[dict[str, Any]] = []
    candidate_ids = np.arange(rows_by_candidate.shape[0], dtype=np.int64)
    score_array = candidate_scores.numpy()
    for query in range(queries):
        order = np.lexsort((candidate_ids, -score_array[:, query]))
        pool = torch.from_numpy(order[: min(POOL_SIZE, len(order))].copy()).long()
        best = int(pool[0])
        best_rows = rows_by_candidate[best].long()
        best_rows = best_rows[best_rows >= 0]
        d2[global_rows[best_rows], query] = True

        current_local = torch.zeros(global_rows.numel(), dtype=torch.bool)
        selected: list[int] = []
        current_iou = 0.0
        truth = local_target[:, query]
        seen = local_observed[:, query]
        truth_count = int(target_global[:, query].bool().sum())
        pool_rows = rows_by_candidate[pool].long()
        valid = pool_rows >= 0
        safe = pool_rows.clamp_min(0)
        for _ in range(MAX_UNION_REGIONS):
            additions = valid & ~current_local[safe]
            added_intersection = (additions & truth[safe] & seen[safe]).sum(dim=1)
            added_prediction = (additions & seen[safe]).sum(dim=1)
            base_intersection = int((current_local & truth & seen).sum())
            base_prediction = int((current_local & seen).sum())
            intersection = added_intersection + base_intersection
            union = truth_count + base_prediction + added_prediction - intersection
            iou = torch.where(
                union > 0,
                intersection.float() / union.float(),
                torch.ones_like(union).float(),
            )
            maximum = float(iou.max())
            if maximum <= current_iou + 1e-12:
                break
            tied = torch.where(iou == iou.max())[0]
            tied_candidate_ids = pool[tied]
            chosen_position = int(tied[int(tied_candidate_ids.argmin())])
            chosen = int(pool[chosen_position])
            chosen_rows = rows_by_candidate[chosen].long()
            chosen_rows = chosen_rows[chosen_rows >= 0]
            current_local[chosen_rows] = True
            selected.append(chosen)
            current_iou = maximum
        d3[global_rows[current_local], query] = True
        rows.append(
            {
                "query_index": int(query),
                "D2": {
                    "candidate_index": best,
                    "primitive_iou": float(candidate_scores[best, query]),
                    "anchor_local": int(anchor_local[best]),
                    "anchor_global": int(global_rows[int(anchor_local[best])]),
                    "scale_index": int(scale_index[best]),
                    "radius_m": float((0.25, 0.45, 0.7)[int(scale_index[best])]),
                    "member_rows": int(best_rows.numel()),
                },
                "D3": {
                    "primitive_iou": current_iou,
                    "regions": len(selected),
                    "candidate_indices": selected,
                    "selected_rows": int(current_local.sum()),
                },
            }
        )
    return d2, d3, rows


def run(args: argparse.Namespace) -> Path:
    root = Path(args.output_dir).expanduser().resolve()
    result_path = root / "result.json"
    membership_path = root / "oracle_memberships.pt"
    if result_path.exists() or membership_path.exists():
        raise FileExistsError("refuses to clobber D0--D5 oracle outputs")
    prereg = _require_file(
        args.preregistration, args.expected_preregistration_sha256, "preregistration"
    )
    addendum = _require_file(
        args.teacher_availability_addendum,
        args.expected_teacher_availability_addendum_sha256,
        "teacher availability addendum",
    )
    scale_addendum = _require_file(
        args.scale_axis_correction_addendum,
        args.expected_scale_axis_correction_addendum_sha256,
        "scale-axis correction addendum",
    )
    d0_path = _require_file(
        args.d0_result,
        args.expected_d0_result_sha256,
        "registered D0 result",
    )
    d1_path = _require_file(
        args.d1_result,
        args.expected_d1_result_sha256,
        "registered D1 result",
    )
    o4_path = _require_file(args.o4_cache, args.expected_o4_cache_sha256, "O4 cache")
    teacher_path = _require_file(
        args.teacher_authority,
        args.expected_teacher_authority_sha256,
        "teacher authority",
    )
    graph_path = _require_file(
        args.support_graph, args.expected_support_graph_sha256, "support graph"
    )
    descriptor_report_path = Path(args.descriptor_report).expanduser().resolve()
    descriptor_report = json.loads(descriptor_report_path.read_text(encoding="utf-8"))
    metadata = descriptor_report["metadata"]
    contract = surface_region_contract_from_metadata(metadata)
    if tuple(contract.radii_m) != (0.25, 0.45, 0.7):
        raise ValueError("registered region radii differ")

    device = torch.device(args.device)
    annotations, categories, height, width = load_lerf_ovs_labels(
        args.label_dir, "figurines"
    )
    official = set(frozen.OPEN_GAUSSIAN_LERF_FRAMES["figurines"])
    annotations = {
        frame: objects for frame, objects in annotations.items() if frame in official
    }
    model, codec, _renderer, sharpener, refiner, config, _hybrid = load_render_pipeline(
        args.config,
        args.checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    del codec, _renderer, sharpener, refiner
    gc.collect()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = frozen.build_lerf_dataset_for_scene(
        "figurines",
        config,
        args.label_dir,
        feature_height=height,
        feature_width=width,
    )
    renderer = frozen.build_mask_renderer(
        config, height=height, width=width, device=device
    )

    o4_payload, _, _ = load_torch_mapping(
        o4_path,
        expected_sha256=args.expected_o4_cache_sha256,
        map_location="cpu",
        label="O4 final relevance",
    )
    o4 = frozen.validate_ours_multiscale_query_score_cache(
        o4_payload,
        expected_xyz=model.get_xyz().detach().cpu(),
        expected_query_ids=categories,
        expected_renderer_geometry_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    if o4.score_semantics != frozen.OURS_CANONICAL_NEGATIVE_PROBABILITY_SEMANTICS:
        raise ValueError("O4 does not carry final canonical-negative relevance")
    teacher, _, _ = load_torch_mapping(
        teacher_path,
        expected_sha256=args.expected_teacher_authority_sha256,
        map_location="cpu",
        label="teacher authority",
    )
    teacher_rows = torch.as_tensor(teacher["global_rows"]).long().cpu()
    teacher_valid_local = (
        torch.as_tensor(teacher["teacher_view_mask"]).bool().any(dim=1)
    )
    if teacher.get("schema") != TEACHER_SCHEMA or teacher_rows.numel() != 82603:
        raise ValueError("teacher authority differs")
    teacher_valid = torch.zeros(len(o4.valid), dtype=torch.bool)
    teacher_valid[teacher_rows] = teacher_valid_local

    target_probability, target_observed, adjoint_frames = (
        _materialize_target_membership(
            annotations=annotations,
            categories=categories,
            height=height,
            width=width,
            model=model,
            renderer=renderer,
            dataset=dataset,
            device=device,
        )
    )
    target_binary = target_probability >= 0.5

    graph, _, _ = load_torch_mapping(
        graph_path,
        expected_sha256=args.expected_support_graph_sha256,
        map_location="cpu",
        label="support graph",
    )
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    if not torch.equal(xyz, model.get_xyz().detach().float().cpu()[global_rows]):
        raise ValueError("support graph/model geometry differs")
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"],
        edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"],
        local_sigma=graph["local_sigma"],
        num_nodes=len(xyz),
        edge_channels=graph.get("edge_channels", {}),
    )
    candidate_rows, candidate_scores, candidate_anchors, candidate_scales = (
        _enumerate_region_candidates(
            contract=contract,
            support=support,
            xyz=xyz,
            global_rows=global_rows,
            target_global=target_binary,
            observed_global=target_observed,
            batch_size=args.region_batch_size,
        )
    )
    d2, d3, oracle_choices = _candidate_memberships(
        rows_by_candidate=candidate_rows,
        candidate_scores=candidate_scores,
        anchor_local=candidate_anchors,
        scale_index=candidate_scales,
        global_rows=global_rows,
        target_global=target_binary,
        observed_global=target_observed,
    )

    d4_scores = _d4_score_variants(o4.query_scores, teacher_valid)
    d4_memberships = {
        name: score >= FIXED_THRESHOLD for name, score in d4_scores.items()
    }
    d4_metrics = {
        name: {
            "available_row_metrics": _primitive_query_metrics(
                score, target_binary, target_observed, teacher_valid
            ),
            "end_to_end_abstain_metrics": _primitive_query_metrics(
                score,
                target_binary,
                target_observed,
                torch.ones_like(teacher_valid),
            ),
        }
        for name, score in d4_scores.items()
    }
    render_memberships = {
        "D2": d2,
        "D3": d3,
        **{
            name.replace("D4_", "D5_", 1): membership
            for name, membership in d4_memberships.items()
        },
    }
    rendered = {
        name: _render_membership_diagnostic(
            name=name,
            membership=membership,
            annotations=annotations,
            categories=categories,
            height=height,
            width=width,
            model=model,
            renderer=renderer,
            dataset=dataset,
            device=device,
        )
        for name, membership in render_memberships.items()
    }

    membership_payload = {
        "schema": SCHEMA,
        "scene": "figurines",
        "query_ids": list(categories),
        "target_probability": target_probability,
        "target_observed": target_observed,
        "D2_membership": d2,
        "D3_membership": d3,
        "D4_scores": d4_scores,
        "D4_memberships": d4_memberships,
        "teacher_valid": teacher_valid,
        "metadata": {
            "development_only": True,
            "benchmark_method": False,
            "gt_oracle": True,
            "preregistration": file_record(prereg),
            "teacher_availability_addendum": file_record(addendum),
            "scale_axis_correction_addendum": file_record(scale_addendum),
        },
    }
    write_torch_noclobber(membership_path, membership_payload)
    del candidate_rows, candidate_scores, candidate_anchors, candidate_scales
    result = {
        "schema": SCHEMA,
        "scene": "figurines",
        "development_only": True,
        "benchmark_method": False,
        "deployable_candidate": False,
        "D0": {
            "source": args.d0_result,
            "fixed_miou": 0.4753887355327606,
            "average_precision": 0.7424453958489884,
            "oracle_threshold_iou": 0.6481929084043052,
        },
        "D1": {
            "source": args.d1_result,
            "fixed_miou": 0.5254328846931458,
            "average_precision": 0.7999563649716682,
            "oracle_threshold_iou": 0.6746785279094568,
        },
        "D2": rendered["D2"],
        "D3": rendered["D3"],
        "D4": {
            "available_teacher_rows": int(teacher_valid.sum()),
            "unavailable_teacher_rows": int((o4.valid & ~teacher_valid).sum()),
            "aggregation_policy": {
                "D4_s0": "fixed_registered_scale_slot_0",
                "D4_s1": "fixed_registered_scale_slot_1",
                "D4_s2": "fixed_registered_scale_slot_2",
                "D4_per_primitive_max": "fixed_target_independent_max_over_three_scale_slots_per_primitive_query",
            },
            "variants": d4_metrics,
        },
        "D5": {
            name: rendered[name]
            for name in (
                "D5_s0",
                "D5_s1",
                "D5_s2",
                "D5_per_primitive_max",
            )
        },
        "region_oracle_choices": oracle_choices,
        "target_adjoint_frames": adjoint_frames,
        "artifacts": {
            "membership": file_record(membership_path),
            "d0_result": file_record(d0_path),
            "d1_result": file_record(d1_path),
            "preregistration": file_record(prereg),
            "teacher_availability_addendum": file_record(addendum),
            "scale_axis_correction_addendum": file_record(scale_addendum),
            "o4_final_relevance": file_record(o4_path),
            "teacher_authority": file_record(teacher_path),
            "support_graph": file_record(graph_path),
            "frozen_evaluator": file_record(Path(frozen.__file__).resolve()),
            "producer": file_record(Path(__file__).resolve()),
        },
        "contract_sha256": canonical_json_sha256(
            {
                "fixed_threshold": FIXED_THRESHOLD,
                "pool_size": POOL_SIZE,
                "maximum_union_regions": MAX_UNION_REGIONS,
                "region_membership": "semantic_core_only",
                "target_membership": "exact_adjoint_ratio_ge_0p5",
                "d4_scale_variants": [
                    "fixed_s0",
                    "fixed_s1",
                    "fixed_s2",
                    "fixed_per_primitive_max",
                ],
                "result_dependent_scale_selection": False,
            }
        ),
    }
    write_frozen_json(result_path, result)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--teacher-availability-addendum", required=True)
    parser.add_argument(
        "--expected-teacher-availability-addendum-sha256", required=True
    )
    parser.add_argument("--scale-axis-correction-addendum", required=True)
    parser.add_argument(
        "--expected-scale-axis-correction-addendum-sha256", required=True
    )
    parser.add_argument("--o4-cache", required=True)
    parser.add_argument("--expected-o4-cache-sha256", required=True)
    parser.add_argument("--teacher-authority", required=True)
    parser.add_argument("--expected-teacher-authority-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--expected-support-graph-sha256", required=True)
    parser.add_argument("--descriptor-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--expected-d0-result-sha256", required=True)
    parser.add_argument("--d1-result", required=True)
    parser.add_argument("--expected-d1-result-sha256", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--region-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
