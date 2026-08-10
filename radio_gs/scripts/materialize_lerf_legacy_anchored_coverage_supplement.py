#!/usr/bin/env python3
"""Materialize a target-blind, one-way LERF coverage supplement."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.querying import legacy_anchored_coverage_supplement as supplement
from radio_gs.scripts import materialize_lerf_valid_domain_knn_candidate as legacy_v1
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_legacy_anchored_coverage_external_scores.v1"
KNN_K = 10
LOGIT_SCALE = 10.0
FROZEN_THRESHOLD = 0.6


def access_audit() -> dict[str, bool]:
    return {
        "accepted_legacy_external_score_cache_opened": True,
        "legacy_raw_o2_caches_opened": True,
        "all_available_raw_o2_caches_opened": True,
        "legacy_source_teacher_opened": True,
        "all_available_source_teacher_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
        "result_dependent_parameters": False,
    }


def _load_teacher(
    path: str,
    sha256: str,
    *,
    all_available: bool,
) -> tuple[Mapping[str, Any], dict[str, str]]:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=sha256,
        map_location="cpu",
        label="all-available source teacher" if all_available else "legacy source teacher",
    )
    expected_schema = (
        "radio_gs.lerf_source_teacher_mean_siglip_all_available.v2"
        if all_available
        else "radio_gs.lerf_source_teacher_mean_siglip.v2"
    )
    rows = payload.get("global_rows")
    valid = payload.get("teacher_valid")
    retained = payload.get("retained_view_count")
    mean = payload.get("teacher_mean")
    if (
        payload.get("schema") != expected_schema
        or not torch.is_tensor(rows)
        or rows.ndim != 1
        or rows.dtype != torch.long
        or not torch.is_tensor(valid)
        or valid.shape != rows.shape
        or valid.dtype != torch.bool
        or not torch.is_tensor(retained)
        or retained.shape != rows.shape
        or retained.dtype != torch.uint8
        or not torch.is_tensor(mean)
        or mean.ndim != 2
        or mean.shape[0] != rows.numel()
        or payload.get("access_audit", {}).get("target_metrics_opened") is not False
    ):
        raise ValueError("source teacher cache contract differs")
    return payload, file_record(path)


def _load_accepted(
    path: str,
    sha256: str,
) -> tuple[Mapping[str, Any], dict[str, str]]:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=sha256,
        map_location="cpu",
        label="accepted legacy valid-domain scores",
    )
    scores = payload.get("query_scores")
    valid = payload.get("valid")
    xyz = payload.get("xyz")
    metadata = payload.get("metadata")
    authority = payload.get("authority")
    if (
        payload.get("schema") != legacy_v1.SCHEMA
        or not torch.is_tensor(scores)
        or scores.ndim != 2
        or scores.dtype != torch.float32
        or not scores.is_contiguous()
        or not torch.is_tensor(valid)
        or valid.shape != scores.shape[:1]
        or valid.dtype != torch.bool
        or not torch.is_tensor(xyz)
        or xyz.shape != (scores.shape[0], 3)
        or not isinstance(metadata, Mapping)
        or not isinstance(authority, Mapping)
        or authority.get("method", {}).get("contract") != legacy_v1.readout.CONTRACT
        or int(authority.get("method", {}).get("knn_k", -1)) != KNN_K
        or float(authority.get("method", {}).get("canonical_negative_logit_scale", -1.0))
        != LOGIT_SCALE
    ):
        raise ValueError("accepted legacy external-score cache contract differs")
    if not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()) or bool(
        (scores > 1).any()
    ):
        raise ValueError("accepted legacy scores must be finite in [0,1]")
    return payload, file_record(path)


def _assert_pair_axes_equal(
    legacy_positive: legacy_v1.frozen.OursMultiscaleQueryScoreCache,
    legacy_negative: legacy_v1.frozen.OursMultiscaleQueryScoreCache,
    all_positive: legacy_v1.frozen.OursMultiscaleQueryScoreCache,
    all_negative: legacy_v1.frozen.OursMultiscaleQueryScoreCache,
) -> None:
    for name, left, right in (
        ("positive query ids", legacy_positive.query_ids, all_positive.query_ids),
        ("negative query ids", legacy_negative.query_ids, all_negative.query_ids),
        ("scale ids", legacy_positive.scale_ids, all_positive.scale_ids),
        ("scale radii", legacy_positive.scale_radii_m, all_positive.scale_radii_m),
        ("xyz sha256", legacy_positive.xyz_sha256, all_positive.xyz_sha256),
        (
            "renderer checkpoint",
            legacy_positive.renderer_geometry_checkpoint_sha256,
            all_positive.renderer_geometry_checkpoint_sha256,
        ),
    ):
        if left != right:
            raise ValueError(f"legacy/all-available {name} differ")
    for name, left, right in (
        ("positive valid", legacy_positive.valid, all_positive.valid),
        ("negative valid", legacy_negative.valid, all_negative.valid),
    ):
        if not torch.equal(left, right):
            raise ValueError(f"legacy/all-available {name} differ")


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    data = torch.as_tensor(values).detach().float().cpu().reshape(-1)
    if not data.numel():
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(data.min()),
        "p50": float(torch.quantile(data, 0.50)),
        "p95": float(torch.quantile(data, 0.95)),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def _shared_anchor_components(neighbor_rows: torch.Tensor) -> dict[str, float | int]:
    rows = torch.as_tensor(neighbor_rows).detach().long().cpu()
    count = int(rows.shape[0])
    if count == 0:
        return {"component_count": 0, "largest_component_rows": 0, "largest_component_fraction": 0.0}
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_anchor: dict[int, int] = {}
    for row_index, anchors in enumerate(rows.tolist()):
        for anchor in anchors:
            first = first_by_anchor.setdefault(int(anchor), row_index)
            union(row_index, first)
    sizes: dict[int, int] = {}
    for row_index in range(count):
        root = find(row_index)
        sizes[root] = sizes.get(root, 0) + 1
    largest = max(sizes.values())
    return {
        "component_count": len(sizes),
        "largest_component_rows": largest,
        "largest_component_fraction": largest / count,
    }


def source_diagnostics(
    result: supplement.LegacyAnchoredCoverageResult,
    accepted_scores: torch.Tensor,
    global_rows: torch.Tensor,
    legacy_teacher_mean: torch.Tensor,
    all_available_teacher_mean: torch.Tensor,
    all_available_retained_count: torch.Tensor,
) -> dict[str, Any]:
    accepted = torch.as_tensor(accepted_scores).detach().float().cpu()
    new_rows = result.supplement_rows
    changed = result.scores != accepted
    old_new = accepted[new_rows]
    new_new = result.scores[new_rows]
    threshold = FROZEN_THRESHOLD
    diagnostics: dict[str, Any] = {
        "supplement_rows": int(new_rows.numel()),
        "changed_rows": int(changed.any(dim=1).sum()),
        "changed_cells": int(changed.sum()),
        "changed_cells_outside_supplement": int(changed[~result.supplement_mask].sum()),
        "frozen_threshold": threshold,
        "crossed_below_to_at_or_above_threshold_cells": int(
            ((old_new < threshold) & (new_new >= threshold)).sum()
        ),
        "crossed_at_or_above_to_below_threshold_cells": int(
            ((old_new >= threshold) & (new_new < threshold)).sum()
        ),
        "old_at_or_above_threshold_cells_on_supplement_rows": int(
            (old_new >= threshold).sum()
        ),
        "new_at_or_above_threshold_cells_on_supplement_rows": int(
            (new_new >= threshold).sum()
        ),
        "legacy_rows_bitwise_unchanged": bool(
            torch.equal(result.scores[~result.supplement_mask], accepted[~result.supplement_mask])
        ),
    }
    if not int(new_rows.numel()):
        diagnostics.update(
            {
                "spatial_attachment": {
                    "nearest_legacy_anchor_distance": _quantiles(torch.empty(0)),
                    "kth_legacy_anchor_distance": _quantiles(torch.empty(0)),
                    "nearest_distance_over_anchor_local_radius": _quantiles(torch.empty(0)),
                    "within_nearest_anchor_local_radius_fraction": 0.0,
                    **_shared_anchor_components(result.neighbor_rows),
                },
                "score_neighbor_consistency": {
                    "mean_cosine": 0.0,
                    "mean_absolute_difference": 0.0,
                    "threshold_agreement_fraction": 0.0,
                },
                "source_teacher_neighbor_consistency": {"mean_cosine": 0.0, "p05_cosine": 0.0},
                "new_row_retained_view_count": _quantiles(torch.empty(0)),
            }
        )
        return diagnostics

    nearest = result.neighbor_distances[:, 0]
    kth = result.neighbor_distances[:, -1]
    radius_ratio = nearest / result.nearest_anchor_local_radius.clamp_min(1e-12)
    neighbor_score_mean = accepted[result.neighbor_rows].mean(dim=1)
    score_cosine = F.cosine_similarity(new_new, neighbor_score_mean, dim=1, eps=1e-8)

    rows = torch.as_tensor(global_rows).detach().long().cpu()
    new_local = torch.searchsorted(rows, new_rows)
    neighbor_local = torch.searchsorted(rows, result.neighbor_rows)
    source_new = torch.as_tensor(all_available_teacher_mean).detach().float().cpu()[new_local]
    source_neighbor = (
        torch.as_tensor(legacy_teacher_mean).detach().float().cpu()[neighbor_local].mean(dim=1)
    )
    source_cosine = F.cosine_similarity(source_new, source_neighbor, dim=1, eps=1e-8)
    retained = torch.as_tensor(all_available_retained_count).detach().float().cpu()[new_local]
    diagnostics.update(
        {
            "spatial_attachment": {
                "nearest_legacy_anchor_distance": _quantiles(nearest),
                "kth_legacy_anchor_distance": _quantiles(kth),
                "nearest_distance_over_anchor_local_radius": _quantiles(radius_ratio),
                "within_nearest_anchor_local_radius_fraction": float(
                    (radius_ratio <= 1.0).float().mean()
                ),
                **_shared_anchor_components(result.neighbor_rows),
            },
            "score_neighbor_consistency": {
                "mean_cosine": float(score_cosine.mean()),
                "p05_cosine": float(torch.quantile(score_cosine, 0.05)),
                "mean_absolute_difference": float((new_new - neighbor_score_mean).abs().mean()),
                "threshold_agreement_fraction": float(
                    ((new_new >= threshold) == (neighbor_score_mean >= threshold)).float().mean()
                ),
            },
            "source_teacher_neighbor_consistency": {
                "mean_cosine": float(source_cosine.mean()),
                "p05_cosine": float(torch.quantile(source_cosine, 0.05)),
            },
            "new_row_retained_view_count": _quantiles(retained),
        }
    )
    return diagnostics


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_cache).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    if str(output) != args.output_cache or str(report_path) != args.output_report:
        raise ValueError("output paths must be canonical absolute")
    if output == report_path or output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("coverage supplement outputs must be new and distinct")

    accepted, accepted_record = _load_accepted(
        args.accepted_cache, args.accepted_cache_sha256
    )
    (
        legacy_positive_raw,
        _legacy_negative_raw,
        legacy_positive,
        legacy_negative,
        legacy_positive_record,
        legacy_negative_record,
    ) = legacy_v1._load_pair(
        args.legacy_positive_cache,
        args.legacy_positive_cache_sha256,
        args.legacy_negative_cache,
        args.legacy_negative_cache_sha256,
    )
    (
        all_positive_raw,
        _all_negative_raw,
        all_positive,
        all_negative,
        all_positive_record,
        all_negative_record,
    ) = legacy_v1._load_pair(
        args.all_available_positive_cache,
        args.all_available_positive_cache_sha256,
        args.all_available_negative_cache,
        args.all_available_negative_cache_sha256,
    )
    _assert_pair_axes_equal(
        legacy_positive, legacy_negative, all_positive, all_negative
    )
    legacy_teacher, legacy_teacher_record = _load_teacher(
        args.legacy_teacher,
        args.legacy_teacher_sha256,
        all_available=False,
    )
    all_teacher, all_teacher_record = _load_teacher(
        args.all_available_teacher,
        args.all_available_teacher_sha256,
        all_available=True,
    )

    query_names = tuple(str(value) for value in accepted["metadata"].get("query_names", []))
    if query_names != legacy_positive.query_ids:
        raise ValueError("accepted/raw positive query axes differ")
    if not torch.equal(torch.as_tensor(accepted["xyz"]), torch.as_tensor(legacy_positive_raw["xyz"])):
        raise ValueError("accepted/raw xyz differ")
    if not torch.equal(torch.as_tensor(accepted["valid"]), legacy_positive.valid):
        raise ValueError("accepted/raw geometry-valid masks differ")
    if not torch.equal(legacy_teacher["global_rows"], all_teacher["global_rows"]):
        raise ValueError("legacy/all-available teacher row axes differ")
    accepted_sources = accepted["authority"].get("source_artifacts", {})
    if accepted_sources.get("positive_raw_score_cache") != legacy_positive_record or accepted_sources.get(
        "canonical_negative_raw_score_cache"
    ) != legacy_negative_record:
        raise ValueError("accepted cache is not bound to supplied legacy raw O2 pair")

    result = supplement.legacy_anchored_coverage_supplement(
        accepted["query_scores"],
        legacy_positive.query_scores,
        legacy_negative.query_scores,
        all_positive.query_scores,
        all_negative.query_scores,
        accepted["xyz"],
        accepted["valid"],
        legacy_teacher["global_rows"],
        legacy_teacher["teacher_valid"],
        all_teacher["teacher_valid"],
        k=KNN_K,
        chunk_size=args.chunk_size,
        logit_scale=LOGIT_SCALE,
    )
    diagnostics = source_diagnostics(
        result,
        accepted["query_scores"],
        legacy_teacher["global_rows"],
        legacy_teacher["teacher_mean"],
        all_teacher["teacher_mean"],
        all_teacher["retained_view_count"],
    )
    if not diagnostics["legacy_rows_bitwise_unchanged"] or diagnostics[
        "changed_cells_outside_supplement"
    ] != 0:
        raise AssertionError("legacy invariance diagnostics failed")

    authority = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene_id": args.scene_id,
        "method": {
            "contract": supplement.CONTRACT,
            "producer": file_record(Path(__file__).resolve()),
            "implementation": file_record(Path(supplement.__file__).resolve()),
            "legacy_reconstruction_bitwise_required": True,
            "legacy_scores_outside_supplement": "direct_bitwise_copy",
            "supplement_domain": "legacy_teacher_invalid_and_all_available_teacher_valid",
            "supplement_neighbor_domain": "legacy_teacher_valid_rows_only",
            "supplement_knn_k": KNN_K,
            "supplement_knn_blend": "0.5*all_available_raw+0.5*legacy_neighbor_mean",
            "normalization": "frozen_legacy_valid_domain_smoothed_minmax",
            "scale_selection": "frozen_legacy_valid_domain_selected_scale_per_query",
            "new_rows_contribute_to_legacy_statistics": False,
            "query_conditioned_or_scene_specific_parameters": False,
            "canonical_negative_logit_scale": LOGIT_SCALE,
        },
        "source_artifacts": {
            "accepted_legacy_external_scores": accepted_record,
            "legacy_positive_raw_o2": legacy_positive_record,
            "legacy_negative_raw_o2": legacy_negative_record,
            "all_available_positive_raw_o2": all_positive_record,
            "all_available_negative_raw_o2": all_negative_record,
            "legacy_source_teacher": legacy_teacher_record,
            "all_available_source_teacher": all_teacher_record,
        },
        "geometry_axis": {
            "num_gaussians": int(result.scores.shape[0]),
            "geometry_valid_gaussians": int(torch.as_tensor(accepted["valid"]).sum()),
            "legacy_teacher_valid_rows": int(legacy_teacher["teacher_valid"].sum()),
            "all_available_teacher_valid_rows": int(all_teacher["teacher_valid"].sum()),
            "supplement_rows": int(result.supplement_rows.numel()),
            "renderer_geometry_checkpoint_sha256": legacy_positive.renderer_geometry_checkpoint_sha256,
            "xyz_sha256": legacy_positive.xyz_sha256,
        },
        "query_axis": list(query_names),
        "selected_scale_indices_frozen_from_legacy": result.selected_scale_indices.tolist(),
        "source_diagnostics": diagnostics,
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
    }
    payload = {
        "schema": SCHEMA,
        "query_scores": result.scores.float().contiguous(),
        "valid": torch.as_tensor(accepted["valid"]).detach().bool().cpu().contiguous(),
        "xyz": torch.as_tensor(accepted["xyz"]).detach().float().cpu().contiguous(),
        "metadata": {
            "query_names": list(query_names),
            "score_semantics": "legacy_valid_domain_scores_bitwise_plus_one_way_all_available_coverage",
            "score_postprocess": "none_already_legacy_anchored",
        },
        "authority": authority,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    report = {
        **authority,
        "status": "complete_source_only_premetric_legacy_anchored_coverage_supplement",
        "output_cache": file_record(output),
        "finite": True,
    }
    write_frozen_json(report_path, report)
    return {**report, "output_report": file_record(report_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--accepted-cache", required=True)
    parser.add_argument("--accepted-cache-sha256", required=True)
    parser.add_argument("--legacy-positive-cache", required=True)
    parser.add_argument("--legacy-positive-cache-sha256", required=True)
    parser.add_argument("--legacy-negative-cache", required=True)
    parser.add_argument("--legacy-negative-cache-sha256", required=True)
    parser.add_argument("--all-available-positive-cache", required=True)
    parser.add_argument("--all-available-positive-cache-sha256", required=True)
    parser.add_argument("--all-available-negative-cache", required=True)
    parser.add_argument("--all-available-negative-cache-sha256", required=True)
    parser.add_argument("--legacy-teacher", required=True)
    parser.add_argument("--legacy-teacher-sha256", required=True)
    parser.add_argument("--all-available-teacher", required=True)
    parser.add_argument("--all-available-teacher-sha256", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--chunk-size", type=int, default=65536)
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "FROZEN_THRESHOLD",
    "KNN_K",
    "LOGIT_SCALE",
    "SCHEMA",
    "access_audit",
    "materialize",
    "source_diagnostics",
]
