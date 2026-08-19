#!/usr/bin/env python3
"""Audit whether query-free P0 proposals cover immutable text identity seeds.

This is a label-free proposal-availability oracle.  For each text argmax it
selects, independently in every source view, the highest-confidence automatic
SAM3 hierarchy node that contains the seed.  It reports source-view and 3-D
extent coverage only; benchmark masks never enter proposal construction or
the audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_query_free_p0_identity_seed_coverage.v1"


def _names(raw: str) -> list[str]:
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    if not values or len(values) != len(set(value.casefold() for value in values)):
        raise ValueError("query names must be non-empty and unique")
    return values


def build(args: argparse.Namespace) -> dict[str, Any]:
    sidecar_path = Path(args.membership_sidecar).expanduser().resolve()
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    sidecar_metadata = dict(sidecar.get("metadata", {}))
    if (
        int(sidecar.get("schema_version", -1)) != 1
        or sidecar_metadata.get("membership_lifting") != "raster_adjoint"
        or sidecar_metadata.get("raster_lifting_semantics")
        != "true_alpha_compositing_adjoint"
        or not bool(sidecar_metadata.get("query_free", False))
        or not bool(sidecar_metadata.get("offline_teacher_audit_only", False))
        or any(bool(sidecar_metadata.get(key, False)) for key in (
            "labels_opened", "instances_opened", "text_opened",
        ))
    ):
        raise ValueError("membership sidecar is not query-free true-adjoint P0 evidence")

    graph_path = Path(args.support_graph).expanduser().resolve()
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_rows = torch.as_tensor(graph.get("global_rows")).long().cpu().reshape(-1)
    graph_xyz = torch.as_tensor(graph.get("xyz")).float().cpu()
    num_global = int(graph.get("num_global_rows", -1))
    if graph_rows.shape != (len(graph_xyz),) or num_global <= 0:
        raise ValueError("support graph row identity differs")
    global_to_local = torch.full((num_global,), -1, dtype=torch.long)
    global_to_local[graph_rows] = torch.arange(len(graph_rows), dtype=torch.long)

    score_path = Path(args.query_score_cache).expanduser().resolve()
    score_payload = torch.load(score_path, map_location="cpu", weights_only=False)
    scores = torch.as_tensor(score_payload.get("query_scores")).float().cpu()
    valid = torch.as_tensor(score_payload.get("valid", torch.ones(len(scores)))).bool().cpu()
    query_names = _names(args.query_names)
    cached_names = [
        str(value) for value in dict(score_payload.get("metadata", {})).get("query_names", [])
    ]
    if (
        scores.shape != (num_global, len(query_names))
        or valid.shape != (num_global,)
        or cached_names != query_names
    ):
        raise ValueError("query score cache does not match declared Gaussian/query rows")
    scores[~valid] = -1.0e4
    seeds = torch.argmax(scores, dim=0)

    records = list(sidecar.get("records", []))
    if not records:
        raise ValueError("P0 membership sidecar has no source records")
    threshold = float(args.inside_threshold)
    query_reports: list[dict[str, Any]] = []
    for query_index, query_name in enumerate(query_names):
        seed_global = int(seeds[query_index])
        seed_local = int(global_to_local[seed_global])
        selected: list[torch.Tensor] = []
        selected_frames: list[str] = []
        selected_indices: list[int] = []
        seed_observed_frames = 0
        for record in records:
            membership = torch.as_tensor(record.get("membership")).float().cpu()
            observed = torch.as_tensor(record.get("observed")).bool().cpu().reshape(-1)
            quality = torch.as_tensor(record.get("quality")).float().cpu().reshape(-1)
            source_indices = torch.as_tensor(
                record.get("source_mask_indices")
            ).long().cpu().reshape(-1)
            if (
                membership.ndim != 2
                or membership.shape[1] != len(graph_rows)
                or observed.shape != (len(graph_rows),)
                or quality.shape != (membership.shape[0],)
                or source_indices.shape != quality.shape
            ):
                raise ValueError("P0 membership record rows differ")
            if seed_local < 0 or not bool(observed[seed_local]):
                continue
            seed_observed_frames += 1
            candidate = torch.where(membership[:, seed_local] >= threshold)[0]
            if not len(candidate):
                continue
            confidence = membership[candidate, seed_local] * quality[candidate]
            local_choice = int(candidate[int(torch.argmax(confidence))])
            selected.append(membership[local_choice].clamp(0.0, 1.0))
            selected_frames.append(str(record.get("mask_frame", "")))
            selected_indices.append(int(source_indices[local_choice]))
        if selected:
            stacked = torch.stack(selected)
            union = 1.0 - torch.prod(1.0 - stacked, dim=0)
            extent_rows = [int((row >= threshold).sum()) for row in stacked]
            union_rows = int((union >= 0.5).sum())
        else:
            extent_rows = []
            union_rows = 0
        query_reports.append({
            "query": query_name,
            "identity_seed_global_row": seed_global,
            "identity_seed_graph_row": seed_local,
            "seed_in_query_free_graph": seed_local >= 0,
            "seed_observed_source_frames": seed_observed_frames,
            "proposal_covering_seed_source_frames": len(selected),
            "two_view_object_hypothesis_available": len(selected) >= 2,
            "selected_mask_frames": selected_frames,
            "selected_source_mask_indices": selected_indices,
            "selected_extent_rows": extent_rows,
            "noisy_or_union_rows_at_0p5": union_rows,
        })

    in_graph = sum(bool(row["seed_in_query_free_graph"]) for row in query_reports)
    one_view = sum(int(row["proposal_covering_seed_source_frames"]) >= 1 for row in query_reports)
    two_view = sum(bool(row["two_view_object_hypothesis_available"]) for row in query_reports)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"coverage audit already exists: {output}")
    report = {
        "schema": SCHEMA,
        "status": "complete_label_free_proposal_coverage_audit",
        "scene": str(args.scene),
        "queries": len(query_reports),
        "identity_seed_in_graph": in_graph,
        "identity_seed_in_graph_fraction": in_graph / len(query_reports),
        "queries_with_at_least_one_covering_p0_view": one_view,
        "one_view_coverage_fraction": one_view / len(query_reports),
        "queries_with_at_least_two_covering_p0_views": two_view,
        "two_view_coverage_fraction": two_view / len(query_reports),
        "inside_threshold": threshold,
        "selection": "best_quality_times_membership_node_containing_immutable_text_argmax_per_view",
        "query_independent_mask_hierarchy": True,
        "benchmark_masks_opened": False,
        "evaluation_rgb_opened": False,
        "prediction_constructed": False,
        "promotion": False,
        "promotion_reason": "coverage audit is not a benchmark prediction",
        "authorities": {
            "membership_sidecar": {
                "path": str(sidecar_path), "sha256": sha256_file(sidecar_path),
            },
            "support_graph": {
                "path": str(graph_path), "sha256": sha256_file(graph_path),
            },
            "query_score_cache": {
                "path": str(score_path), "sha256": sha256_file(score_path),
            },
        },
        "per_query": query_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--membership-sidecar", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--query-score-cache", required=True)
    parser.add_argument("--query-names", required=True)
    parser.add_argument("--inside-threshold", type=float, default=0.8)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
