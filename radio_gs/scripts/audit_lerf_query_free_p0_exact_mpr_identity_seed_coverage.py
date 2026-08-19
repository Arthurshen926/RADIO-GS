#!/usr/bin/env python3
"""Label-free identity-seed coverage audit for query-free exact-MPR P0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_query_free_p0_exact_mpr_identity_seed_coverage.v1"
MEMBERSHIP_SCHEMA = "radio_gs.lerf_query_free_sam3_exact_mpr_memberships.v1"
IDENTITY_SCORE_SCHEMA = "radio_gs.lerf_field_only_primitive_identity_scores.v1"


def _names(raw: str) -> list[str]:
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    if not values or len(values) != len(set(value.casefold() for value in values)):
        raise ValueError("query names must be non-empty and unique")
    return values


def build(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership_cache).expanduser().resolve()
    payload = torch.load(membership_path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if (
        payload.get("schema") != MEMBERSHIP_SCHEMA
        or int(payload.get("schema_version", -1)) != 1
        or str(payload.get("scene", "")) != str(args.scene)
        or metadata.get("query_independent_proposal_set") is not True
        or metadata.get("query_independent_mask_hierarchy") is not False
        or metadata.get("hierarchy_parent_edges_materialized") is not False
        or metadata.get("mask_tensor_semantics") != "packed_boolean"
        or metadata.get("membership_lifting")
        != "exact_front_to_back_marginal_target_weight"
    ):
        raise ValueError("membership cache is not query-free packed-bool exact-MPR P0")
    if any(bool(metadata.get(key, False)) for key in (
        "benchmark_masks_opened", "evaluation_rgb_opened", "text_queries_opened",
    )):
        raise ValueError("P0 cache violates source-only construction provenance")
    num_rows = int(payload.get("num_rows", -1))
    num_proposals = int(payload.get("num_proposals", -1))
    rows = torch.as_tensor(payload.get("row_indices")).long().cpu().reshape(-1)
    proposals = torch.as_tensor(payload.get("proposal_indices")).long().cpu().reshape(-1)
    weights = torch.as_tensor(payload.get("weights")).float().cpu().reshape(-1)
    proposal_views = torch.as_tensor(payload.get("proposal_view_indices")).long().cpu()
    proposal_scores = torch.as_tensor(payload.get("proposal_scores")).float().cpu()
    proposal_area = torch.as_tensor(payload.get("proposal_area_fraction")).float().cpu()
    proposal_stability = torch.as_tensor(payload.get("proposal_stability")).float().cpu()
    proposal_boxes = torch.as_tensor(payload.get("proposal_boxes_xyxy")).long().cpu()
    proposal_seeds = torch.as_tensor(payload.get("proposal_seed_xy")).float().cpu()
    proposal_prompt = torch.as_tensor(payload.get("proposal_prompt_index")).long().cpu()
    proposal_candidate = torch.as_tensor(payload.get("proposal_candidate_index")).long().cpu()
    view_denominator = torch.as_tensor(payload.get("view_denominator")).float().cpu()
    view_observed = torch.as_tensor(payload.get("view_observed")).bool().cpu()
    num_views = int(metadata.get("source_view_count", -1))
    if (
        not (rows.shape == proposals.shape == weights.shape)
        or proposal_views.shape != (num_proposals,)
        or proposal_scores.shape != (num_proposals,)
        or proposal_area.shape != (num_proposals,)
        or proposal_stability.shape != (num_proposals,)
        or proposal_boxes.shape != (num_proposals, 4)
        or proposal_seeds.shape != (num_proposals, 2)
        or proposal_prompt.shape != (num_proposals,)
        or proposal_candidate.shape != (num_proposals,)
        or view_denominator.shape != (num_views, num_rows)
        or view_observed.shape != view_denominator.shape
        or not torch.equal(view_observed, view_denominator > 0)
        or not bool(torch.isfinite(view_denominator).all())
        or bool((view_denominator < 0).any())
        or not bool(torch.isfinite(weights).all())
        or bool(((weights <= 0) | (weights > 1)).any())
        or not bool(torch.isfinite(proposal_scores).all())
        or bool(((proposal_scores < 0) | (proposal_scores > 1)).any())
        or not bool(torch.isfinite(proposal_area).all())
        or bool(((proposal_area < 0) | (proposal_area > 1)).any())
        or not bool(torch.isfinite(proposal_stability).all())
        or bool(((proposal_stability < 0) | (proposal_stability > 1)).any())
        or not bool(torch.isfinite(proposal_seeds).all())
        or (len(proposal_views) and (
            int(proposal_views.min()) < 0 or int(proposal_views.max()) >= num_views
        ))
        or (len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= num_rows))
        or (len(proposals) and (int(proposals.min()) < 0 or int(proposals.max()) >= num_proposals))
    ):
        raise ValueError("P0 sparse membership rows differ")

    score_path = Path(args.query_score_cache).expanduser().resolve()
    score_payload = torch.load(score_path, map_location="cpu", weights_only=False)
    score_metadata = dict(score_payload.get("metadata", {}))
    query_names = _names(args.query_names)
    scores = torch.as_tensor(score_payload.get("query_scores")).float().cpu()
    valid = torch.as_tensor(score_payload.get("valid")).bool().cpu()
    cached_names = [
        str(value) for value in dict(score_payload.get("metadata", {})).get("query_names", [])
    ]
    primitive_path = Path(str(metadata.get("primitive_cache", ""))).resolve()
    if (
        not primitive_path.is_file()
        or sha256_file(primitive_path) != str(metadata.get("primitive_cache_sha256", ""))
    ):
        raise ValueError("P0 primitive-cache binding differs")
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    primitive_xyz = torch.as_tensor(primitive.get("xyz")).float().cpu()
    score_xyz = torch.as_tensor(score_payload.get("xyz")).float().cpu()
    xyz_digest = hashlib.sha256(
        primitive_xyz.contiguous().numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    if (
        score_payload.get("schema") != IDENTITY_SCORE_SCHEMA
        or int(score_payload.get("schema_version", -1)) != 1
        or str(score_payload.get("scene", "")) != str(args.scene)
        or scores.shape != (num_rows, len(query_names))
        or valid.shape != (num_rows,)
        or cached_names != query_names
        or score_metadata.get("score_role") != "field_only_primitive_identity_seed"
        or score_metadata.get("region_membership_cache_opened") is not False
        or score_metadata.get("proposal_cache_opened") is not False
        or any(bool(score_metadata.get(key, False)) for key in (
            "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened",
        ))
        or Path(str(score_metadata.get("primitive_query_cache", ""))).resolve()
        != primitive_path
        or str(score_metadata.get("primitive_query_cache_sha256", ""))
        != str(metadata.get("primitive_cache_sha256", ""))
        or str(score_metadata.get("xyz_sha256", "")) != xyz_digest
        or score_metadata.get("primitive_field_checkpoint")
        != dict(primitive.get("metadata", {})).get("field_checkpoint")
        or primitive_xyz.shape != (num_rows, 3)
        or score_xyz.shape != primitive_xyz.shape
        or not torch.equal(score_xyz, primitive_xyz)
        or xyz_digest != str(metadata.get("xyz_sha256", ""))
        or not bool(valid.any())
        or not bool(torch.isfinite(scores[valid]).all())
    ):
        raise ValueError("query score cache rows or names differ")
    for path_key, sha_key in (
        ("text_embedding_cache", "text_embedding_cache_sha256"),
        ("canonical_embedding_cache", "canonical_embedding_cache_sha256"),
    ):
        bound_path = Path(str(score_metadata.get(path_key, ""))).resolve()
        if (
            not bound_path.is_file()
            or sha256_file(bound_path) != str(score_metadata.get(sha_key, ""))
        ):
            raise ValueError(f"field-only identity score {path_key} binding differs")
    if not bool(valid.any()) or any(not bool(torch.isfinite(scores[valid, q]).any()) for q in range(scores.shape[1])):
        raise ValueError("query score cache has no finite valid identity candidate")
    scores[~valid] = float("-inf")
    seeds = torch.argmax(scores, dim=0)
    threshold = float(args.inside_threshold)

    reports: list[dict[str, Any]] = []
    for query_index, query_name in enumerate(query_names):
        seed = int(seeds[query_index])
        observed_views = torch.where(view_observed[:, seed])[0]
        at_seed = (rows == seed) & (weights >= threshold)
        seed_pairs = torch.where(at_seed)[0]
        candidate_props = torch.unique(proposals[seed_pairs], sorted=True)
        chosen: list[int] = []
        chosen_seed_membership: list[float] = []
        for view in torch.unique(proposal_views[candidate_props], sorted=True):
            in_view = candidate_props[proposal_views[candidate_props] == view]
            candidate_values: list[float] = []
            for proposal in in_view.tolist():
                pair = seed_pairs[proposals[seed_pairs] == int(proposal)]
                membership = float(weights[pair].max())
                candidate_values.append(membership * float(proposal_scores[proposal]))
            best_index = int(torch.tensor(candidate_values).argmax())
            best = int(in_view[best_index])
            chosen.append(best)
            pair = seed_pairs[proposals[seed_pairs] == best]
            chosen_seed_membership.append(float(weights[pair].max()))

        if chosen:
            chosen_tensor = torch.tensor(chosen, dtype=torch.long)
            selected_pairs = torch.isin(proposals, chosen_tensor)
            selected_rows = rows[selected_pairs]
            selected_weights = weights[selected_pairs].clamp(0.0, 1.0)
            log_survival = torch.zeros(num_rows, dtype=torch.float32)
            log_survival.index_add_(
                0, selected_rows,
                torch.log1p(-selected_weights.clamp_max(1.0 - 1e-6)),
            )
            union = 1.0 - torch.exp(log_survival)
            union_rows = int((union >= 0.5).sum())
            per_proposal_rows = [
                int(((proposals == proposal) & (weights >= threshold)).sum())
                for proposal in chosen
            ]
        else:
            union_rows = 0
            per_proposal_rows = []
        reports.append({
            "query": query_name,
            "identity_seed_global_row": seed,
            "identity_seed_observed_source_views": int(len(observed_views)),
            "candidate_proposals_covering_seed": int(len(candidate_props)),
            "source_views_with_covering_proposal": len(chosen),
            "observed_view_proposal_misses": int(len(observed_views) - len(chosen)),
            "unobserved_view_unknowns": int(num_views - len(observed_views)),
            "two_view_object_hypothesis_available": len(chosen) >= 2,
            "selected_proposal_indices": chosen,
            "selected_seed_membership": chosen_seed_membership,
            "selected_extent_rows_at_inside_threshold": per_proposal_rows,
            "noisy_or_union_rows_at_0p5": union_rows,
        })

    one_view = sum(int(row["source_views_with_covering_proposal"]) >= 1 for row in reports)
    two_view = sum(bool(row["two_view_object_hypothesis_available"]) for row in reports)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"coverage audit already exists: {output}")
    report = {
        "schema": SCHEMA,
        "status": "complete_label_free_field_identity_seed_coverage_audit",
        "scene": str(args.scene),
        "queries": len(reports),
        "p0_proposals": num_proposals,
        "p0_source_views": num_views,
        "p0_memberships": int(len(rows)),
        "grid_size": int(metadata.get("grid_size", -1)),
        "formal_stage_a_complete": bool(metadata.get("formal_stage_a_complete", False)),
        "sparse_p0_pilot_complete": bool(metadata.get("sparse_p0_pilot_complete", False)),
        "queries_with_at_least_one_covering_view": one_view,
        "one_view_coverage_fraction": one_view / len(reports),
        "queries_with_at_least_two_covering_views": two_view,
        "two_view_coverage_fraction": two_view / len(reports),
        "inside_threshold": threshold,
        "selection": "best_quality_times_exact_mpr_membership_node_containing_field_only_text_argmax_per_view",
        "query_independent_proposal_set": True,
        "query_independent_mask_hierarchy": False,
        "benchmark_masks_opened": False,
        "evaluation_rgb_opened": False,
        "prediction_constructed": False,
        "promotion": False,
        "promotion_reason": (
            "grid4 is an undercoverage smoke pilot"
            if int(metadata.get("grid_size", -1)) < 12
            else "coverage audit is not a benchmark prediction"
        ),
        "membership_cache": {
            "path": str(membership_path), "sha256": sha256_file(membership_path),
        },
        "query_score_cache": {
            "path": str(score_path), "sha256": sha256_file(score_path),
        },
        "per_query": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--membership-cache", required=True)
    parser.add_argument("--query-score-cache", required=True)
    parser.add_argument("--query-names", required=True)
    parser.add_argument("--inside-threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
