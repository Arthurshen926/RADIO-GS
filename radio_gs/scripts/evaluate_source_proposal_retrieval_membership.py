#!/usr/bin/env python3
"""Source-heldout gate for proposal identity transport in one Gaussian carrier.

A held-out source mask supplies only a native object descriptor.  Candidate
memberships come from disjoint source views.  The calibration residue selects
one global descriptor mode and top-k value before the evaluation residue is
opened.  No benchmark query, image, mask or metric is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _proposal_support,
    compose_membership_query_features,
)
from radio_gs.utils.immutable_artifacts import (
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
)


def _load(path: str, digest: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    value, actual, source = load_sha_bound_project_checkpoint_mapping(
        path, expected_sha256=digest, map_location="cpu", label=label
    )
    return dict(value), {"path": str(source), "sha256": actual}


def unique_view_topk(
    scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    proposal_views: torch.Tensor,
    maximum: int,
) -> torch.Tensor:
    """Return score-ranked proposals with at most one proposal per view."""

    order = torch.argsort(scores, descending=True, stable=True)
    selected: list[int] = []
    used_views: set[int] = set()
    for position in order.tolist():
        proposal = int(candidate_indices[position])
        view = int(proposal_views[proposal])
        if view in used_views:
            continue
        selected.append(proposal)
        used_views.add(view)
        if len(selected) == int(maximum):
            break
    return torch.tensor(selected, dtype=torch.long)


def transferred_support_iou(
    *,
    candidate_supports: list[torch.Tensor],
    target_support: torch.Tensor,
    visible: torch.Tensor,
    num_rows: int,
) -> float:
    if not candidate_supports:
        return 0.0
    del num_rows
    visible_set = set(torch.as_tensor(visible).long().tolist())
    target = set(torch.as_tensor(target_support).long().tolist()).intersection(
        visible_set
    )
    predicted: set[int] = set()
    for support in candidate_supports:
        predicted.update(torch.as_tensor(support).long().tolist())
    predicted.intersection_update(visible_set)
    intersection = len(target.intersection(predicted))
    union = len(target.union(predicted))
    return float(intersection / union) if union else 1.0


def _evaluate_queries(
    *,
    query_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    features: torch.Tensor,
    support_sets: list[set[int]],
    proposal_views: torch.Tensor,
    visible_sets: dict[int, set[int]],
    topk: int,
) -> tuple[float, list[float]]:
    values: list[float] = []
    for query in query_indices.tolist():
        scores = features.index_select(0, candidate_indices) @ features[query]
        selected = unique_view_topk(
            scores, candidate_indices, proposal_views, maximum=int(topk)
        )
        visible = visible_sets[int(proposal_views[query])]
        target = support_sets[query].intersection(visible)
        predicted: set[int] = set()
        for index in selected.tolist():
            predicted.update(support_sets[index])
        predicted.intersection_update(visible)
        intersection = len(target.intersection(predicted))
        union = len(target.union(predicted))
        values.append(float(intersection / union) if union else 1.0)
    if not values:
        raise ValueError("source split has no evaluable proposal")
    return float(torch.tensor(values).mean()), values


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    membership, membership_record = _load(
        args.membership, args.expected_membership_sha256, "source SAM membership"
    )
    language, language_record = _load(
        args.language_teacher,
        args.expected_language_teacher_sha256,
        "native source language teacher",
    )
    appearance, appearance_record = _load(
        args.appearance_teacher,
        args.expected_appearance_teacher_sha256,
        "native source appearance teacher",
    )
    metadata = membership.get("metadata", {})
    language_meta = language.get("metadata", {})
    appearance_meta = appearance.get("metadata", {})
    if (
        metadata.get("benchmark_masks_opened") is not False
        or language_meta.get("source_only") is not True
        or language_meta.get("benchmark_masks_opened") is not False
        or appearance_meta.get("source_only") is not True
        or appearance_meta.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("source proposal information policy differs")
    rows = torch.as_tensor(membership.get("row_indices")).long()
    proposals = torch.as_tensor(membership.get("proposal_indices")).long()
    weights = torch.as_tensor(membership.get("weights")).float()
    proposal_views = torch.as_tensor(membership.get("proposal_view_indices")).long()
    view_observed = torch.as_tensor(membership.get("view_observed")).bool()
    proposal_count = int(membership.get("num_proposals", -1))
    supports = _proposal_support(
        rows[weights >= float(args.membership_threshold)],
        proposals[weights >= float(args.membership_threshold)],
        proposal_count,
    )
    support_sets = [set(support.tolist()) for support in supports]
    visible_sets = {
        view: set(torch.where(view_observed[view])[0].tolist())
        for view in range(int(view_observed.shape[0]))
    }
    valid = torch.tensor([support.numel() > 0 for support in supports])
    descriptor = F.normalize(torch.as_tensor(language.get("descriptors")).float(), dim=-1)
    context = F.normalize(
        torch.as_tensor(language.get("context_descriptors")).float(), dim=-1
    )
    semantic = F.normalize(0.75 * descriptor + 0.25 * context, dim=-1)
    physical = F.normalize(torch.as_tensor(appearance.get("descriptors")).float(), dim=-1)
    if (
        semantic.shape != (proposal_count, 1536)
        or physical.shape[0] != proposal_count
        or proposal_views.shape != (proposal_count,)
        or view_observed.ndim != 2
    ):
        raise ValueError("source proposal teacher axes differ")
    features = {
        "native_siglip2": semantic,
        "native_dinov2": physical,
        "native_siglip2_plus_dinov2": compose_membership_query_features(
            semantic, physical
        ),
    }
    residues = torch.remainder(proposal_views, int(args.split_stride))
    candidates = torch.where(
        valid
        & (residues != int(args.calibration_residue))
        & (residues != int(args.evaluation_residue))
    )[0]
    calibration = torch.where(valid & (residues == int(args.calibration_residue)))[0]
    evaluation = torch.where(valid & (residues == int(args.evaluation_residue)))[0]
    if min(candidates.numel(), calibration.numel(), evaluation.numel()) <= 0:
        raise ValueError("fixed source split has an empty cohort")
    topks = tuple(int(value) for value in args.topk.split(","))
    if not topks or min(topks) <= 0:
        raise ValueError("top-k candidates must be positive")
    calibration_table: dict[str, dict[str, float]] = {}
    ranked: list[tuple[float, int, str]] = []
    for mode, feature in features.items():
        calibration_table[mode] = {}
        for topk in topks:
            macro, _values = _evaluate_queries(
                query_indices=calibration,
                candidate_indices=candidates,
                features=feature,
                support_sets=support_sets,
                proposal_views=proposal_views,
                visible_sets=visible_sets,
                topk=topk,
            )
            calibration_table[mode][str(topk)] = macro
            ranked.append((macro, -topk, mode))
    _macro, negative_topk, selected_mode = max(ranked)
    selected_topk = -int(negative_topk)
    evaluation_iou, per_proposal = _evaluate_queries(
        query_indices=evaluation,
        candidate_indices=candidates,
        features=features[selected_mode],
        support_sets=support_sets,
        proposal_views=proposal_views,
        visible_sets=visible_sets,
        topk=selected_topk,
    )
    semantic_iou, _ = _evaluate_queries(
        query_indices=evaluation,
        candidate_indices=candidates,
        features=features["native_siglip2"],
        support_sets=support_sets,
        proposal_views=proposal_views,
        visible_sets=visible_sets,
        topk=selected_topk,
    )
    passed = evaluation_iou >= float(args.minimum_iou) and evaluation_iou >= (
        semantic_iou + float(args.minimum_gain)
    )
    output = {
        "schema": "radio_gs.source_proposal_retrieval_membership_gate.v1",
        "schema_version": 1,
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "method": {
            "candidate_pool": "disjoint_source_views_only",
            "one_proposal_per_candidate_view": True,
            "membership_composition": "hard_exact_mpr_union",
            "descriptor_selection": "calibration_residue_macro_iou_then_smaller_topk",
            "selected_mode": selected_mode,
            "selected_topk": selected_topk,
            "membership_threshold": float(args.membership_threshold),
        },
        "cohorts": {
            "candidate_proposals": int(candidates.numel()),
            "calibration_proposals": int(calibration.numel()),
            "evaluation_proposals": int(evaluation.numel()),
            "split": (
                f"view_mod_{args.split_stride}: candidates=other, "
                f"calibration={args.calibration_residue}, "
                f"evaluation={args.evaluation_residue}"
            ),
        },
        "calibration_macro_iou": calibration_table,
        "evaluation": {
            "selected_macro_iou": evaluation_iou,
            "native_siglip2_same_topk_macro_iou": semantic_iou,
            "delta_over_native_siglip2": evaluation_iou - semantic_iou,
            "minimum_proposal_iou": min(per_proposal),
            "median_proposal_iou": float(torch.tensor(per_proposal).median()),
        },
        "gate": {
            "minimum_iou": float(args.minimum_iou),
            "minimum_gain_over_native_siglip2": float(args.minimum_gain),
            "passed": passed,
        },
        "access_audit": {
            "benchmark_queries_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
        },
        "inputs": {
            "membership": membership_record,
            "language_teacher": language_record,
            "appearance_teacher": appearance_record,
        },
    }
    write_frozen_json(Path(args.output).expanduser().resolve(), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--language-teacher", required=True)
    parser.add_argument("--expected-language-teacher-sha256", required=True)
    parser.add_argument("--appearance-teacher", required=True)
    parser.add_argument("--expected-appearance-teacher-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--membership-threshold", type=float, default=0.5)
    parser.add_argument("--split-stride", type=int, default=4)
    parser.add_argument("--calibration-residue", type=int, default=2)
    parser.add_argument("--evaluation-residue", type=int, default=3)
    parser.add_argument("--topk", default="1,2,4,8")
    parser.add_argument("--minimum-iou", type=float, default=0.2)
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    print(json.dumps(evaluate(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
