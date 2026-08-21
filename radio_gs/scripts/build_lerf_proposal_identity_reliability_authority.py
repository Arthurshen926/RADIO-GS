#!/usr/bin/env python3
"""Materialize source-only per-proposal LERF identity reliability authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from radio_gs.querying.latent_proposal_posterior import (
    DIFFERENT_RELATION,
    SAME_RELATION,
    UNKNOWN_RELATION,
    latent_proposal_null_posterior,
)
from radio_gs.scripts.build_lerf_sam_siglip_object_posterior_scores import (
    _score_embeddings,
)
from radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores import (
    _select_embedding_rows,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import vala_knn_minmax_scores
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_proposal_identity_reliability_authority.v1"


def ternary_cross_view_edges(
    support_sets: list[set[int]],
    proposal_views: torch.Tensor,
    proposal_view_visibility: torch.Tensor,
    *,
    minimum_jaccard: float = 0.02,
    minimum_overlap: float = 0.15,
    minimum_cross_visibility: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return upper-triangular cross-view ternary edges and same strength."""

    views = torch.as_tensor(proposal_views).long().cpu()
    visibility = torch.as_tensor(proposal_view_visibility).float().cpu()
    count = len(support_sets)
    if views.shape != (count,) or visibility.ndim != 2 or visibility.shape[0] != count:
        raise ValueError("proposal visibility axes differ")
    lefts: list[int] = []
    rights: list[int] = []
    labels: list[int] = []
    strengths: list[float] = []
    for left in range(count):
        for right in range(left + 1, count):
            lv, rv = int(views[left]), int(views[right])
            if lv == rv:
                continue
            a, b = support_sets[left], support_sets[right]
            intersection = len(a.intersection(b)) if a and b else 0
            union = len(a) + len(b) - intersection
            jaccard = intersection / max(union, 1)
            overlap = intersection / max(min(len(a), len(b)), 1)
            strength = max(jaccard, overlap)
            if jaccard >= float(minimum_jaccard) or overlap >= float(minimum_overlap):
                relation = SAME_RELATION
            elif (
                intersection == 0
                and float(visibility[left, rv]) >= float(minimum_cross_visibility)
                and float(visibility[right, lv]) >= float(minimum_cross_visibility)
            ):
                relation = DIFFERENT_RELATION
            else:
                relation = UNKNOWN_RELATION
            lefts.append(left); rights.append(right); labels.append(relation); strengths.append(strength)
    return (
        torch.tensor(lefts, dtype=torch.long),
        torch.tensor(rights, dtype=torch.long),
        torch.tensor(labels, dtype=torch.int8),
        torch.tensor(strengths, dtype=torch.float32),
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"authority exists: {output}")
    score_path = Path(args.latent_score_cache).expanduser().resolve()
    membership_path = Path(args.membership_cache).expanduser().resolve()
    teacher_path = Path(args.proposal_teacher).expanduser().resolve()
    text_path = Path(args.text_embedding_cache).expanduser().resolve()
    canonical_path = Path(args.canonical_embedding_cache).expanduser().resolve()
    score = torch.load(score_path, map_location="cpu", weights_only=False)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    metadata = dict(score.get("metadata", {}))
    query_names = [str(value) for value in metadata.get("query_names", [])]
    xyz = torch.as_tensor(score.get("xyz")).float().cpu()
    valid_rows = torch.as_tensor(score.get("valid")).bool().cpu()
    raw_identity = torch.as_tensor(score.get("identity_query_scores")).float().cpu()
    rows = torch.as_tensor(membership.get("row_indices")).long().cpu()
    props = torch.as_tensor(membership.get("proposal_indices")).long().cpu()
    weights = torch.as_tensor(membership.get("weights")).float().cpu()
    views = torch.as_tensor(membership.get("proposal_view_indices")).long().cpu()
    observed = torch.as_tensor(membership.get("view_observed")).bool().cpu()
    proposal_count = int(membership.get("num_proposals", -1))
    if (
        not query_names
        or raw_identity.shape != (xyz.shape[0], len(query_names))
        or valid_rows.shape != (xyz.shape[0],)
        or views.shape != (proposal_count,)
        or observed.shape[1] != xyz.shape[0]
        or rows.shape != props.shape
        or rows.shape != weights.shape
        or metadata.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("source-only score/membership identity differs")
    descriptors = torch.as_tensor(teacher.get("descriptors")).float().cpu()
    contexts = torch.as_tensor(teacher.get("context_descriptors")).float().cpu()
    if descriptors.shape != (proposal_count, 1536) or contexts.shape != descriptors.shape:
        raise ValueError("proposal teacher axes differ")
    text_payload = torch.load(text_path, map_location="cpu", weights_only=False)
    canonical_payload = torch.load(canonical_path, map_location="cpu", weights_only=False)
    text = torch.nn.functional.normalize(_select_embedding_rows(text_payload, query_names), dim=-1)
    canonical = torch.nn.functional.normalize(torch.as_tensor(canonical_payload["embeddings"]).float(), dim=-1)
    device = torch.device(args.device)
    descriptor_score = _score_embeddings(descriptors, text, canonical, device=device, chunk_size=8192)
    context_score = _score_embeddings(contexts, text, canonical, device=device, chunk_size=8192)

    raw = raw_identity.clone(); raw[~valid_rows] = -1e4
    base = vala_knn_minmax_scores(raw, xyz, k=10, chunk_size=int(args.knn_chunk_size), valid_mask=valid_rows)
    base[~valid_rows] = -1e4
    keep = (rows >= 0) & (rows < xyz.shape[0]) & (props >= 0) & (props < proposal_count) & (weights > 0)
    rows, props, weights = rows[keep], props[keep], weights[keep]
    proposal_max = torch.zeros(proposal_count); proposal_max.scatter_reduce_(0, props, weights, reduce="amax", include_self=True)
    conditional = weights / proposal_max[props].clamp_min(1e-8)
    support_sets: list[set[int]] = [set() for _ in range(proposal_count)]
    support_rows: list[list[int]] = [[] for _ in range(proposal_count)]
    for row, prop in zip(rows[conditional >= 0.5].tolist(), props[conditional >= 0.5].tolist()):
        support_sets[prop].add(row)
    proposal_visibility = torch.zeros((proposal_count, observed.shape[0]), dtype=torch.float32)
    for prop, support in enumerate(support_sets):
        if support:
            selected = torch.tensor(sorted(support), dtype=torch.long)
            proposal_visibility[prop] = observed[:, selected].float().mean(dim=1)
    edge_left, edge_right, edge_relation, edge_strength = ternary_cross_view_edges(
        support_sets, views, proposal_visibility
    )
    same_count = torch.zeros(proposal_count, dtype=torch.long)
    different_count = torch.zeros_like(same_count)
    unknown_count = torch.zeros_like(same_count)
    same_max = torch.zeros(proposal_count)
    for label, target in ((SAME_RELATION, same_count), (DIFFERENT_RELATION, different_count), (UNKNOWN_RELATION, unknown_count)):
        mask = edge_relation == label
        target.index_add_(0, edge_left[mask], torch.ones(int(mask.sum()), dtype=torch.long))
        target.index_add_(0, edge_right[mask], torch.ones(int(mask.sum()), dtype=torch.long))
    same_mask = edge_relation == SAME_RELATION
    if bool(same_mask.any()):
        same_max.scatter_reduce_(0, edge_left[same_mask], edge_strength[same_mask], reduce="amax", include_self=True)
        same_max.scatter_reduce_(0, edge_right[same_mask], edge_strength[same_mask], reduce="amax", include_self=True)

    queries = len(query_names)
    peaks, peak_rows = base.max(dim=0)
    field_tail = torch.zeros((proposal_count, queries)); core_fraction = torch.zeros_like(field_tail); peak_membership = torch.zeros_like(field_tail)
    mass = torch.zeros(proposal_count); mass.index_add_(0, props, weights)
    for q in range(queries):
        field_tail[:, q].scatter_reduce_(0, props, base[rows, q], reduce="amax", include_self=True)
        core = base[rows, q] >= peaks[q] * 0.8
        numerator = torch.zeros(proposal_count); numerator.index_add_(0, props, weights * core.float())
        core_fraction[:, q] = numerator / mass.clamp_min(1e-8)
        exact = rows == peak_rows[q]
        if bool(exact.any()):
            peak_membership[:, q].scatter_reduce_(0, props[exact], conditional[exact], reduce="amax", include_self=True)
    quality = torch.as_tensor(membership.get("proposal_scores")).float().clamp(0, 1)
    identity = (0.55*descriptor_score + 0.20*field_tail + 0.15*core_fraction.sqrt() + 0.10*peak_membership + 0.05*(descriptor_score-context_score).clamp_min(0)) * (0.75+0.25*quality[:,None])
    area = torch.as_tensor(membership.get("proposal_area_fraction")).float()
    proposal_valid = (area[:,None] <= 0.25) & (descriptor_score >= 0.55) & (same_max[:,None] > 0) & (peak_membership > 0)
    logits = 8.0 * (identity - 0.55) + torch.log(same_max.clamp_min(1e-8))[:,None]
    counts = proposal_valid.sum(dim=0)
    logits = logits - torch.log(counts.clamp_min(1).float())[None]
    posterior = latent_proposal_null_posterior(
        torch.zeros_like(base).clamp(0,1), rows, props, conditional.clamp(0,1), logits, torch.zeros(queries), proposal_valid=proposal_valid
    )
    payload = {
        "schema": SCHEMA, "schema_version": 1, "scene": str(args.scene),
        "query_names": query_names, "proposal_view_indices": views,
        "edge_left": edge_left, "edge_right": edge_right, "edge_relation": edge_relation, "edge_same_strength": edge_strength,
        "proposal_view_visibility": proposal_visibility,
        "same_edge_count": same_count, "different_edge_count": different_count, "unknown_edge_count": unknown_count,
        "same_strength_max": same_max,
        "descriptor_score": descriptor_score, "context_score": context_score,
        "field_tail": field_tail, "core_fraction": core_fraction, "peak_membership": peak_membership,
        "identity_score": identity, "proposal_valid": proposal_valid, "proposal_logits": logits,
        "proposal_probability": posterior.proposal_probability, "null_probability": posterior.null_probability,
        "metadata": {
            "source_only": True, "benchmark_masks_opened": False, "evaluation_rgb_opened": False,
            "ternary_relation": {"same": SAME_RELATION, "different": DIFFERENT_RELATION, "unknown": UNKNOWN_RELATION},
            "different_requires": "zero_overlap_and_bidirectional_cross_visibility_ge_0.5",
            "same_requires": "jaccard_ge_0.02_or_overlap_ge_0.15",
            "membership_floor": 0.5,
            "latent_score_cache": {"path": str(score_path), "sha256": sha256_file(score_path)},
            "membership_cache": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "proposal_teacher": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)},
            "text_embedding_cache": {"path": str(text_path), "sha256": sha256_file(text_path)},
            "canonical_embedding_cache": {"path": str(canonical_path), "sha256": sha256_file(canonical_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp"); torch.save(payload, temporary); os.replace(temporary, output)
    relation_counts = {"same": int((edge_relation==SAME_RELATION).sum()), "different": int((edge_relation==DIFFERENT_RELATION).sum()), "unknown": int((edge_relation==UNKNOWN_RELATION).sum())}
    report = {"schema": SCHEMA, "status": "complete", "scene": str(args.scene), "proposals": proposal_count, "queries": queries, "relations": relation_counts, "valid_proposal_counts": counts.tolist(), "output": str(output), "output_sha256": sha256_file(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True); parser.add_argument("--latent-score-cache", required=True)
    parser.add_argument("--membership-cache", required=True); parser.add_argument("--proposal-teacher", required=True)
    parser.add_argument("--text-embedding-cache", required=True); parser.add_argument("--canonical-embedding-cache", required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--knn-chunk-size", type=int, default=8192); parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
