#!/usr/bin/env python3
"""Evaluate native SAM3+SigLIP2 region identity on a ScanNet sentinel.

Official query-free source SAM3 proposals provide extent.  Independent native
SigLIP2 masked/context crops provide categorical identity.  Proposal evidence
is averaged once per source view before it is fused with the unchanged
primitive score, preventing views with more nested masks from receiving more
authority.  Rows without two-view region support, and structural rows, replay
the primitive prediction exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.evaluate_scannet_object_aware_category_vote_cpu import _metrics
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


TEACHER_SCHEMA = "radio_gs.multiscale_sam_mask_aligned_crop_summary_teacher.v2"
STRUCTURAL_IDS = frozenset({1, 2, 22})


def _load_torch(path: Path, label: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not one mapping")
    return dict(value)


def _text_embeddings(path: Path, class_ids: tuple[int, ...]) -> torch.Tensor:
    value = _load_torch(path, "native SigLIP2 text cache")
    expected = [NYU40_ID_TO_NAME[int(class_id)] for class_id in class_ids]
    if value.get("model") != "google/siglip2-giant-opt-patch16-384":
        raise ValueError("text cache is not the paired native SigLIP2 model")
    if list(value.get("queries", [])) != expected:
        raise ValueError("text cache category order differs from ScanNet split")
    embedding = F.normalize(torch.as_tensor(value.get("embeddings")).float(), dim=-1)
    if embedding.shape != (len(class_ids), 1536):
        raise ValueError("native SigLIP2 text embedding shape differs")
    return embedding


def _region_per_view(
    *,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    proposal_views: torch.Tensor,
    proposal_scores: torch.Tensor,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_views = int(proposal_views.max()) + 1
    classes = int(proposal_scores.shape[1])
    entry_views = proposal_views[proposal_indices]
    flat = row_indices * num_views + entry_views
    score_sum = torch.zeros(num_rows * num_views, classes, dtype=torch.float32)
    mass = torch.zeros(num_rows * num_views, dtype=torch.float32)
    score_sum.index_add_(
        0,
        flat,
        weights[:, None] * proposal_scores.index_select(0, proposal_indices),
    )
    mass.index_add_(0, flat, weights)
    valid = mass > 0
    score_sum[valid] /= mass[valid, None]
    score_sum = score_sum.reshape(num_rows, num_views, classes)
    valid = valid.reshape(num_rows, num_views)
    view_count = valid.sum(dim=1)
    region = (score_sum * valid[:, :, None]).sum(dim=1) / view_count.clamp_min(1)[:, None]
    view_label = score_sum.argmax(dim=-1)
    agreement = torch.zeros(num_rows, dtype=torch.float32)
    for class_index in range(classes):
        count = ((view_label == class_index) & valid).sum(dim=1)
        agreement = torch.maximum(agreement, count.float())
    agreement /= view_count.clamp_min(1).float()
    return region, view_count, agreement


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).expanduser().resolve(strict=True)
    teacher_path = Path(args.proposal_teacher).expanduser().resolve(strict=True)
    score_path = Path(args.score_cache).expanduser().resolve(strict=True)
    membership = _load_torch(membership_path, "native SAM3 membership")
    teacher = _load_torch(teacher_path, "native SigLIP2 proposal teacher")
    if teacher.get("schema") != TEACHER_SCHEMA:
        raise ValueError("proposal teacher is not independent native SigLIP2")
    teacher_meta = teacher.get("metadata", {})
    membership_meta = membership.get("metadata", {})
    if (
        teacher_meta.get("source_only") is not True
        or teacher_meta.get("benchmark_masks_opened") is not False
        or teacher_meta.get("encoder_binding", {}).get("backend")
        != "native_siglip2_vision"
        or membership_meta.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("native region information policy differs")
    rows = torch.as_tensor(membership.get("row_indices")).long()
    proposals = torch.as_tensor(membership.get("proposal_indices")).long()
    weights = torch.as_tensor(membership.get("weights")).float().clamp_min(0)
    proposal_views = torch.as_tensor(membership.get("proposal_view_indices")).long()
    descriptors = F.normalize(torch.as_tensor(teacher.get("descriptors")).float(), dim=-1)
    contexts = F.normalize(
        torch.as_tensor(teacher.get("context_descriptors")).float(), dim=-1
    )
    descriptor = F.normalize(0.75 * descriptors + 0.25 * contexts, dim=-1)
    proposal_count = int(membership.get("num_proposals", -1))
    num_rows = int(membership.get("num_rows", -1))
    if (
        proposal_count <= 0
        or num_rows <= 0
        or descriptor.shape != (proposal_count, 1536)
        or proposal_views.shape != (proposal_count,)
        or rows.shape != proposals.shape
        or rows.shape != weights.shape
        or int(rows.min()) < 0
        or int(rows.max()) >= num_rows
        or int(proposals.min()) < 0
        or int(proposals.max()) >= proposal_count
    ):
        raise ValueError("native proposal and Gaussian axes differ")
    cached = np.load(score_path, allow_pickle=False)
    if len(cached["pseudo_labels"]) != num_rows:
        raise ValueError("ScanNet score and native membership Gaussian axes differ")
    blend_weights = tuple(float(value) for value in args.blend_weights.split(","))
    if any(value < 0 or value > 1 for value in blend_weights):
        raise ValueError("blend weights must lie in [0,1]")
    reports: dict[str, Any] = {}
    for split in ("19", "15", "10"):
        class_ids = tuple(int(value) for value in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])
        text = _text_embeddings(Path(getattr(args, f"text_cache_{split}")), class_ids)
        proposal_cosine = descriptor @ text.T
        region, view_count, agreement = _region_per_view(
            row_indices=rows,
            proposal_indices=proposals,
            weights=weights,
            proposal_views=proposal_views,
            proposal_scores=proposal_cosine,
            num_rows=num_rows,
        )
        primitive = torch.from_numpy(cached[f"scores_split_{split}"]).float()
        primitive_centered = F.normalize(
            primitive - primitive.mean(dim=-1, keepdim=True), dim=-1, eps=1e-8
        )
        region_centered = F.normalize(
            region - region.mean(dim=-1, keepdim=True), dim=-1, eps=1e-8
        )
        primitive_labels = torch.tensor(class_ids)[primitive.argmax(dim=-1)]
        eligible = (
            (view_count >= int(args.minimum_views))
            & (agreement >= float(args.minimum_view_agreement))
            & ~torch.isin(primitive_labels, torch.tensor(sorted(STRUCTURAL_IDS)))
        )
        baseline_labels = np.asarray(class_ids)[primitive.argmax(dim=-1).numpy()]
        baseline = _metrics(
            cached["pseudo_labels"], baseline_labels, cached["significance"], class_ids
        )
        candidates: dict[str, Any] = {}
        for alpha in blend_weights:
            mixed = primitive_centered.clone()
            mixed[eligible] = F.normalize(
                (1.0 - alpha) * primitive_centered[eligible]
                + alpha * agreement[eligible, None] * region_centered[eligible],
                dim=-1,
                eps=1e-8,
            )
            labels = np.asarray(class_ids)[mixed.argmax(dim=-1).numpy()]
            metric = _metrics(
                cached["pseudo_labels"], labels, cached["significance"], class_ids
            )
            candidates[f"alpha_{alpha:g}"] = {
                "metrics": metric,
                "delta": {
                    key: float(metric[key] - baseline[key]) for key in ("miou", "macc")
                },
                "changed_rows": int(np.count_nonzero(labels != baseline_labels)),
            }
        reports[split] = {
            "baseline": baseline,
            "eligible_rows": int(eligible.sum()),
            "rows_with_region_support": int((view_count > 0).sum()),
            "rows_with_multiview_support": int((view_count >= 2).sum()),
            "mean_view_count_supported": float(view_count[view_count > 0].float().mean())
            if bool((view_count > 0).any())
            else 0.0,
            "candidates": candidates,
        }
    output = {
        "schema": "radio_gs.scannet_native_sam_siglip_region_vote.v1",
        "schema_version": 1,
        "scene": str(args.scene),
        "status": "development_diagnostic_complete_not_promoted",
        "method": {
            "identity": "independent_native_siglip2_masked_context_crop",
            "extent": "official_native_sam3_exact_marginal_membership",
            "aggregation": "equal_source_view_then_class_symmetric_centered_fusion",
            "minimum_views": int(args.minimum_views),
            "minimum_view_agreement": float(args.minimum_view_agreement),
            "structural_primitive_replay_ids": sorted(STRUCTURAL_IDS),
            "blend_weights_preregistered_diagnostic": list(blend_weights),
            "per_class_or_scene_parameter": False,
        },
        "access_audit": {
            "source_rgb_used_for_native_teachers": True,
            "benchmark_masks_used_for_prediction": False,
            "benchmark_labels_used_for_training_or_threshold": False,
            "benchmark_metrics_opened_only_by_this_development_evaluator": True,
        },
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "proposal_teacher": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)},
            "score_cache": {"path": str(score_path), "sha256": sha256_file(score_path)},
        },
        "splits": reports,
    }
    write_frozen_json(Path(args.output).expanduser().resolve(), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--proposal-teacher", required=True)
    parser.add_argument("--score-cache", required=True)
    parser.add_argument("--text-cache-19", required=True)
    parser.add_argument("--text-cache-15", required=True)
    parser.add_argument("--text-cache-10", required=True)
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--minimum-view-agreement", type=float, default=0.5)
    parser.add_argument("--blend-weights", default="0.25,0.5,1.0")
    parser.add_argument("--output", required=True)
    result = evaluate(parser.parse_args())
    print(json.dumps(result["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
