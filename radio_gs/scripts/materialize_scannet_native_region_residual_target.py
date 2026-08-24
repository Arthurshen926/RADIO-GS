#!/usr/bin/env python3
"""Materialize a source-only native-region residual target for frozen L512.

Official SAM3 proposal memberships define extent and paired native SigLIP2
masked/context descriptors define identity.  Proposal descriptors are first
averaged within each source view, then equally across views.  Only rows with
two-view support and fixed descriptor agreement receive the alpha=0.25 region
residual; every other row replays the deployed primitive descriptor exactly.
The artifact is a training teacher, not an additional deployment field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


TEACHER_SCHEMA = "radio_gs.multiscale_sam_mask_aligned_crop_summary_teacher.v2"


def aggregate_sparse_view_descriptors(
    *,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    proposal_views: torch.Tensor,
    proposal_descriptors: torch.Tensor,
    num_rows: int,
    minimum_views: int = 2,
    minimum_view_cosine: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return equal-view descriptor, validity, view count and min agreement."""

    rows = torch.as_tensor(row_indices).long().cpu().reshape(-1)
    proposals = torch.as_tensor(proposal_indices).long().cpu().reshape(-1)
    mass = torch.as_tensor(weights).float().cpu().reshape(-1).clamp_min(0)
    views = torch.as_tensor(proposal_views).long().cpu().reshape(-1)
    descriptors = F.normalize(
        torch.as_tensor(proposal_descriptors).float().cpu(), dim=-1, eps=1e-8
    )
    if (
        rows.shape != proposals.shape
        or rows.shape != mass.shape
        or descriptors.ndim != 2
        or views.shape != (descriptors.shape[0],)
        or int(num_rows) <= 0
        or int(minimum_views) < 1
        or not 0.0 <= float(minimum_view_cosine) <= 1.0
        or (rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(num_rows)))
        or (proposals.numel() and (
            int(proposals.min()) < 0 or int(proposals.max()) >= descriptors.shape[0]
        ))
    ):
        raise ValueError("native region sparse axes or constants differ")
    dimension = int(descriptors.shape[1])
    descriptor_sum = torch.zeros(int(num_rows), dimension, dtype=torch.float32)
    first_descriptor = torch.zeros_like(descriptor_sum)
    view_count = torch.zeros(int(num_rows), dtype=torch.int16)
    minimum_cosine = torch.ones(int(num_rows), dtype=torch.float32)
    entry_views = views.index_select(0, proposals)
    for view in sorted(int(value) for value in torch.unique(views)):
        selected = torch.where(entry_views == int(view))[0]
        if not selected.numel():
            continue
        selected_rows = rows.index_select(0, selected)
        unique_rows, inverse = torch.unique(selected_rows, sorted=True, return_inverse=True)
        numerator = torch.zeros(unique_rows.numel(), dimension, dtype=torch.float32)
        denominator = torch.zeros(unique_rows.numel(), dtype=torch.float32)
        selected_mass = mass.index_select(0, selected)
        numerator.index_add_(
            0,
            inverse,
            selected_mass[:, None]
            * descriptors.index_select(0, proposals.index_select(0, selected)),
        )
        denominator.index_add_(0, inverse, selected_mass)
        positive = denominator > 0
        unique_rows = unique_rows[positive]
        current = F.normalize(
            numerator[positive] / denominator[positive, None], dim=-1, eps=1e-8
        )
        prior_count = view_count.index_select(0, unique_rows)
        repeated = prior_count > 0
        if bool(repeated.any()):
            repeated_rows = unique_rows[repeated]
            cosine = (
                current[repeated]
                * first_descriptor.index_select(0, repeated_rows)
            ).sum(dim=-1)
            minimum_cosine[repeated_rows] = torch.minimum(
                minimum_cosine.index_select(0, repeated_rows), cosine
            )
        first = ~repeated
        if bool(first.any()):
            first_descriptor[unique_rows[first]] = current[first]
        descriptor_sum.index_add_(0, unique_rows, current)
        view_count[unique_rows] += 1
    descriptor = F.normalize(
        descriptor_sum / view_count.clamp_min(1).float()[:, None], dim=-1, eps=1e-8
    )
    valid = (view_count >= int(minimum_views)) & (
        minimum_cosine >= float(minimum_view_cosine)
    )
    return descriptor, valid, view_count, minimum_cosine


def build(args: argparse.Namespace) -> dict[str, Any]:
    membership, membership_sha, membership_path = (
        load_sha_bound_project_checkpoint_mapping(
            args.membership,
            expected_sha256=args.expected_membership_sha256,
            map_location="cpu",
            label="native SAM3 membership",
        )
    )
    teacher, teacher_sha, teacher_path = load_sha_bound_project_checkpoint_mapping(
        args.proposal_teacher,
        expected_sha256=args.expected_proposal_teacher_sha256,
        map_location="cpu",
        label="native SigLIP2 proposal teacher",
    )
    baseline, baseline_sha, baseline_path = load_sha_bound_project_checkpoint_mapping(
        args.baseline_query_cache,
        expected_sha256=args.expected_baseline_query_cache_sha256,
        map_location="cpu",
        label="deployed primitive query cache",
    )
    if (
        teacher.get("schema") != TEACHER_SCHEMA
        or teacher.get("metadata", {}).get("source_only") is not True
        or teacher.get("metadata", {}).get("benchmark_masks_opened") is not False
        or membership.get("metadata", {}).get("benchmark_masks_opened") is not False
    ):
        raise ValueError("native source teacher information policy differs")
    descriptors = F.normalize(torch.as_tensor(teacher.get("descriptors")).float(), dim=-1)
    contexts = F.normalize(
        torch.as_tensor(teacher.get("context_descriptors")).float(), dim=-1
    )
    proposal_descriptor = F.normalize(
        0.75 * descriptors + 0.25 * contexts, dim=-1, eps=1e-8
    )
    num_rows = int(membership.get("num_rows", -1))
    if (
        proposal_descriptor.shape != (int(membership.get("num_proposals", -1)), 1536)
        or not torch.equal(
            torch.as_tensor(membership.get("proposal_view_indices")).long(),
            torch.as_tensor(teacher.get("proposal_view_indices")).long(),
        )
    ):
        raise ValueError("native SAM and SigLIP proposal identities differ")
    region, valid, view_count, agreement = aggregate_sparse_view_descriptors(
        row_indices=torch.as_tensor(membership.get("row_indices")),
        proposal_indices=torch.as_tensor(membership.get("proposal_indices")),
        weights=torch.as_tensor(membership.get("weights")),
        proposal_views=torch.as_tensor(membership.get("proposal_view_indices")),
        proposal_descriptors=proposal_descriptor,
        num_rows=num_rows,
        minimum_views=int(args.minimum_views),
        minimum_view_cosine=float(args.minimum_view_cosine),
    )
    xyz = torch.as_tensor(baseline.get("xyz")).float().cpu().contiguous()
    primitive = F.normalize(
        torch.as_tensor(
            baseline.get("summary_features", baseline.get("features"))
        ).float(),
        dim=-1,
        eps=1e-8,
    )
    if xyz.shape != (num_rows, 3) or primitive.shape != (num_rows, 1536):
        raise ValueError("native region and deployed primitive row domains differ")
    alpha = float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("native region residual alpha must lie in [0,1]")
    target = primitive.clone()
    target[valid] = F.normalize(
        (1.0 - alpha) * primitive[valid] + alpha * region[valid],
        dim=-1,
        eps=1e-8,
    )
    payload = {
        "xyz": xyz,
        "features": target.half().contiguous(),
        "valid": valid.contiguous(),
        "view_count": view_count.contiguous(),
        "minimum_view_cosine": agreement.contiguous(),
        "metadata": {
            "schema": "radio_gs.scannet_native_region_residual_target.v1",
            "scene": str(args.scene),
            "source_only": True,
            "teacher_order": "native_sam_extent_then_native_siglip_region_then_equal_view",
            "residual_alpha": alpha,
            "minimum_views": int(args.minimum_views),
            "minimum_view_cosine": float(args.minimum_view_cosine),
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_queries_opened": False,
            "inputs": {
                "membership": {"path": str(membership_path), "sha256": membership_sha},
                "proposal_teacher": {"path": str(teacher_path), "sha256": teacher_sha},
                "baseline_query_cache": {"path": str(baseline_path), "sha256": baseline_sha},
            },
        },
    }
    output = Path(args.output).expanduser().resolve()
    write_torch_noclobber(output, payload)
    report = {
        "status": "complete_source_only_teacher",
        "scene": str(args.scene),
        "valid_rows": int(valid.sum()),
        "total_rows": num_rows,
        "valid_fraction": float(valid.float().mean()),
        "rows_with_two_views": int((view_count >= 2).sum()),
        "output": file_record(output),
        "inputs": payload["metadata"]["inputs"],
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--proposal-teacher", required=True)
    parser.add_argument("--expected-proposal-teacher-sha256", required=True)
    parser.add_argument("--baseline-query-cache", required=True)
    parser.add_argument("--expected-baseline-query-cache-sha256", required=True)
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--minimum-view-cosine", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    report = build(parser.parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
