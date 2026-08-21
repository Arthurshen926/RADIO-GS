#!/usr/bin/env python3
"""Build conservative source-only cross-view track/null/occlusion authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
import numpy as np
from scipy import sparse

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_source_geometry_cycle_track_authority.v1"


def reciprocal_cycle_labels(
    geometry_strength: torch.Tensor,
    proposal_views: torch.Tensor,
    intersection: torch.Tensor,
    cross_visibility: torch.Tensor,
) -> torch.Tensor:
    """Return same=1, different=0, unknown=-1 without target thresholds."""

    strength = torch.as_tensor(geometry_strength).float()
    views = torch.as_tensor(proposal_views).long()
    intersect = torch.as_tensor(intersection).long()
    visibility = torch.as_tensor(cross_visibility).float()
    count = int(views.numel())
    if strength.shape != (count, count) or intersect.shape != strength.shape or visibility.shape != strength.shape:
        raise ValueError("cycle authority axes differ")
    labels = torch.full((count, count), -1, dtype=torch.int8)
    for left in range(count):
        for view in torch.unique(views):
            candidates = torch.where(views == view)[0]
            if int(view) == int(views[left]) or candidates.numel() == 0:
                continue
            values = strength[left, candidates]
            best = int(candidates[torch.argmax(values)])
            if float(strength[left, best]) <= 0:
                continue
            reverse_candidates = torch.where(views == views[left])[0]
            reverse = int(reverse_candidates[torch.argmax(strength[best, reverse_candidates])])
            if reverse == left:
                labels[left, best] = labels[best, left] = 1
    for left in range(count):
        for right in range(left + 1, count):
            if views[left] == views[right] or labels[left, right] == 1:
                continue
            if intersect[left, right] == 0 and visibility[left, right] > 0 and visibility[right, left] > 0:
                labels[left, right] = labels[right, left] = 0
    return labels


def build(args: argparse.Namespace) -> dict:
    membership_path = Path(args.membership).expanduser().resolve()
    teacher_path = Path(args.teacher).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"track authority exists: {output}")
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    rows = torch.as_tensor(membership["row_indices"]).long()
    props = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    areas = torch.as_tensor(membership["proposal_area_fraction"]).float()
    observed = torch.as_tensor(membership["view_observed"]).bool()
    proposal_count = int(membership["num_proposals"])
    keep = weights >= 0.5
    incidence = sparse.coo_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float32),
            (props[keep].numpy(), rows[keep].numpy()),
        ),
        shape=(proposal_count, int(membership["num_rows"])),
    ).tocsr()
    incidence.sum_duplicates(); incidence.data[:] = 1.0
    intersection_np = (incidence @ incidence.T).toarray().astype(np.int32)
    sizes = np.asarray(incidence.sum(axis=1)).reshape(-1)
    union = sizes[:, None] + sizes[None, :] - intersection_np
    minimum = np.minimum(sizes[:, None], sizes[None, :])
    jaccard_np = np.divide(intersection_np, union, out=np.zeros_like(union, dtype=np.float32), where=union > 0)
    overlap_np = np.divide(intersection_np, minimum, out=np.zeros_like(minimum, dtype=np.float32), where=minimum > 0)
    observed_np = observed.numpy().T.astype(np.float32, copy=False)
    proposal_visible_count = np.asarray(incidence @ observed_np)
    proposal_visibility = np.divide(
        proposal_visible_count, sizes[:, None],
        out=np.zeros_like(proposal_visible_count, dtype=np.float32),
        where=sizes[:, None] > 0,
    )
    view_np = views.numpy()
    visibility_np = proposal_visibility[:, view_np]
    intersection = torch.from_numpy(intersection_np)
    jaccard = torch.from_numpy(jaccard_np)
    overlap = torch.from_numpy(overlap_np)
    visibility = torch.from_numpy(visibility_np)
    strength = torch.maximum(jaccard, overlap)
    labels = reciprocal_cycle_labels(strength, views, intersection, visibility)
    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    lefts, rights = torch.triu_indices(proposal_count, proposal_count, offset=1)
    cross = views[lefts] != views[rights]
    lefts, rights = lefts[cross], rights[cross]
    feature = torch.stack((
        jaccard[lefts, rights], overlap[lefts, rights],
        (descriptors[lefts] * descriptors[rights]).sum(-1),
        (contexts[lefts] * contexts[rights]).sum(-1),
        -torch.abs(torch.log2(areas[lefts].clamp_min(1e-8) / areas[rights].clamp_min(1e-8))),
        visibility[lefts, rights], visibility[rights, lefts],
    ), dim=1)
    edge_label = labels[lefts, rights]
    edge_weight = torch.where(
        edge_label == 1, strength[lefts, rights].clamp_min(1e-6),
        torch.minimum(visibility[lefts, rights], visibility[rights, lefts]).clamp_min(1e-6),
    )
    # Per seed/target-view outcome: matched proposal, visible-null, or occluded.
    outcome = torch.full((proposal_count, observed.shape[0]), -1, dtype=torch.int8)
    matched = torch.full_like(outcome, -1, dtype=torch.long)
    for proposal in range(proposal_count):
        outcome[proposal, int(views[proposal])] = 1; matched[proposal, int(views[proposal])] = proposal
        for view in range(observed.shape[0]):
            if view == int(views[proposal]): continue
            visible = bool(proposal_visibility[proposal, view] > 0)
            if not visible:
                outcome[proposal, view] = -1  # occluded/unknown
                continue
            same = torch.where((views == view) & (labels[proposal] == 1))[0]
            if same.numel():
                chosen = int(same[torch.argmax(strength[proposal, same])]); outcome[proposal, view] = 1; matched[proposal, view] = chosen
            # Visible proposal miss or granularity conflict is unknown, not null.
    payload = {
        "schema": SCHEMA, "schema_version": 1, "scene": str(args.scene),
        "edge_left": lefts, "edge_right": rights, "edge_features": feature,
        "edge_label": edge_label, "edge_weight": edge_weight,
        "proposal_views": views, "proposal_area_fraction": areas,
        "view_outcome": outcome, "matched_proposal": matched,
        "feature_names": ["jaccard", "minimum_overlap", "masked_descriptor_cosine", "context_descriptor_cosine", "negative_absolute_log2_area_ratio", "left_visible_in_right_view", "right_visible_in_left_view"],
        "metadata": {
            "source_only": True, "benchmark_masks_opened": False, "evaluation_rgb_opened": False,
            "same": "reciprocal_best_positive_3D_overlap_cycle", "different": "zero_3D_overlap_with_bidirectional_positive_visibility",
            "unknown": "occlusion_missing_visibility_or_nonreciprocal_granularity_conflict",
            "formal_stage_a_complete": False,
            "authority_limit": "conservative_geometry_cycle_proxy_not_external_video_or_human_physical_object_track;no_explicit_null_labels_available",
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "teacher": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp"); torch.save(payload, temporary); os.replace(temporary, output)
    report = {"schema": SCHEMA, "status": "complete", "scene": str(args.scene), "proposals": proposal_count,
              "same_edges": int((edge_label == 1).sum()), "different_edges": int((edge_label == 0).sum()), "unknown_edges": int((edge_label == -1).sum()),
              "matched_outcomes": int((outcome == 1).sum()), "visible_null_outcomes": int((outcome == 0).sum()), "occluded_unknown_outcomes": int((outcome == -1).sum()),
              "formal_stage_a_complete": False, "output": str(output), "output_sha256": sha256_file(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--scene", required=True); parser.add_argument("--membership", required=True); parser.add_argument("--teacher", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
