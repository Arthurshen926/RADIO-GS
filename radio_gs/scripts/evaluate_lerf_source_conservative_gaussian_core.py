#!/usr/bin/env python3
"""Evaluate a conservative Gaussian-core track authority on source-only folds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from radio_gs.scripts.build_lerf_object_aware_visibility_track_posterior import (
    _calibrated_association_logit,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_source_conservative_gaussian_core.v1"


def wilson_lower_bound(success: torch.Tensor, total: torch.Tensor, z: float = 1.96) -> torch.Tensor:
    """Wilson lower confidence bound for a bounded shared-support fraction."""

    s, n = torch.as_tensor(success).float(), torch.as_tensor(total).float()
    p = s / n.clamp_min(1)
    z2 = float(z) ** 2
    center = p + z2 / (2 * n.clamp_min(1))
    radius = float(z) * torch.sqrt((p * (1 - p) + z2 / (4 * n.clamp_min(1))) / n.clamp_min(1))
    out = (center - radius) / (1 + z2 / n.clamp_min(1))
    return torch.where(n > 0, out.clamp_min(0), torch.zeros_like(out))


def reciprocal_best(probability: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
    """Directed per-target-view best matches that agree in both directions."""

    count = int(views.numel()); best = torch.zeros((count, count), dtype=torch.bool)
    for left in range(count):
        for view in torch.unique(views):
            if int(view) == int(views[left]): continue
            candidates = torch.where(views == view)[0]
            if candidates.numel():
                right = int(candidates[torch.argmax(probability[left, candidates])])
                if probability[left, right] >= 0.5: best[left, right] = True
    return best & best.T


def triangle_core_score(
    probability: torch.Tensor, geometry_lcb: torch.Tensor, views: torch.Tensor
) -> torch.Tensor:
    """Keep only reciprocal complete triangles spanning three source views."""

    reciprocal = reciprocal_best(probability, views)
    count = int(views.numel()); score = torch.zeros((count, count))
    for left in range(count):
        for right in torch.where(reciprocal[left] & (torch.arange(count) > left))[0].tolist():
            third = torch.where(reciprocal[left] & reciprocal[right] &
                                (views != views[left]) & (views != views[right]))[0]
            if not third.numel(): continue
            values = torch.stack([
                torch.minimum(torch.minimum(probability[left, right], probability[left, third]), probability[right, third]),
                torch.minimum(torch.minimum(geometry_lcb[left, right], geometry_lcb[left, third]), geometry_lcb[right, third]),
            ]).amin(0)
            best = float(values.max())
            score[left, right] = score[right, left] = best
    return score


def reciprocal_core_score(
    probability: torch.Tensor, geometry_lcb: torch.Tensor, views: torch.Tensor
) -> torch.Tensor:
    """Two-view conservative endpoint: reciprocal identity plus Gaussian LCB."""

    reciprocal = reciprocal_best(probability, views)
    return torch.where(
        reciprocal,
        torch.minimum(probability, geometry_lcb),
        torch.zeros_like(probability),
    )


def _pair_metrics(score: torch.Tensor, label: torch.Tensor, threshold: float) -> dict:
    accepted = score >= float(threshold); positive = label == 1
    true_positive = int((accepted & positive).sum()); accepted_count = int(accepted.sum())
    return {"threshold": threshold, "accepted": accepted_count,
            "purity": true_positive / max(accepted_count, 1),
            "positive_coverage": true_positive / max(int(positive.sum()), 1)}


def build(args: argparse.Namespace) -> dict:
    authority_path = Path(args.authority).resolve(); membership_path = Path(args.membership).resolve()
    calibrator_path = Path(args.calibrator).resolve(); output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists(): raise FileExistsError(f"core authority exists: {output}")
    authority = torch.load(authority_path, map_location="cpu", weights_only=False)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    calibrator = torch.load(calibrator_path, map_location="cpu", weights_only=False)
    if authority.get("feature_names") != calibrator.get("feature_names"):
        raise ValueError("authority/calibrator features differ")
    views = torch.as_tensor(authority["proposal_views"]).long(); count = int(views.numel())
    left = torch.as_tensor(authority["edge_left"]).long(); right = torch.as_tensor(authority["edge_right"]).long()
    label = torch.as_tensor(authority["edge_label"]).long()
    logit = _calibrated_association_logit(torch.as_tensor(authority["edge_features"]).float(), calibrator)
    probability = torch.full((count, count), 0.5); probability[left, right] = probability[right, left] = torch.sigmoid(logit)
    probability.fill_diagonal_(1.0)
    rows = torch.as_tensor(membership["row_indices"]).long(); props = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float(); keep = weights >= 0.5
    incidence = sparse.coo_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                                   (props[keep].numpy(), rows[keep].numpy())),
                                  shape=(count, int(membership["num_rows"]))).tocsr()
    incidence.sum_duplicates(); incidence.data[:] = 1
    intersection = torch.from_numpy((incidence @ incidence.T).toarray().astype(np.float32))
    sizes = torch.from_numpy(np.asarray(incidence.sum(axis=1)).reshape(-1).astype(np.float32))
    minimum = torch.minimum(sizes[:, None], sizes[None, :])
    geometry_lcb = wilson_lower_bound(intersection, minimum)
    mutual_score = reciprocal_core_score(probability, geometry_lcb, views)
    core_score = triangle_core_score(probability, geometry_lcb, views)
    heldout = ((views[left] % 4 == 3) | (views[right] % 4 == 3)) & (label >= 0)
    heldout_label = label[heldout]; heldout_soft = probability[left[heldout], right[heldout]]
    heldout_mutual = mutual_score[left[heldout], right[heldout]]
    heldout_core = core_score[left[heldout], right[heldout]]
    thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
    soft_pareto = [_pair_metrics(heldout_soft, heldout_label, value) for value in thresholds]
    mutual_pareto = [_pair_metrics(heldout_mutual, heldout_label, value) for value in (1e-8, 0.3, 0.4, 0.5, 0.6)]
    # Core score is a product-free lower envelope; positive support alone is
    # meaningful, then the same fixed probability thresholds are reported.
    core_pareto = [_pair_metrics(heldout_core, heldout_label, value) for value in (1e-8,) + thresholds]
    payload = {"schema": SCHEMA, "schema_version": 1, "scene": args.scene,
               "edge_left": left, "edge_right": right, "core_score": core_score[left, right].half(),
               "mutual_core_score": mutual_score[left, right].half(),
               "core_known": (core_score[left, right] > 0), "geometry_lcb": geometry_lcb[left, right].half(),
               "metadata": {"source_only": True, "figurines_opened": False,
                            "boundary_and_single_view_shell": "unknown",
                            "core_rule": "reciprocal_best_complete_three_view_triangle_and_minimum_Wilson95_Gaussian_support_LCB",
                            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
                            "authority": {"path": str(authority_path), "sha256": sha256_file(authority_path)},
                            "calibrator": {"path": str(calibrator_path), "sha256": sha256_file(calibrator_path)}}}
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, output)
    report = {"schema": SCHEMA, "status": "complete", "scene": args.scene,
              "heldout_known_edges": int(heldout.sum()), "heldout_same_edges": int((heldout_label == 1).sum()),
              "soft_track_pareto": soft_pareto, "reciprocal_core_pareto": mutual_pareto,
              "conservative_core_pareto": core_pareto,
              "core_known_edges_all": int((core_score[left, right] > 0).sum()),
              "figurines_opened": False, "output": str(output), "output_sha256": sha256_file(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--scene", required=True); parser.add_argument("--authority", required=True); parser.add_argument("--membership", required=True); parser.add_argument("--calibrator", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
