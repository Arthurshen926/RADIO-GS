#!/usr/bin/env python3
"""Evaluate correspondence-conditioned probabilistic Gaussian object support.

Frozen DINO transport supplies cross-view identity candidates.  Exact-MPR SAM
memberships remain continuous: no membership threshold is applied.  Only
reciprocal best identity pairs receive a range score; every other pair remains
unknown.  This preserves the precision role of correspondence while avoiding
the zero-coverage failure caused by hard Gaussian-support intersection.
"""

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
from radio_gs.scripts.evaluate_lerf_source_conservative_gaussian_core import (
    reciprocal_best,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_source_probabilistic_gaussian_support.v1"


def sparse_fuzzy_containment(
    incidence: sparse.csr_matrix, pairs: torch.Tensor
) -> torch.Tensor:
    """Return sum(min(a,b))/min(sum(a),sum(b)) for requested sparse rows."""

    matrix = incidence.tocsr()
    if matrix.ndim != 2 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("probabilistic support axes differ")
    masses = np.asarray(matrix.sum(axis=1)).reshape(-1)
    output = np.zeros(pairs.shape[0], dtype=np.float32)
    for position, (left, right) in enumerate(pairs.cpu().numpy().tolist()):
        first, second = matrix.getrow(int(left)), matrix.getrow(int(right))
        overlap = first.minimum(second).sum()
        denominator = min(float(masses[left]), float(masses[right]))
        if denominator > 0:
            output[position] = float(overlap) / denominator
    return torch.from_numpy(output).clamp(0, 1)


def probabilistic_reciprocal_score(
    probability: torch.Tensor,
    views: torch.Tensor,
    incidence: sparse.csr_matrix,
) -> torch.Tensor:
    """Lower envelope of reciprocal identity and continuous range support."""

    reciprocal = reciprocal_best(probability, views)
    pairs = torch.nonzero(torch.triu(reciprocal, diagonal=1), as_tuple=False)
    score = torch.zeros_like(probability)
    if pairs.numel():
        support = sparse_fuzzy_containment(incidence, pairs)
        values = torch.minimum(probability[pairs[:, 0], pairs[:, 1]], support)
        score[pairs[:, 0], pairs[:, 1]] = values
        score[pairs[:, 1], pairs[:, 0]] = values
    return score


def _pair_metrics(score: torch.Tensor, label: torch.Tensor, threshold: float) -> dict:
    accepted = score >= float(threshold)
    positive = label == 1
    true_positive = int((accepted & positive).sum())
    return {
        "threshold": float(threshold),
        "accepted": int(accepted.sum()),
        "purity": true_positive / max(int(accepted.sum()), 1),
        "positive_coverage": true_positive / max(int(positive.sum()), 1),
    }


def build(args: argparse.Namespace) -> dict:
    authority_path = Path(args.authority).resolve()
    membership_path = Path(args.membership).resolve()
    calibrator_path = Path(args.calibrator).resolve()
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"probabilistic support authority exists: {output}")
    authority = torch.load(authority_path, map_location="cpu", weights_only=False)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    calibrator = torch.load(calibrator_path, map_location="cpu", weights_only=False)
    if authority.get("feature_names") != calibrator.get("feature_names"):
        raise ValueError("authority/calibrator features differ")
    views = torch.as_tensor(authority["proposal_views"]).long()
    count = int(views.numel())
    if int(membership["num_proposals"]) != count:
        raise ValueError("authority/membership proposal domains differ")
    left = torch.as_tensor(authority["edge_left"]).long()
    right = torch.as_tensor(authority["edge_right"]).long()
    label = torch.as_tensor(authority["edge_label"]).long()
    logit = _calibrated_association_logit(
        torch.as_tensor(authority["edge_features"]).float(), calibrator
    )
    probability = torch.full((count, count), 0.5)
    probability[left, right] = probability[right, left] = torch.sigmoid(logit)
    probability.fill_diagonal_(1.0)
    rows = torch.as_tensor(membership["row_indices"]).long().numpy()
    proposals = torch.as_tensor(membership["proposal_indices"]).long().numpy()
    weights = torch.as_tensor(membership["weights"]).float().clamp(0, 1).numpy()
    incidence = sparse.coo_matrix(
        (weights, (proposals, rows)),
        shape=(count, int(membership["num_rows"])),
    ).tocsr()
    incidence.sum_duplicates()
    score = probabilistic_reciprocal_score(probability, views, incidence)
    heldout = (
        ((views[left] % 4 == 3) | (views[right] % 4 == 3)) & (label >= 0)
    )
    heldout_label = label[heldout]
    heldout_score = score[left[heldout], right[heldout]]
    thresholds = (1e-8, 0.1, 0.2, 0.3, 0.4, 0.5)
    pareto = [_pair_metrics(heldout_score, heldout_label, value) for value in thresholds]
    edge_score = score[left, right]
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": args.scene,
        "edge_left": left,
        "edge_right": right,
        "probabilistic_core_score": edge_score.half(),
        "core_known": edge_score > 0,
        "metadata": {
            "source_only": True,
            "figurines_opened": False,
            "identity_rule": "DINO-transport-calibrated reciprocal best",
            "range_rule": "sum_min_continuous_exact_MPR_membership_over_minimum_membership_mass",
            "unknown": "nonreciprocal identity or zero continuous shared support",
            "membership_threshold_applied": False,
            "authority": {"path": str(authority_path), "sha256": sha256_file(authority_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "calibrator": {"path": str(calibrator_path), "sha256": sha256_file(calibrator_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scene": args.scene,
        "heldout_known_edges": int(heldout.sum()),
        "heldout_same_edges": int((heldout_label == 1).sum()),
        "probabilistic_core_pareto": pareto,
        "core_known_edges_all": int((edge_score > 0).sum()),
        "figurines_opened": False,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
