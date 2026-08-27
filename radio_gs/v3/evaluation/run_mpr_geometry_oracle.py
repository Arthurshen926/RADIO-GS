#!/usr/bin/env python3
"""Execute the source-dev exact-MPR/frozen-geometry oracle ladder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.mpr_geometry_oracle import (
    exact_mpr_membership,
    positive_hit_coverage,
    render_exact_membership,
    score_oracle_prediction,
    union_memberships,
)
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.run_instance_upper_bound import load_episodes


def _macro(metrics: list[object]) -> dict[str, float]:
    if not metrics:
        raise ValueError("oracle cohort is empty")
    names = ("mask_iou", "brier", "boundary_f", "unknown_fp_mass")
    return {
        name: sum(float(getattr(item, name)) for item in metrics) / len(metrics)
        for name in names
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.evaluation_split != "dev":
        raise ValueError("MPR/geometry diagnosis may open source dev only")
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    validate_source_only_inputs(membership, relation)
    episodes, _supports = load_episodes(membership, relation)
    by_proposal = {item.proposal_index: item for item in episodes}
    train = {
        item.proposal_index for item in episodes if item.view_index % 4 in (1, 2)
    }
    dev = [item for item in episodes if item.view_index % 4 == 3]
    num_gaussians = int(membership["num_rows"])
    lifted = {
        index: exact_mpr_membership(item, num_gaussians)
        for index, item in by_proposal.items()
        if index in train or item.view_index % 4 == 3
    }
    edge_left = torch.as_tensor(relation["edge_left"]).long()
    edge_right = torch.as_tensor(relation["edge_right"]).long()
    edge_relation = torch.as_tensor(relation["edge_relation"]).to(torch.int8)
    same_peers: dict[int, list[int]] = {}
    for left, right, label in zip(edge_left.tolist(), edge_right.tolist(), edge_relation.tolist()):
        if label != 1:
            continue
        if left in train and right in by_proposal and by_proposal[right].view_index % 4 == 3:
            same_peers.setdefault(right, []).append(left)
        if right in train and left in by_proposal and by_proposal[left].view_index % 4 == 3:
            same_peers.setdefault(left, []).append(right)

    roundtrip_metrics = []
    transfer_metrics = []
    roundtrip_coverage: list[float] = []
    transfer_coverage: list[float] = []
    records = []
    for episode in dev:
        if not bool((episode.known & ~episode.target).any()):
            continue
        own = lifted[episode.proposal_index]
        own_prediction = render_exact_membership(own, episode)
        own_metrics = score_oracle_prediction(own_prediction, episode)
        peers = sorted(set(same_peers.get(episode.proposal_index, [])))
        if not peers:
            continue
        transferred = union_memberships([lifted[index] for index in peers])
        transfer_prediction = render_exact_membership(transferred, episode)
        transfer_score = score_oracle_prediction(transfer_prediction, episode)
        own_coverage = positive_hit_coverage(own, episode)
        cross_coverage = positive_hit_coverage(transferred, episode)
        roundtrip_metrics.append(own_metrics)
        transfer_metrics.append(transfer_score)
        roundtrip_coverage.append(own_coverage)
        transfer_coverage.append(cross_coverage)
        records.append({
            "proposal_index": episode.proposal_index,
            "view_index": episode.view_index,
            "train_same_peers": peers,
            "roundtrip": asdict(own_metrics),
            "cross_view_transfer": asdict(transfer_score),
            "roundtrip_positive_hit_coverage": own_coverage,
            "transfer_positive_hit_coverage": cross_coverage,
        })
    report = {
        "schema": "radio_gs.sugm_v3.mpr_geometry_oracle.v1",
        "evaluation_split": "dev",
        "source_audit_opened": False,
        "benchmark_rgb_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
        "aggregation": "proposal_macro_fixed_probability_threshold_0.5",
        "cohort": {"episodes": len(records), "num_gaussians": num_gaussians},
        "same_view_roundtrip": {
            "metrics": _macro(roundtrip_metrics),
            "positive_hit_coverage": sum(roundtrip_coverage) / len(roundtrip_coverage),
        },
        "cross_view_known_same_union": {
            "metrics": _macro(transfer_metrics),
            "positive_hit_coverage": sum(transfer_coverage) / len(transfer_coverage),
        },
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
        },
        "records": records,
    }
    write_frozen_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--evaluation-split", choices=("dev",), default="dev")
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
